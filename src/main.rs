#![forbid(unsafe_code)]

mod brain;

use brain::Brain;
use serde::Deserialize;
use serde_json::json;
use std::env;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

const MAX_BODY_BYTES: usize = 1_048_576;
const MAX_REQUEST_LINE_BYTES: usize = 8 * 1024;
const MAX_HEADER_LINE_BYTES: usize = 8 * 1024;
const MAX_HEADER_BYTES: usize = 32 * 1024;
const MAX_HEADER_COUNT: usize = 100;
const MAX_CONCURRENT_CONNECTIONS: usize = 64;

#[derive(Debug, Deserialize)]
struct ControlCommand {
    action: String,
    region: Option<String>,
    intensity: Option<f32>,
    value: Option<f32>,
    cue: Option<String>,
    target: Option<String>,
    query: Option<String>,
    paused: Option<bool>,
}

fn main() -> std::io::Result<()> {
    let bind = env::var("BRAINLAB_BIND").unwrap_or_else(|_| "127.0.0.1:8787".to_owned());
    let listener = TcpListener::bind(&bind)?;
    let requested_scale = env::var("BRAINLAB_SCALE")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(1.0);
    let brain = Arc::new(Mutex::new(Brain::with_scale(requested_scale)));
    let initial_state = brain
        .lock()
        .map(|model| model.snapshot())
        .unwrap_or_else(|_| serde_json::Value::Null);

    {
        let simulation = Arc::clone(&brain);
        thread::Builder::new()
            .name("cortex-simulation".to_owned())
            .spawn(move || loop {
                let started = Instant::now();
                let mut speed = 1.0f32;
                if let Ok(mut model) = simulation.lock() {
                    speed = model.clock_speed();
                    model.step();
                }
                let target = Duration::from_secs_f32((0.001 / speed.clamp(0.25, 8.0)).max(0.000125));
                if let Some(remaining) = target.checked_sub(started.elapsed()) {
                    thread::sleep(remaining);
                }
            })?;
    }

    println!(
        "\n╭────────────────────────────────────────────────────╮\n│  CORTEX LAB · brain-inspired systems workbench     │\n╰────────────────────────────────────────────────────╯\n  Dashboard : http://{bind}\n  Model     : {} neurons · {} synapses\n  Scale     : {:.2}× baseline (BRAINLAB_SCALE)\n  Safety    : localhost-only by default; no telemetry\n  Stop      : Ctrl+C\n",
        initial_state["meta"]["neurons"],
        initial_state["meta"]["synapses"],
        initial_state["meta"]["scale"].as_f64().unwrap_or(1.0)
    );
    let active_connections = Arc::new(AtomicUsize::new(0));
    for incoming in listener.incoming() {
        match incoming {
            Ok(mut stream) => {
                let active = active_connections.fetch_add(1, Ordering::AcqRel);
                if active >= MAX_CONCURRENT_CONNECTIONS {
                    active_connections.fetch_sub(1, Ordering::Release);
                    let _ = stream.set_write_timeout(Some(Duration::from_secs(1)));
                    let _ = respond(
                        &mut stream,
                        "503 Service Unavailable",
                        "application/json; charset=utf-8",
                        br#"{"ok":false,"error":"Server is at its connection limit. Retry shortly."}"#,
                    );
                    continue;
                }

                let shared = Arc::clone(&brain);
                let connection_counter = Arc::clone(&active_connections);
                if let Err(error) = thread::Builder::new()
                    .name("cortex-http".to_owned())
                    .spawn(move || {
                        let _guard = ConnectionGuard(connection_counter);
                        if let Err(error) = handle_connection(stream, shared) {
                            eprintln!("HTTP request failed: {error}");
                        }
                    })
                {
                    active_connections.fetch_sub(1, Ordering::Release);
                    eprintln!("Could not start HTTP worker: {error}");
                }
            }
            Err(error) => eprintln!("Connection accept failed: {error}"),
        }
    }
    Ok(())
}

struct ConnectionGuard(Arc<AtomicUsize>);

impl Drop for ConnectionGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Release);
    }
}

fn read_http_line(reader: &mut BufReader<TcpStream>, limit: usize) -> std::io::Result<Option<String>> {
    let mut bytes = Vec::with_capacity(128);
    let read = reader
        .take(limit.saturating_add(1) as u64)
        .read_until(b'\n', &mut bytes)?;
    if read == 0 {
        return Ok(None);
    }
    if bytes.len() > limit {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "HTTP line exceeds the configured limit",
        ));
    }
    if bytes.last() != Some(&b'\n') {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "HTTP line is not terminated",
        ));
    }
    String::from_utf8(bytes)
        .map(Some)
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "HTTP line is not UTF-8"))
}

fn parse_request_line(line: &str) -> Option<(&str, &str, &str)> {
    let line = line.trim_end_matches(|character| character == '\r' || character == '\n');
    let mut parts = line.split_whitespace();
    let method = parts.next()?;
    let target = parts.next()?;
    let version = parts.next()?;
    if parts.next().is_some() || method.is_empty() || !target.starts_with('/') {
        return None;
    }
    Some((method, target, version))
}

fn bad_request(stream: &mut TcpStream, status: &str, message: &'static str) -> std::io::Result<()> {
    let payload = json!({"ok": false, "error": message}).to_string();
    respond(stream, status, "application/json; charset=utf-8", payload.as_bytes())
}

fn handle_connection(mut stream: TcpStream, brain: Arc<Mutex<Brain>>) -> std::io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    stream.set_write_timeout(Some(Duration::from_secs(5)))?;
    let mut reader = BufReader::new(stream.try_clone()?);

    let request_line = match read_http_line(&mut reader, MAX_REQUEST_LINE_BYTES) {
        Ok(Some(line)) => line,
        Ok(None) => return Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::InvalidData => {
            return bad_request(&mut stream, "414 URI Too Long", "Request line is invalid or too long.");
        }
        Err(error) => return Err(error),
    };
    let (method, raw_target, version) = match parse_request_line(&request_line) {
        Some(parts) => parts,
        None => return bad_request(&mut stream, "400 Bad Request", "Malformed HTTP request line."),
    };
    if version != "HTTP/1.0" && version != "HTTP/1.1" {
        return bad_request(&mut stream, "505 HTTP Version Not Supported", "Only HTTP/1.0 and HTTP/1.1 are supported.");
    }

    let path = raw_target.split('?').next().unwrap_or("/");
    let mut content_length = None;
    let mut transfer_encoding = false;
    let mut total_header_bytes = 0usize;
    let mut header_count = 0usize;

    loop {
        let header = match read_http_line(&mut reader, MAX_HEADER_LINE_BYTES) {
            Ok(Some(line)) => line,
            Ok(None) => return bad_request(&mut stream, "400 Bad Request", "Headers ended before the terminating blank line."),
            Err(error) if error.kind() == std::io::ErrorKind::InvalidData => {
                return bad_request(&mut stream, "431 Request Header Fields Too Large", "A request header is invalid or too large.");
            }
            Err(error) => return Err(error),
        };
        total_header_bytes = total_header_bytes.saturating_add(header.len());
        header_count = header_count.saturating_add(1);
        if total_header_bytes > MAX_HEADER_BYTES || header_count > MAX_HEADER_COUNT {
            return bad_request(&mut stream, "431 Request Header Fields Too Large", "Request headers exceed the configured limit.");
        }

        let line = header.trim_end_matches(|character| character == '\r' || character == '\n');
        if line.is_empty() {
            break;
        }
        let (name, value) = match line.split_once(':') {
            Some(parts) => parts,
            None => return bad_request(&mut stream, "400 Bad Request", "Malformed request header."),
        };
        let name = name.trim();
        if name.is_empty() || !name.bytes().all(|byte| byte.is_ascii_alphanumeric() || byte == b'-') {
            return bad_request(&mut stream, "400 Bad Request", "Invalid request header name.");
        }
        let value = value.trim();
        if name.eq_ignore_ascii_case("content-length") {
            if content_length.is_some() {
                return bad_request(&mut stream, "400 Bad Request", "Duplicate Content-Length headers are not accepted.");
            }
            let parsed = match value.parse::<usize>() {
                Ok(length) => length,
                Err(_) => return bad_request(&mut stream, "400 Bad Request", "Content-Length must be a non-negative integer."),
            };
            if parsed > MAX_BODY_BYTES {
                return bad_request(&mut stream, "413 Payload Too Large", "Request body exceeds 1 MiB.");
            }
            content_length = Some(parsed);
        } else if name.eq_ignore_ascii_case("transfer-encoding") && !value.is_empty() {
            transfer_encoding = true;
        }
    }

    if transfer_encoding && content_length.is_some() {
        return bad_request(&mut stream, "400 Bad Request", "Transfer-Encoding and Content-Length cannot be combined.");
    }
    if transfer_encoding {
        return bad_request(&mut stream, "501 Not Implemented", "Transfer-Encoding is not supported; use Content-Length.");
    }

    let content_length = content_length.unwrap_or(0);
    if content_length > 0 && !(method == "POST" && path == "/api/control") {
        return bad_request(&mut stream, "400 Bad Request", "This route does not accept a request body.");
    }
    let mut body = vec![0u8; content_length];
    if let Err(error) = reader.read_exact(&mut body) {
        if error.kind() == std::io::ErrorKind::UnexpectedEof {
            return bad_request(&mut stream, "400 Bad Request", "Request body ended before Content-Length bytes were received.");
        }
        return Err(error);
    }

    match (method, path) {
        ("GET", "/") => respond(
            &mut stream,
            "200 OK",
            "text/html; charset=utf-8",
            include_str!("../web/index.html").as_bytes(),
        ),
        ("GET", "/healthz") => respond(
            &mut stream,
            "200 OK",
            "application/json; charset=utf-8",
            br#"{"ok":true,"service":"cortex-lab"}"#,
        ),
        ("GET", "/api/state") => {
            let state = match brain.lock() {
                Ok(model) => model.snapshot(),
                Err(_) => return respond(
                    &mut stream,
                    "500 Internal Server Error",
                    "application/json; charset=utf-8",
                    br#"{"ok":false,"error":"Simulation state unavailable."}"#,
                ),
            };
            match serde_json::to_vec(&state) {
                Ok(bytes) => respond(&mut stream, "200 OK", "application/json; charset=utf-8", &bytes),
                Err(error) => {
                    let payload = json!({"ok":false,"error":format!("Serialization failed: {error}")}).to_string();
                    respond(&mut stream, "500 Internal Server Error", "application/json; charset=utf-8", payload.as_bytes())
                }
            }
        }
        ("POST", "/api/control") => control(&mut stream, &brain, &body),
        _ => respond(
            &mut stream,
            "404 Not Found",
            "application/json; charset=utf-8",
            br#"{"ok":false,"error":"Route not found."}"#,
        ),
    }
}
fn control(stream: &mut TcpStream, brain: &Arc<Mutex<Brain>>, body: &[u8]) -> std::io::Result<()> {
    let command: ControlCommand = match serde_json::from_slice(body) {
        Ok(command) => command,
        Err(error) => {
            let payload = json!({"ok":false,"error":format!("Invalid JSON command: {error}")}).to_string();
            return respond(stream, "400 Bad Request", "application/json; charset=utf-8", payload.as_bytes());
        }
    };

    let mut model = match brain.lock() {
        Ok(model) => model,
        Err(_) => return respond(stream, "500 Internal Server Error", "application/json; charset=utf-8",
            br#"{"ok":false,"error":"Simulation lock is poisoned."}"#),
    };

    let result: Result<String, String> = match command.action.as_str() {
        "stimulate" => {
            let region = command.region.as_deref().unwrap_or("Visual cortex");
            let intensity = command.intensity.unwrap_or(18.0);
            model.stimulate(region, intensity).map(|()| format!("Stimulus delivered to {region}."))
        }
        "reward" => {
            model.reward(command.value.unwrap_or(0.65));
            Ok("Reward signal delivered.".to_owned())
        }
        "stress" => {
            model.induce_stress(command.value.unwrap_or(0.6));
            Ok("Arousal / stress modulation applied.".to_owned())
        }
        "novelty" => {
            model.novelty();
            Ok("Novelty and encoding gates elevated.".to_owned())
        }
        "pause" => {
            model.set_paused(command.paused.unwrap_or(true));
            Ok(if command.paused.unwrap_or(true) { "Simulation paused.".to_owned() } else { "Simulation resumed.".to_owned() })
        }
        "toggle_pause" => {
            model.toggle_pause();
            Ok("Simulation state toggled.".to_owned())
        }
        "speed" => {
            let speed = command.value.unwrap_or(1.0);
            model.set_speed(speed);
            Ok(format!("Clock speed set to {:.2}×.", speed.clamp(0.25, 8.0)))
        }
        "plasticity" => {
            model.toggle_plasticity();
            Ok("Plasticity state toggled.".to_owned())
        }
        "associate" => {
            let cue = command.cue.as_deref().or(command.query.as_deref()).unwrap_or("");
            let target = command.target.as_deref().unwrap_or("");
            model.associate(cue, target).map(|()| "Associative trace encoded.".to_owned())
        }
        "recall" => {
            let query = command.query.as_deref().or(command.cue.as_deref()).unwrap_or("");
            Ok(model.recall(query))
        }
        "sleep" => {
            model.sleep_consolidation();
            Ok("Offline replay / consolidation pass completed.".to_owned())
        }
        "reset" => {
            let scale = model.scale();
            *model = Brain::with_scale(scale);
            Ok("Network and experiment state reset.".to_owned())
        }
        _ => Err(format!("Unknown action: {}", command.action)),
    };

    let response = match result {
        Ok(message) => json!({"ok":true,"message":message}),
        Err(error) => json!({"ok":false,"error":error}),
    };
    let status = if response["ok"].as_bool() == Some(true) { "200 OK" } else { "400 Bad Request" };
    respond(stream, status, "application/json; charset=utf-8", response.to_string().as_bytes())
}

fn respond(stream: &mut TcpStream, status: &str, content_type: &str, body: &[u8]) -> std::io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nContent-Security-Policy: default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:;\r\n\r\n",
        body.len()
    )?;
    stream.write_all(body)?;
    stream.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_line_parser_accepts_only_origin_form_and_known_http_versions_are_checked_by_handler() {
        assert_eq!(
            parse_request_line("GET /api/state?fresh=1 HTTP/1.1\r\n"),
            Some(("GET", "/api/state?fresh=1", "HTTP/1.1"))
        );
        assert!(parse_request_line("GET /api/state HTTP/1.1 extra\r\n").is_none());
        assert!(parse_request_line("GET https://example.com/ HTTP/1.1\r\n").is_none());
        assert!(parse_request_line("GET /api/state\r\n").is_none());
    }

    #[test]
    fn command_requires_an_action() {
        assert!(serde_json::from_str::<ControlCommand>("{}").is_err());
    }

    #[test]
    fn command_accepts_stimulus_payload() {
        let command: ControlCommand = serde_json::from_str(
            r#"{"action":"stimulate","region":"Hippocampus","intensity":16.5}"#,
        ).unwrap();
        assert_eq!(command.action, "stimulate");
        assert_eq!(command.region.as_deref(), Some("Hippocampus"));
        assert_eq!(command.intensity, Some(16.5));
    }
}

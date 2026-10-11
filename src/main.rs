#![forbid(unsafe_code)]
//! Cortex Lab server: fixed-step simulation thread plus a small hardened
//! std-only HTTP server that serves the dashboard and a JSON control API.
//!
//! Security posture: loopback-only by default, bounded request sizes, no
//! chunked encoding, connection cap, strict Content-Length handling, and
//! no-store responses. There is no authentication, so a LAN-facing
//! `BRAINLAB_BIND` intentionally exposes experiment controls.

mod brain;

use brain::Brain;
use serde::Deserialize;
use serde_json::json;
use std::env;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::{Duration, Instant};

const MAX_BODY_BYTES: usize = 1_048_576;
const MAX_REQUEST_LINE_BYTES: usize = 8 * 1024;
const MAX_HEADER_LINE_BYTES: usize = 8 * 1024;
const MAX_HEADER_BYTES: usize = 32 * 1024;
const MAX_HEADER_COUNT: usize = 100;
const MAX_CONCURRENT_CONNECTIONS: usize = 64;

/// Control envelope accepted on `POST /api/control`.
///
/// `enabled` gives callers an idempotent way to set plasticity explicitly;
/// when omitted the action toggles (backwards compatible).
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
    enabled: Option<bool>,
}

fn main() -> std::io::Result<()> {
    let bind = env::var("BRAINLAB_BIND").unwrap_or_else(|_| "127.0.0.1:8787".to_owned());
    if !is_loopback_bind(&bind) {
        eprintln!(
            "WARNING: BRAINLAB_BIND={bind} is not loopback-only. \
             The control API has no authentication; anyone who can reach this \
             address can stimulate, reward, and reset the experiment."
        );
    }
    let listener = TcpListener::bind(&bind)?;
    let requested_scale = match env::var("BRAINLAB_SCALE") {
        Ok(raw) => match raw.parse::<f32>() {
            Ok(value) => value,
            Err(_) => {
                eprintln!("WARNING: invalid BRAINLAB_SCALE={raw:?}; falling back to 1.0");
                1.0
            }
        },
        Err(_) => 1.0,
    };
    let brain = Arc::new(RwLock::new(Brain::with_scale(requested_scale)));
    let initial_state = brain
        .read()
        .map(|model| model.snapshot())
        .unwrap_or_else(|_| serde_json::Value::Null);

    {
        let simulation = Arc::clone(&brain);
        thread::Builder::new()
            .name("cortex-simulation".to_owned())
            .spawn(move || loop {
                let started = Instant::now();
                let mut speed = 1.0f32;
                if let Ok(mut model) = simulation.write() {
                    speed = model.clock_speed();
                    model.step();
                }
                let target =
                    Duration::from_secs_f32((0.001 / speed.clamp(0.25, 8.0)).max(0.000125));
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
            Ok(stream) => {
                let active = active_connections.fetch_add(1, Ordering::AcqRel);
                if active >= MAX_CONCURRENT_CONNECTIONS {
                    active_connections.fetch_sub(1, Ordering::Release);
                    let _ = stream.set_write_timeout(Some(Duration::from_secs(1)));
                    let mut stream = stream;
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
                if let Err(error) =
                    thread::Builder::new()
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

fn is_loopback_bind(bind: &str) -> bool {
    let host = bind.rsplit_once(':').map(|(h, _)| h).unwrap_or(bind);
    let host = host
        .trim()
        .trim_matches(|c| c == '[' || c == ']')
        .to_ascii_lowercase();
    host == "127.0.0.1" || host == "localhost" || host == "::1"
}

struct ConnectionGuard(Arc<AtomicUsize>);

impl Drop for ConnectionGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Release);
    }
}

fn read_http_line(
    reader: &mut BufReader<TcpStream>,
    limit: usize,
    deadline: Instant,
) -> std::io::Result<Option<String>> {
    let mut bytes = Vec::with_capacity(128);
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "HTTP request deadline exceeded",
            ));
        }
        reader.get_mut().set_read_timeout(Some(remaining))?;
        let available = reader.fill_buf()?;
        if available.is_empty() {
            if bytes.is_empty() {
                return Ok(None);
            }
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "HTTP line ended before a newline",
            ));
        }

        let chunk_len = available
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(available.len(), |index| index + 1);
        if bytes.len().saturating_add(chunk_len) > limit {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "HTTP line exceeds the configured limit",
            ));
        }
        let complete = available[chunk_len - 1] == b'\n';
        bytes.extend_from_slice(&available[..chunk_len]);
        reader.consume(chunk_len);
        if complete {
            return String::from_utf8(bytes).map(Some).map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "HTTP line is not UTF-8")
            });
        }
    }
}

fn read_http_body(
    reader: &mut BufReader<TcpStream>,
    body: &mut [u8],
    deadline: Instant,
) -> std::io::Result<()> {
    let mut filled = 0usize;
    while filled < body.len() {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "HTTP request deadline exceeded",
            ));
        }
        reader.get_mut().set_read_timeout(Some(remaining))?;
        let available = reader.fill_buf()?;
        if available.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "request body ended before Content-Length bytes were received",
            ));
        }
        let count = available.len().min(body.len() - filled);
        body[filled..filled + count].copy_from_slice(&available[..count]);
        reader.consume(count);
        filled += count;
    }
    Ok(())
}

fn is_valid_method(method: &str) -> bool {
    !method.is_empty() && method.len() <= 8 && method.bytes().all(|b| b.is_ascii_uppercase())
}

fn is_valid_target(target: &str) -> bool {
    if !target.starts_with('/') || target.len() > MAX_REQUEST_LINE_BYTES {
        return false;
    }
    // Origin-form only: no spaces, controls, or DEL. Query strings are
    // stripped later; fragments must never be sent to a server.
    target.bytes().all(|b| b >= 0x21 && b <= 0x7E) && !target.contains('#')
}

fn is_valid_header_value(value: &str) -> bool {
    // RFC 9110 field-value: visible ASCII, spaces and horizontal tabs.
    !value.bytes().any(|b| b == b'\r' || b == b'\n')
        && value
            .bytes()
            .all(|b| b == b'\t' || (0x20..=0x7E).contains(&b))
}

fn parse_request_line(line: &str) -> Option<(&str, &str, &str)> {
    let line = line.trim_end_matches(|character| character == '\r' || character == '\n');
    let mut parts = line.split_whitespace();
    let method = parts.next()?;
    let target = parts.next()?;
    let version = parts.next()?;
    if parts.next().is_some() || !is_valid_method(method) || !is_valid_target(target) {
        return None;
    }
    Some((method, target, version))
}

fn bad_request(stream: &mut TcpStream, status: &str, message: &'static str) -> std::io::Result<()> {
    let payload = json!({"ok": false, "error": message}).to_string();
    respond(
        stream,
        status,
        "application/json; charset=utf-8",
        payload.as_bytes(),
    )
}

fn method_not_allowed(stream: &mut TcpStream) -> std::io::Result<()> {
    let payload = json!({"ok": false, "error": "Method not allowed for this route."}).to_string();
    write!(
        stream,
        "HTTP/1.1 405 Method Not Allowed\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\nCache-Control: no-store\r\nAllow: GET, POST\r\n\r\n",
        payload.len()
    )?;
    stream.write_all(payload.as_bytes())?;
    stream.flush()
}

fn handle_connection(mut stream: TcpStream, brain: Arc<RwLock<Brain>>) -> std::io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    stream.set_write_timeout(Some(Duration::from_secs(5)))?;
    let mut reader = BufReader::new(stream.try_clone()?);
    // A total deadline prevents slow clients from keeping a worker occupied by trickling bytes.
    let request_deadline = Instant::now() + Duration::from_secs(10);

    let request_line = match read_http_line(&mut reader, MAX_REQUEST_LINE_BYTES, request_deadline) {
        Ok(Some(line)) => line,
        Ok(None) => return Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::InvalidData => {
            return bad_request(
                &mut stream,
                "414 URI Too Long",
                "Request line is invalid or too long.",
            );
        }
        Err(error) => return Err(error),
    };
    let (method, raw_target, version) = match parse_request_line(&request_line) {
        Some(parts) => parts,
        None => {
            return bad_request(
                &mut stream,
                "400 Bad Request",
                "Malformed HTTP request line.",
            )
        }
    };
    if version != "HTTP/1.0" && version != "HTTP/1.1" {
        return bad_request(
            &mut stream,
            "505 HTTP Version Not Supported",
            "Only HTTP/1.0 and HTTP/1.1 are supported.",
        );
    }

    let path = raw_target.split('?').next().unwrap_or("/");
    let mut content_length = None;
    let mut content_type: Option<String> = None;
    let mut transfer_encoding = false;
    let mut total_header_bytes = 0usize;
    let mut header_count = 0usize;

    loop {
        let header = match read_http_line(&mut reader, MAX_HEADER_LINE_BYTES, request_deadline) {
            Ok(Some(line)) => line,
            Ok(None) => {
                return bad_request(
                    &mut stream,
                    "400 Bad Request",
                    "Headers ended before the terminating blank line.",
                )
            }
            Err(error) if error.kind() == std::io::ErrorKind::InvalidData => {
                return bad_request(
                    &mut stream,
                    "431 Request Header Fields Too Large",
                    "A request header is invalid or too large.",
                );
            }
            Err(error) => return Err(error),
        };
        total_header_bytes = total_header_bytes.saturating_add(header.len());
        header_count = header_count.saturating_add(1);
        if total_header_bytes > MAX_HEADER_BYTES || header_count > MAX_HEADER_COUNT {
            return bad_request(
                &mut stream,
                "431 Request Header Fields Too Large",
                "Request headers exceed the configured limit.",
            );
        }

        let line = header.trim_end_matches(|character| character == '\r' || character == '\n');
        if line.is_empty() {
            break;
        }
        let (name, value) = match line.split_once(':') {
            Some(parts) => parts,
            None => {
                return bad_request(&mut stream, "400 Bad Request", "Malformed request header.")
            }
        };
        let name = name.trim();
        if name.is_empty()
            || !name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
        {
            return bad_request(
                &mut stream,
                "400 Bad Request",
                "Invalid request header name.",
            );
        }
        let value = value.trim();
        if !is_valid_header_value(value) {
            return bad_request(
                &mut stream,
                "400 Bad Request",
                "Invalid request header value.",
            );
        }
        if name.eq_ignore_ascii_case("content-length") {
            if content_length.is_some() {
                return bad_request(
                    &mut stream,
                    "400 Bad Request",
                    "Duplicate Content-Length headers are not accepted.",
                );
            }
            // Strict: digits only, no `+`, no whitespace, no overflow tricks.
            if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
                return bad_request(
                    &mut stream,
                    "400 Bad Request",
                    "Content-Length must be a non-negative integer.",
                );
            }
            let parsed = match value.parse::<usize>() {
                Ok(length) => length,
                Err(_) => {
                    return bad_request(
                        &mut stream,
                        "400 Bad Request",
                        "Content-Length must be a non-negative integer.",
                    )
                }
            };
            if parsed > MAX_BODY_BYTES {
                return bad_request(
                    &mut stream,
                    "413 Payload Too Large",
                    "Request body exceeds 1 MiB.",
                );
            }
            content_length = Some(parsed);
        } else if name.eq_ignore_ascii_case("transfer-encoding") && !value.is_empty() {
            transfer_encoding = true;
        } else if name.eq_ignore_ascii_case("content-type") {
            // Keep the first value only; duplicates are merged leniently.
            if content_type.is_none() {
                content_type = Some(value.to_owned());
            }
        }
    }

    if transfer_encoding && content_length.is_some() {
        return bad_request(
            &mut stream,
            "400 Bad Request",
            "Transfer-Encoding and Content-Length cannot be combined.",
        );
    }
    if transfer_encoding {
        return bad_request(
            &mut stream,
            "501 Not Implemented",
            "Transfer-Encoding is not supported; use Content-Length.",
        );
    }

    let content_length = content_length.unwrap_or(0);
    if content_length > 0 && !(method == "POST" && path == "/api/control") {
        return bad_request(
            &mut stream,
            "400 Bad Request",
            "This route does not accept a request body.",
        );
    }
    // Enforce a JSON content type on the control API when the client sends one.
    // Missing Content-Type is still accepted for `curl` convenience.
    if method == "POST" && path == "/api/control" && content_length > 0 {
        if let Some(ct) = content_type.as_deref() {
            if !ct.to_ascii_lowercase().contains("application/json") {
                let payload =
                    json!({"ok": false, "error": "Content-Type must be application/json."})
                        .to_string();
                return respond(
                    &mut stream,
                    "415 Unsupported Media Type",
                    "application/json; charset=utf-8",
                    payload.as_bytes(),
                );
            }
        }
    }
    let mut body = vec![0u8; content_length];
    if let Err(error) = read_http_body(&mut reader, &mut body, request_deadline) {
        if error.kind() == std::io::ErrorKind::UnexpectedEof {
            return bad_request(
                &mut stream,
                "400 Bad Request",
                "Request body ended before Content-Length bytes were received.",
            );
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
        ("GET", "/favicon.ico") => respond(
            &mut stream,
            "204 No Content",
            "text/plain; charset=utf-8",
            b"",
        ),
        ("GET", "/api/state") => {
            let state = match brain.read() {
                Ok(model) => model.snapshot(),
                Err(_) => {
                    return respond(
                        &mut stream,
                        "500 Internal Server Error",
                        "application/json; charset=utf-8",
                        br#"{"ok":false,"error":"Simulation state unavailable."}"#,
                    )
                }
            };
            match serde_json::to_vec(&state) {
                Ok(bytes) => respond(
                    &mut stream,
                    "200 OK",
                    "application/json; charset=utf-8",
                    &bytes,
                ),
                Err(error) => {
                    let payload =
                        json!({"ok":false,"error":format!("Serialization failed: {error}")})
                            .to_string();
                    respond(
                        &mut stream,
                        "500 Internal Server Error",
                        "application/json; charset=utf-8",
                        payload.as_bytes(),
                    )
                }
            }
        }
        ("POST", "/api/control") => control(&mut stream, &brain, &body),
        (_, "/" | "/healthz" | "/api/state" | "/api/control" | "/favicon.ico") => {
            method_not_allowed(&mut stream)
        }
        _ => respond(
            &mut stream,
            "404 Not Found",
            "application/json; charset=utf-8",
            br#"{"ok":false,"error":"Route not found."}"#,
        ),
    }
}
fn control(stream: &mut TcpStream, brain: &Arc<RwLock<Brain>>, body: &[u8]) -> std::io::Result<()> {
    let command: ControlCommand = match serde_json::from_slice(body) {
        Ok(command) => command,
        Err(error) => {
            let payload =
                json!({"ok":false,"error":format!("Invalid JSON command: {error}")}).to_string();
            return respond(
                stream,
                "400 Bad Request",
                "application/json; charset=utf-8",
                payload.as_bytes(),
            );
        }
    };
    if command.action.is_empty() || command.action.len() > 32 {
        let payload = json!({"ok": false, "error": "Unknown or missing action."}).to_string();
        return respond(
            stream,
            "400 Bad Request",
            "application/json; charset=utf-8",
            payload.as_bytes(),
        );
    }

    // Run the mutation under the write lock, then drop the guard before
    // doing socket I/O so a slow client cannot stall the simulation thread.
    let result: Result<String, String> = {
        let mut model = match brain.write() {
            Ok(model) => model,
            Err(_) => {
                return respond(
                    stream,
                    "500 Internal Server Error",
                    "application/json; charset=utf-8",
                    br#"{"ok":false,"error":"Simulation lock is poisoned."}"#,
                )
            }
        };

        match command.action.as_str() {
            "stimulate" => {
                let region = command.region.as_deref().unwrap_or("Visual cortex");
                let intensity = command.intensity.unwrap_or(18.0);
                model
                    .stimulate(region, intensity)
                    .map(|()| format!("Stimulus delivered to {region}."))
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
                let paused = command.paused.unwrap_or(true);
                model.set_paused(paused);
                Ok(if paused {
                    "Simulation paused.".to_owned()
                } else {
                    "Simulation resumed.".to_owned()
                })
            }
            "toggle_pause" => {
                model.toggle_pause();
                Ok("Simulation state toggled.".to_owned())
            }
            "speed" => {
                model.set_speed(command.value.unwrap_or(1.0));
                let actual = model.clock_speed();
                Ok(format!("Clock speed set to {actual:.2}×."))
            }
            "plasticity" => {
                if let Some(enabled) = command.enabled {
                    model.set_plasticity(enabled);
                    Ok(if enabled {
                        "Synaptic plasticity enabled.".to_owned()
                    } else {
                        "Synaptic plasticity frozen.".to_owned()
                    })
                } else if let Some(value) = command.value {
                    // Accept `{"action":"plasticity","value":1}` / `0` for simple clients.
                    let enabled = value != 0.0;
                    model.set_plasticity(enabled);
                    Ok(if enabled {
                        "Synaptic plasticity enabled.".to_owned()
                    } else {
                        "Synaptic plasticity frozen.".to_owned()
                    })
                } else {
                    model.toggle_plasticity();
                    Ok("Plasticity state toggled.".to_owned())
                }
            }
            "associate" => {
                let cue = command
                    .cue
                    .as_deref()
                    .or(command.query.as_deref())
                    .unwrap_or("");
                let target = command.target.as_deref().unwrap_or("");
                model
                    .associate(cue, target)
                    .map(|()| "Associative trace encoded.".to_owned())
            }
            "recall" => {
                let query = command
                    .query
                    .as_deref()
                    .or(command.cue.as_deref())
                    .unwrap_or("");
                Ok(model.recall(query))
            }
            "forget" => {
                let cue = command
                    .cue
                    .as_deref()
                    .or(command.query.as_deref())
                    .unwrap_or("");
                if cue.trim().is_empty() {
                    Err("A cue is required to forget a trace.".to_owned())
                } else if model.forget_memory(cue) {
                    Ok("Associative trace forgotten.".to_owned())
                } else {
                    Err("No stored trace matches that cue.".to_owned())
                }
            }
            "clear_memory" | "clear_memories" => {
                model.clear_memories();
                Ok("Associative memory cleared.".to_owned())
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
        }
    };

    let response = match result {
        Ok(message) => json!({"ok":true,"message":message}),
        Err(error) => json!({"ok":false,"error":error}),
    };
    let status = if response["ok"].as_bool() == Some(true) {
        "200 OK"
    } else {
        "400 Bad Request"
    };
    respond(
        stream,
        status,
        "application/json; charset=utf-8",
        response.to_string().as_bytes(),
    )
}

fn respond(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    body: &[u8],
) -> std::io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nX-Frame-Options: DENY\r\nReferrer-Policy: no-referrer\r\nPermissions-Policy: accelerometer=(), camera=(), microphone=(), geolocation=()\r\nCross-Origin-Opener-Policy: same-origin\r\nCross-Origin-Resource-Policy: same-origin\r\nContent-Security-Policy: default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'self'; frame-ancestors 'self'; form-action 'self'\r\n\r\n",
        body.len()
    )?;
    stream.write_all(body)?;
    stream.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_line_parser_accepts_only_origin_form_and_known_http_versions_are_checked_by_handler()
    {
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
        )
        .unwrap();
        assert_eq!(command.action, "stimulate");
        assert_eq!(command.region.as_deref(), Some("Hippocampus"));
        assert_eq!(command.intensity, Some(16.5));
    }
}

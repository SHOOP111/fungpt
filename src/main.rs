#![forbid(unsafe_code)]

mod brain;

use brain::Brain;
use serde::Deserialize;
use serde_json::json;
use std::env;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

const MAX_BODY_BYTES: usize = 1_048_576;

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
    let brain = Arc::new(Mutex::new(Brain::new()));

    {
        let simulation = Arc::clone(&brain);
        thread::Builder::new()
            .name("cortex-simulation".to_owned())
            .spawn(move || loop {
                let started = Instant::now();
                let mut speed = 1.0f32;
                if let Ok(mut model) = simulation.lock() {
                    speed = model.snapshot()["meta"]["speed"].as_f64().unwrap_or(1.0) as f32;
                    model.step();
                }
                let target = Duration::from_secs_f32((0.001 / speed.clamp(0.25, 8.0)).max(0.000125));
                if let Some(remaining) = target.checked_sub(started.elapsed()) {
                    thread::sleep(remaining);
                }
            })?;
    }

    println!("
╭────────────────────────────────────────────────────╮
│  CORTEX LAB · brain-inspired systems workbench     │
╰────────────────────────────────────────────────────╯
  Dashboard : http://{bind}
  Model     : 292 spiking units · sparse recurrent synapses
  Safety    : localhost-only by default; no telemetry
  Stop      : Ctrl+C
");
    for incoming in listener.incoming() {
        match incoming {
            Ok(stream) => {
                let shared = Arc::clone(&brain);
                thread::Builder::new()
                    .name("cortex-http".to_owned())
                    .spawn(move || {
                        if let Err(error) = handle_connection(stream, shared) {
                            eprintln!("HTTP request failed: {error}");
                        }
                    })?;
            }
            Err(error) => eprintln!("Connection accept failed: {error}"),
        }
    }
    Ok(())
}

fn handle_connection(mut stream: TcpStream, brain: Arc<Mutex<Brain>>) -> std::io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    stream.set_write_timeout(Some(Duration::from_secs(5)))?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request_line = String::new();
    if reader.read_line(&mut request_line)? == 0 {
        return Ok(());
    }
    let mut pieces = request_line.split_whitespace();
    let method = pieces.next().unwrap_or("");
    let raw_path = pieces.next().unwrap_or("/");
    let path = raw_path.split('?').next().unwrap_or("/");
    let mut content_length = 0usize;

    loop {
        let mut header = String::new();
        if reader.read_line(&mut header)? == 0 || header == "\r\n" || header == "\n" {
            break;
        }
        if let Some((key, val)) = header.split_once(':') {
            if key.eq_ignore_ascii_case("content-length") {
                content_length = val.trim().parse().unwrap_or(0);
            }
        }
    }

    if content_length > MAX_BODY_BYTES {
        return respond(&mut stream, "413 Payload Too Large", "application/json; charset=utf-8",
            br#"{"ok":false,"error":"Request body exceeds 1 MiB."}"#);
    }
    let mut body = vec![0u8; content_length];
    reader.read_exact(&mut body)?;

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
                Err(_) => return respond(&mut stream, "500 Internal Server Error", "application/json; charset=utf-8",
                    br#"{"ok":false,"error":"Simulation state unavailable."}"#),
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
        _ => respond(&mut stream, "404 Not Found", "application/json; charset=utf-8",
            br#"{"ok":false,"error":"Route not found."}"#),
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
            *model = Brain::new();
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

//! Integration tests that exercise `ZepMemoryClient` against a minimal
//! in-process HTTP mock. The mock speaks just enough HTTP/1.1 to satisfy
//! `reqwest::blocking` without requiring external crates.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{self, Sender};
use std::thread;

use memory_client::{MemoryClient, MemoryMessage, MemoryRole, ZepConfig, ZepMemoryClient};

struct MockResponse {
    status_line: &'static str,
    body: String,
}

struct CapturedRequest {
    method: String,
    path: String,
    body: String,
    headers: Vec<(String, String)>,
}

struct MockServer {
    port: u16,
    requests: std::sync::mpsc::Receiver<CapturedRequest>,
}

impl MockServer {
    fn spawn(responses: Vec<MockResponse>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("addr").port();
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || serve(&listener, responses, &tx));
        Self { port, requests: rx }
    }

    fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    fn next_request(&self) -> CapturedRequest {
        self.requests
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("request captured within timeout")
    }
}

fn serve(listener: &TcpListener, mut responses: Vec<MockResponse>, tx: &Sender<CapturedRequest>) {
    responses.reverse();
    for incoming in listener.incoming() {
        let Ok(stream) = incoming else { continue };
        let Some(response) = responses.pop() else {
            break;
        };
        if let Some(captured) = handle_connection(stream, &response) {
            let _ = tx.send(captured);
        }
        if responses.is_empty() {
            break;
        }
    }
}

fn handle_connection(mut stream: TcpStream, response: &MockResponse) -> Option<CapturedRequest> {
    let mut reader = BufReader::new(stream.try_clone().ok()?);
    let mut request_line = String::new();
    reader.read_line(&mut request_line).ok()?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next()?.to_string();
    let path = parts.next()?.to_string();

    let mut headers: Vec<(String, String)> = Vec::new();
    let mut content_length = 0usize;
    loop {
        let mut header_line = String::new();
        if reader.read_line(&mut header_line).ok()? == 0 {
            break;
        }
        if header_line == "\r\n" || header_line == "\n" {
            break;
        }
        let trimmed = header_line.trim_end_matches(&['\r', '\n'][..]);
        if let Some((name, value)) = trimmed.split_once(':') {
            let name = name.trim().to_string();
            let value = value.trim().to_string();
            if name.eq_ignore_ascii_case("content-length") {
                content_length = value.parse().unwrap_or(0);
            }
            headers.push((name, value));
        }
    }

    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body).ok()?;
    }

    let body = String::from_utf8_lossy(&body).into_owned();

    let payload = format!(
        "{status_line}\r\nContent-Type: application/json\r\nContent-Length: {len}\r\nConnection: close\r\n\r\n{body}",
        status_line = response.status_line,
        len = response.body.len(),
        body = response.body,
    );
    stream.write_all(payload.as_bytes()).ok()?;
    stream.flush().ok()?;

    Some(CapturedRequest {
        method,
        path,
        body,
        headers,
    })
}

#[test]
fn recall_returns_facts_from_zep_response() {
    let server = MockServer::spawn(vec![MockResponse {
        status_line: "HTTP/1.1 200 OK",
        body: r#"{"results":[{"fact":"User prefers Rust"},{"content":"Working on claw-code"}]}"#
            .to_string(),
    }]);

    let config = ZepConfig::new(server.base_url(), "alice").with_api_key("secret");
    let mut client = ZepMemoryClient::new(config).expect("client");

    let recalled = client
        .recall("session-42", "what is the user working on?", 3)
        .expect("recall");

    assert_eq!(
        recalled,
        vec![
            "User prefers Rust".to_string(),
            "Working on claw-code".to_string(),
        ]
    );

    let captured = server.next_request();
    assert_eq!(captured.method, "POST");
    assert_eq!(captured.path, "/api/v2/sessions/session-42/search");
    assert!(captured
        .body
        .contains("\"text\":\"what is the user working on?\""));
    assert!(captured.body.contains("\"limit\":3"));
    assert!(captured.body.contains("\"user_id\":\"alice\""));
    let auth = captured
        .headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("authorization"))
        .expect("authorization header sent");
    assert_eq!(auth.1, "Api-Key secret");
}

#[test]
fn recall_surfaces_http_errors_as_protocol_errors() {
    let server = MockServer::spawn(vec![MockResponse {
        status_line: "HTTP/1.1 503 Service Unavailable",
        body: r#"{"error":"down"}"#.to_string(),
    }]);

    let config = ZepConfig::new(server.base_url(), "alice");
    let mut client = ZepMemoryClient::new(config).expect("client");

    let error = client
        .recall("session-42", "anything", 1)
        .expect_err("server 5xx must surface");
    assert!(format!("{error}").contains("503"));
}

#[test]
fn ingest_posts_messages_to_expected_endpoint() {
    let server = MockServer::spawn(vec![MockResponse {
        status_line: "HTTP/1.1 200 OK",
        body: "{}".to_string(),
    }]);

    let config = ZepConfig::new(server.base_url(), "alice");
    let mut client = ZepMemoryClient::new(config).expect("client");

    let messages = vec![
        MemoryMessage::new(MemoryRole::User, "remember I like Rust"),
        MemoryMessage::new(MemoryRole::Assistant, "noted"),
    ];
    client.ingest("session-42", &messages).expect("ingest");

    let captured = server.next_request();
    assert_eq!(captured.method, "POST");
    assert_eq!(captured.path, "/api/v2/sessions/session-42/messages");
    assert!(captured.body.contains("\"role\":\"user\""));
    assert!(captured.body.contains("\"role\":\"assistant\""));
    assert!(captured.body.contains("remember I like Rust"));
}

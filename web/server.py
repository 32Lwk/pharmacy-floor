"""Local-only demo UI: chat on the left, decision panel on the right. Binds 127.0.0.1."""
import gc, json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from floor.floor import PharmacyFloor

FLOOR = PharmacyFloor()
HTML = (Path(__file__).parent / "index.html").read_bytes()


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        self.send_response(code); self.send_header("Content-Type", ctype); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            return self._send(200, HTML, "text/html; charset=utf-8")
        if self.path == "/api/info":
            return self._send(200, json.dumps({"llm": FLOOR.llm_label()}).encode(), "application/json")
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/chat":
            return self._send(404, b"not found", "text/plain")
        n = int(self.headers.get("Content-Length", 0))
        msg = json.loads(self.rfile.read(n) or b"{}").get("message", "")[:1000]
        try:
            body = json.dumps(FLOOR.handle(msg).to_dict()).encode()
        except Exception as e:  # keep the server up; the UI shows the error
            body = json.dumps({"final_agent": "error", "reply": f"Error: {type(e).__name__}", "entries": [], "total_ms": 0}).encode()
        finally:
            gc.collect()
        self._send(200, body, "application/json")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"Pharmacy Floor on http://127.0.0.1:{port} (LLM: {FLOOR.llm_label()})", flush=True)
    HTTPServer(("127.0.0.1", port), H).serve_forever()

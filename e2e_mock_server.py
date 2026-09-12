"""E2E 验证用 mock：确定性发出 step/sources/delta 事件，不依赖外部网络。"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

class Handler(BaseHTTPRequestHandler):
    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send_json({})

    def do_GET(self):
        if self.path.startswith("/health"):
            self._send_json({"status": "ok", "components": {}})
        elif self.path.startswith("/v1/modules"):
            self._send_json({"modules": [{"name": "chat", "description": "mock"}]})
        else:
            self._send_json({})

    def do_DELETE(self):
        self._send_json({"deleted": True})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        def ev(name, data):
            self.wfile.write(f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode())
            self.wfile.flush()
        ev("ping", {"type": "ping"})
        ev("step", {"type": "step", "name": "agent", "status": "running"})
        # 两次并行同名步骤 + sources，验证 FIFO 配对与来源卡片
        ev("step", {"type": "step", "name": "web_search", "status": "running", "detail": "搜索：mock A"})
        ev("step", {"type": "step", "name": "web_search", "status": "running", "detail": "搜索：mock B"})
        ev("step", {"type": "step", "name": "web_search", "status": "completed", "detail": "找到 2 条结果"})
        ev("sources", {"type": "sources", "sources": [
            {"title": "Mock 来源一：Rust 官方博客", "url": "https://blog.rust-lang.org/mock-a"},
            {"title": "Mock 来源二：Go 发布公告", "url": "https://go.dev/doc/mock-b"},
            {"title": "无链接来源", "url": None},
        ]})
        ev("step", {"type": "step", "name": "web_search", "status": "completed", "detail": "找到 3 条结果"})
        ev("delta", {"type": "delta", "content": "这是 mock 回复的正文。"})
        ev("step", {"type": "step", "name": "agent", "status": "completed"})
        ev("done", {"type": "done", "thread_id": "mock-thread-1"})
        self.wfile.write(b"event: done\ndata: {\"type\":\"done\"}\n\n")

HTTPServer(("127.0.0.1", 8765), Handler).serve_forever()

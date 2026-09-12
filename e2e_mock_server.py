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
        # 记忆端点（M6）：更具体的前缀先匹配，避免 /v1/memory 吞掉子路径。
        if self.path.startswith("/v1/memory/blocks"):
            self._send_json({"blocks": [
                {"agent_id": "*", "label": "persona",
                 "content": "mock 人设：简洁、务实的工程助手。", "char_limit": 2000,
                 "version": 3, "updated_at": 1757600000.0},
            ]})
        elif self.path.startswith("/v1/memory/profile"):
            self._send_json({"profile": {
                "基本信息": ["用户在开发 agent-base 基座"],
                "偏好": ["简洁的中文回答"],
            }})
        elif self.path.startswith("/v1/memory/audit"):
            self._send_json({"ops": [
                {"op_id": "mockop1", "op": "extract", "user_id": "default",
                 "agent_id": "chat", "thread_id": "chat:mock-1", "detail": {"candidates": 2},
                 "status": "ok", "error_text": None, "duration_ms": 812,
                 "created_at": 1757600000.0},
            ]})
        elif self.path.startswith("/v1/memory"):
            self._send_json({"memories": [
                {"memory_id": "mockmem1", "user_id": "default", "agent_id": "*",
                 "kind": "semantic", "content": "用户偏好深色主题（mock）", "tags": ["UI"],
                 "salience": 0.8, "status": "active", "source_thread_id": "chat:mock-1",
                 "has_embedding": True, "access_count": 2, "created_at": 1757600000.0,
                 "updated_at": 1757600000.0},
            ]})
        elif self.path.startswith("/health"):
            self._send_json({"status": "ok", "components": {"memory": "ok"}})
        elif self.path.startswith("/v1/modules"):
            self._send_json({"modules": [{"name": "chat", "description": "mock"}]})
        else:
            self._send_json({})

    def do_DELETE(self):
        self._send_json({"deleted": True})

    def do_PUT(self):
        self._read_body()
        self._send_json({"agent_id": "*", "label": "persona",
                         "content": "mock 人设（已写入）。", "char_limit": 2000,
                         "version": 4, "updated_at": 1757600001.0})

    def do_PATCH(self):
        self._read_body()
        self._send_json({"memory_id": "mockmem1", "content": "mock 记忆（已更新）。",
                         "status": "active"})

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)

    def do_POST(self):
        self._read_body()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        def ev(name, data):
            frame = f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
            self.wfile.write(frame.encode())
            self.wfile.flush()
        ev("ping", {"type": "ping"})
        ev("step", {"type": "step", "name": "agent", "status": "running"})
        # 两次并行同名工具调用（M5）：tool_call start/end 带参数与结果
        ev("tool_call", {"type": "tool_call", "call_id": "mockc1", "name": "web_search",
                         "phase": "start", "args": {"query": "Rust 1.75", "max_results": 5}})
        ev("tool_call", {"type": "tool_call", "call_id": "mockc2", "name": "web_search",
                         "phase": "start", "args": {"query": "Go 1.22", "max_results": 8}})
        ev("step", {"type": "step", "name": "web_search", "status": "running",
                    "detail": "搜索：mock A"})
        ev("step", {"type": "step", "name": "web_search", "status": "completed",
                    "detail": "找到 2 条结果"})
        ev("sources", {"type": "sources", "sources": [
            {"title": "Mock 来源一：Rust 官方博客", "url": "https://blog.rust-lang.org/mock-a"},
            {"title": "Mock 来源二：Go 发布公告", "url": "https://go.dev/doc/mock-b"},
            {"title": "无链接来源", "url": None},
        ]})
        ev("tool_call", {"type": "tool_call", "call_id": "mockc1", "name": "web_search",
                         "phase": "end", "status": "ok", "duration_ms": 1234,
                         "result": "Rust 1.75 稳定了 async fn in trait。"})
        ev("tool_call", {"type": "tool_call", "call_id": "mockc2", "name": "web_search",
                         "phase": "end", "status": "error", "duration_ms": 567,
                         "error": "tool execution failed: mock 网络超时"})
        ev("delta", {"type": "delta", "content": "这是 mock 回复的正文。"})
        ev("step", {"type": "step", "name": "agent", "status": "completed"})
        ev("done", {"type": "done", "thread_id": "mock-thread-1"})
        self.wfile.write(b"event: done\ndata: {\"type\":\"done\"}\n\n")

HTTPServer(("127.0.0.1", 8765), Handler).serve_forever()

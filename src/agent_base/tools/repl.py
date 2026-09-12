"""python_repl 工具（M5）：子进程隔离执行模型生成的 Python 代码。

隔离手段（诚实边界）：``-I`` 隔离模式（不加载用户 site-packages 与
环境变量）、独立子进程（超时强制 kill——不像线程那样无法杀死）、
POSIX 上的 CPU/内存 rlimit、一次性临时工作目录。**它不是安全边界**：
子进程仍可访问文件系统与网络，因此工具的安全级别标记为 ``exec``，
默认不在 ``TOOLKIT_ENABLED`` 中——只在部署方能接受该风险时显式开启。

输出封顶：stdout/stderr 合计超过上限即截断并标注，防止失控循环把
上下文灌爆。执行超时返回可读的错误说明（模型据此改短代码重试），
而不是让池的兜底超时把它当成失败。
"""

from __future__ import annotations

import subprocess
import sys
import tempfile

from langchain_core.tools import tool

from agent_base.tools.spec import ToolSpec

# 子进程自身的硬超时；注册表声明的池超时（60s）比它更大，只做兜底。
REPL_SUBPROCESS_TIMEOUT_SECONDS = 30.0
# stdout + stderr 合计的输出封顶（字符）。
_OUTPUT_MAX_CHARS = 10_000
# POSIX rlimit：CPU 秒与地址空间（字节）。
_POSIX_CPU_SECONDS = 10
_POSIX_AS_BYTES = 512 * 1024 * 1024


def _posix_limits() -> None:
    """子进程资源限额；仅 POSIX 平台生效（CI 的 Linux job 覆盖本函数）。"""
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (_POSIX_CPU_SECONDS, _POSIX_CPU_SECONDS))
    # RLIMIT_AS 在 macOS 上会让解释器启动失败（地址空间语义不同），只对 Linux 施加。
    if sys.platform == "linux":
        resource.setrlimit(resource.RLIMIT_AS, (_POSIX_AS_BYTES, _POSIX_AS_BYTES))


def _truncate(text: str) -> tuple[str, bool]:
    if len(text) <= _OUTPUT_MAX_CHARS:
        return text, False
    return text[:_OUTPUT_MAX_CHARS] + f"\n…（输出超过 {_OUTPUT_MAX_CHARS} 字符，已截断）", True


@tool
def python_repl(code: str) -> str:
    """在隔离子进程中执行一段 Python 代码并返回 stdout/stderr 与退出码。

    适用场景：数据变换、格式转换、文本计算等需要真正"运行"的任务。
    代码在独立子进程中运行，超时（30 秒）会被强制终止；它不是安全
    边界，不要用它执行会影响外部系统的操作。print() 的内容会出现在
    stdout 中——想看到中间结果必须显式 print。

    Args:
        code: 要执行的 Python 代码（一次一段完整脚本）。
    """
    if not code.strip():
        raise ValueError("code 不能为空：请提供要执行的 Python 代码")

    preexec = _posix_limits if sys.platform != "win32" else None
    with tempfile.TemporaryDirectory(prefix="agent-base-repl-") as workdir:
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", code],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=REPL_SUBPROCESS_TIMEOUT_SECONDS,
                cwd=workdir,
                preexec_fn=preexec,
            )
        except subprocess.TimeoutExpired:
            return (
                f"执行超时：代码在 {REPL_SUBPROCESS_TIMEOUT_SECONDS:.0f} 秒内没有跑完，"
                "已被强制终止。请把代码改短或降低计算规模后重试。"
            )

    stdout, truncated_out = _truncate(completed.stdout)
    stderr, truncated_err = _truncate(completed.stderr)
    sections = [f"exit code: {completed.returncode}"]
    if stdout:
        sections.append(f"--- stdout ---\n{stdout}")
    if stderr:
        sections.append(f"--- stderr ---\n{stderr}")
    if truncated_out or truncated_err:
        sections.append("(输出已截断)")
    return "\n".join(sections)


PYTHON_REPL_SPEC = ToolSpec(
    name="python_repl",
    category="code",
    safety="exec",
    factory=lambda _settings: [python_repl],
    available=lambda _settings: True,
    unavailable_reason="",
    # 子进程硬超时 30s；池超时给足余量（进程启动/产物回收的开销）。
    timeout=60.0,
)

__all__ = ["PYTHON_REPL_SPEC", "python_repl"]

The Art of Loop Engineering（循环工程的艺术）—— 全面解读与完整案例
原文：https://www.langchain.com/blog/the-art-of-loop-engineering 
作者：Sydney Runkle · 发布日期：2026年6月16日 · 阅读时长：约7分钟
灵感来源：Swyx 的 “loopcraft: the art of stacking loops”

目录
●核心论点
●Loop 1：Agent 循环（The Agent Loop）
●Loop 2：验证循环（Verification Loop）
●Loop 3：事件驱动循环（Event-Driven Loop）
●Loop 4：爬山循环（Hill Climbing Loop）
●人类监督与专家介入
●四层循环总览表
●核心结论

核心论点
构建可靠的 Agent 不仅仅需要一个好模型，更需要一个精心设计的、适配特定任务的"循环骨架"（harness）。Agent 的核心算法很简单 —— 让 LLM 在循环中调用工具直到完成。但这不是唯一的循环。真正的威力在于堆叠和扩展多层循环。

Loop 1：Agent 循环（The Agent Loop）
架构

模型 → 调用工具 → 获取结果 → 模型 → 调用工具 → ... → 完成

核心要点
●本质：最基本的循环 —— 模型在循环中调用工具，直到任务完成。
●LangChain 实现：create_agent —— 选择任意模型，接入工具，即可获得一个可工作的 Agent。
●工具（Tools） 是 Agent 在真实世界中采取行动的能力来源。
●实例（LangChain 内部 Docs Agent）：接收文档改进请求 → 模型规划并起草修改 → 使用工具克隆仓库、读取文件、撰写文档、发起 Pull Request。
完整案例：Docs Agent

"""
Loop 1: The Agent Loop — 完整的 Docs Agent 实现
使用 LangChain 的 create_agent 构建一个自动化文档编写 Agent
"""

import os
from typing import Annotated, List
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
import subprocess
import tempfile


# ============================================================
# 1. 定义工具（Tools）
# ============================================================

class CloneRepoInput(BaseModel):
    """克隆仓库的输入参数"""
    repo_url: str = Field(description="GitHub 仓库 URL，如 https://github.com/user/repo")
    branch: str = Field(default="main", description="要克隆的分支名")


class ReadFileInput(BaseModel):
    """读取文件的输入参数"""
    file_path: str = Field(description="相对于仓库根目录的文件路径")
    start_line: int = Field(default=1, description="起始行号")
    end_line: int = Field(default=-1, description="结束行号，-1 表示读完")


class WriteFileInput(BaseModel):
    """写入文件的输入参数"""
    file_path: str = Field(description="要写入的文件路径")
    content: str = Field(description="要写入的完整文件内容")


class CreatePRIput(BaseModel):
    """创建 PR 的输入参数"""
    title: str = Field(description="PR 标题")
    body: str = Field(description="PR 描述")
    branch_name: str = Field(description="新分支名")


@tool(args_schema=CloneRepoInput)
def clone_repo(repo_url: str, branch: str = "main") -> str:
    """克隆一个 GitHub 仓库到本地临时目录。"""
    work_dir = tempfile.mkdtemp(prefix="docs_agent_")
    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    target_dir = os.path.join(work_dir, repo_name)

    result = subprocess.run(
        ["git", "clone", "-b", branch, repo_url, target_dir],
        capture_output=True, text=True, timeout=120
    )

    if result.returncode != 0:
        return f"克隆失败: {result.stderr}"

    return f"已克隆到: {target_dir}\n当前目录: {target_dir}"


@tool(args_schema=ReadFileInput)
def read_file(file_path: str, start_line: int = 1, end_line: int = -1) -> str:
    """读取仓库中的文件内容。"""
    if not os.path.exists(file_path):
        return f"文件不存在: {file_path}"

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if end_line == -1:
        end_line = len(lines)

    selected = lines[start_line - 1:end_line]
    return "".join(selected)


@tool(args_schema=WriteFileInput)
def write_file(file_path: str, content: str) -> str:
    """将内容写入文件（覆盖模式）。"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"已写入 {len(content)} 字符到 {file_path}"


@tool(args_schema=CreatePRIput)
def create_pr(title: str, body: str, branch_name: str) -> str:
    """创建分支、提交修改并创建 Pull Request。"""
    # 创建新分支
    subprocess.run(["git", "checkout", "-b", branch_name], capture_output=True, text=True)
    # 添加并提交
    subprocess.run(["git", "add", "-A"], capture_output=True, text=True)
    commit_result = subprocess.run(
        ["git", "commit", "-m", title],
        capture_output=True, text=True
    )
    # 推送
    push_result = subprocess.run(
        ["git", "push", "origin", branch_name],
        capture_output=True, text=True
    )
    # 使用 gh CLI 创建 PR
    pr_result = subprocess.run(
        ["gh", "pr", "create", "--title", title, "--body", body, "--base", "main"],
        capture_output=True, text=True
    )

    if pr_result.returncode == 0:
        return f"PR 创建成功!\n{pr_result.stdout}"
    return f"PR 创建失败: {pr_result.stderr}"


# ============================================================
# 2. 组装 Agent
# ============================================================

# 系统提示词
SYSTEM_PROMPT = """你是一个专业的文档编写 Agent。你的职责是：

1. 接收文档改进请求
2. 分析现有文档结构
3. 规划修改方案
4. 执行修改（克隆仓库、读取文件、编写文档）
5. 创建 Pull Request

工作流程：
- 先用 clone_repo 克隆目标仓库
- 用 read_file 查看需要修改的文件
- 用 write_file 写入修改后的文档
- 用 create_pr 提交 PR

请确保文档质量：
- 所有链接必须可解析
- 代码示例必须可运行
- 遵循项目的文档风格约定
"""

# 创建 LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 工具列表
tools = [clone_repo, read_file, write_file, create_pr]

# 创建 Agent（带记忆）
memory = MemorySaver()
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=memory,
)

# ============================================================
# 3. 运行 Agent
# ============================================================

def run_docs_agent(request: str, thread_id: str = "default"):
    """运行文档 Agent，处理一个文档改进请求。"""
    config = {"configurable": {"thread_id": thread_id}}

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=request),
    ]

    result = agent.invoke({"messages": messages}, config=config)

    # 收集 Agent 的最终响应
    for msg in reversed(result["messages"]):
        if hasattr(msg, "content") and msg.content and not hasattr(msg, "tool_calls"):
            return msg.content

    return "Agent 未产生最终响应"


# ============================================================
# 4. 使用示例
# ============================================================

if __name__ == "__main__":
    request = """
    请改进 langchain 项目中 /docs/getting_started.md 的文档。
    需要添加一个关于如何使用 create_agent 的快速入门章节，
    包含完整的代码示例和解释。
    """
    response = run_docs_agent(request)
    print(response)


Loop 2：验证循环（Verification Loop）
架构

Agent 输出 → 评分器（Grader）检查 → 不合格？→ 反馈给模型 → 重新生成 → 再次检查 ✅

核心要点
●动机：Agent 循环能完成工作，但不一定一次就正确或一致。当一致性很重要时，需要验证循环包裹它。
●核心机制：添加一个评分器（Grader），按标准（rubric）检查 Agent 的输出，不合格则带反馈信息返回模型重试。
●评分器类型： 确定性评分器（Deterministic）—— 如链接检查、CI 检查
○代理式评分器（Agentic）—— 如 “LLM as a judge”
●LangChain 实现：RubricMiddleware 或 create_agent 的 after_agent 钩子
●Trade-off：增加验证会增加延迟和成本，但当质量优先于速度时（大多数生产场景），这是值得的。
完整案例：带验证循环的 Docs Agent

"""
Loop 2: Verification Loop — 完整的验证循环实现
在 Agent 循环之上添加评分器，确保输出质量
"""

import re
import asyncio
from typing import Literal, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


# ============================================================
# 1. 定义评分标准（Rubric）
# ============================================================

class Grade(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NEEDS_REVISION = "NEEDS_REVISION"


@dataclass
class RubricItem:
    """单条评分标准"""
    name: str
    description: str
    weight: float = 1.0  # 权重


@dataclass
class GradingResult:
    """评分结果"""
    grade: Grade
    score: float  # 0.0 - 1.0
    feedback: str
    details: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# 2. 确定性评分器（Deterministic Graders）
# ============================================================

class LinkChecker:
    """检查文档中所有链接是否可解析"""

    def __init__(self):
        import requests
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "DocsAgent-LinkChecker/1.0"
        })

    def check(self, content: str) -> GradingResult:
        """检查文档中的所有链接"""
        # 提取所有 markdown 链接: [text](url) 和裸 URL
        md_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
        urls = re.findall(r'(?<!\()https?://[^\s<>"\)]+', content)

        all_links = [url for _, url in md_links] + urls
        broken_links = []

        for url in all_links[:20]:  # 限制检查数量
            try:
                resp = self.session.head(url, timeout=5, allow_redirects=True)
                if resp.status_code >= 400:
                    broken_links.append((url, resp.status_code))
            except Exception as e:
                broken_links.append((url, str(e)))

        if not broken_links:
            return GradingResult(
                grade=Grade.PASS,
                score=1.0,
                feedback=f"所有 {len(all_links)} 个链接均可解析",
                details={"total_links": len(all_links), "broken": []}
            )
        else:
            return GradingResult(
                grade=Grade.FAIL,
                score=1.0 - (len(broken_links) / len(all_links)) if all_links else 0,
                feedback=f"发现 {len(broken_links)} 个失效链接: {broken_links}",
                details={"total_links": len(all_links), "broken": broken_links}
            )


class CodeBlockValidator:
    """检查代码块是否语法正确"""

    def check(self, content: str) -> GradingResult:
        """检查文档中的代码块"""
        code_blocks = re.findall(r'```(\w+)?\n(.*?)```', content, re.DOTALL)
        errors = []

        for lang, code in code_blocks:
            if lang and lang.lower() == "python":
                try:
                    compile(code.strip(), "<docs>", "exec")
                except SyntaxError as e:
                    errors.append({
                        "language": lang,
                        "error": str(e),
                        "snippet": code[:100]
                    })

        if not errors:
            return GradingResult(
                grade=Grade.PASS,
                score=1.0,
                feedback=f"所有 {len(code_blocks)} 个代码块语法正确",
                details={"total_blocks": len(code_blocks), "errors": []}
            )
        else:
            return GradingResult(
                grade=Grade.FAIL,
                score=1.0 - (len(errors) / len(code_blocks)) if code_blocks else 0,
                feedback=f"发现 {len(errors)} 个语法错误",
                details={"total_blocks": len(code_blocks), "errors": errors}
            )


class DiffScopeChecker:
    """检查 diff 是否在请求范围内"""

    def __init__(self, expected_files: list[str]):
        self.expected_files = expected_files

    def check(self, diff_content: str) -> GradingResult:
        """检查 diff 中的文件变更是否在预期范围内"""
        changed_files = re.findall(r'^\+\+\+ b/(.+)$', diff_content, re.MULTILINE)
        unexpected = [f for f in changed_files if f not in self.expected_files]

        if not unexpected:
            return GradingResult(
                grade=Grade.PASS,
                score=1.0,
                feedback=f"所有变更文件在预期范围内: {changed_files}",
                details={"changed_files": changed_files, "unexpected": []}
            )
        else:
            return GradingResult(
                grade=Grade.FAIL,
                score=0.0,
                feedback=f"发现意外变更文件: {unexpected}",
                details={"changed_files": changed_files, "unexpected": unexpected}
            )


# ============================================================
# 3. 代理式评分器（LLM-as-Judge）
# ============================================================

class LLMJudge:
    """使用 LLM 作为裁判，评估文档质量"""

    JUDGE_PROMPT = """你是一个文档质量评审专家。请根据以下标准评估文档质量：

评分标准（每项 0-10 分）：
1. **清晰度**：内容是否清晰易懂，结构是否合理
2. **完整性**：是否覆盖了所有必要的主题和细节
3. **准确性**：技术内容是否准确无误
4. **代码质量**：代码示例是否可运行、格式是否正确
5. **风格一致性**：是否符合项目的文档风格约定

请给出每项评分和总体评价。如果发现问题，请明确指出需要修改的地方。

响应格式（JSON）：
{
    "clarity": {"score": 8, "comment": "..."},
    "completeness": {"score": 7, "comment": "..."},
    "accuracy": {"score": 9, "comment": "..."},
    "code_quality": {"score": 8, "comment": "..."},
    "style_consistency": {"score": 7, "comment": "..."},
    "overall_score": 7.8,
    "verdict": "PASS|FAIL|NEEDS_REVISION",
    "feedback": "总体评价和改进建议..."
}
"""

    def __init__(self, model: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=model, temperature=0)

    def evaluate(self, content: str, original_request: str) -> GradingResult:
        """使用 LLM 评估文档质量"""
        messages = [
            SystemMessage(content=self.JUDGE_PROMPT),
            HumanMessage(content=f"""
原始请求：
{original_request}

文档内容：
{content}

请评估以上文档质量。
""")
        ]

        response = self.llm.invoke(messages)

        # 解析 LLM 的 JSON 响应
        import json
        try:
            # 提取 JSON 部分
            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                result = json.loads(json_match.group())
                verdict = Grade(result.get("verdict", "FAIL"))
                return GradingResult(
                    grade=verdict,
                    score=result.get("overall_score", 0) / 10.0,
                    feedback=result.get("feedback", ""),
                    details=result
                )
        except (json.JSONDecodeError, KeyError) as e:
            pass

        return GradingResult(
            grade=Grade.FAIL,
            score=0,
            feedback=f"评分解析失败: {response.content[:500]}"
        )


# ============================================================
# 4. 验证循环（Verification Loop）
# ============================================================

@dataclass
class VerificationConfig:
    """验证循环配置"""
    max_retries: int = 3
    graders: list = field(default_factory=list)
    pass_threshold: float = 0.8  # 总体通过阈值
    require_all_pass: bool = False  # 是否要求所有评分器都通过


class VerificationLoop:
    """
    验证循环：包裹 Agent，对输出进行评分，不合格则重试

    工作流程：
    1. Agent 执行任务
    2. 所有评分器对输出打分
    3. 如果通过 → 返回结果
    4. 如果失败 → 构建反馈消息，让 Agent 重新执行
    5. 超过最大重试次数 → 返回最佳结果
    """

    def __init__(self, config: VerificationConfig):
        self.config = config
        self.run_history: list[dict] = []

    def _aggregate_results(self, results: list[GradingResult]) -> GradingResult:
        """聚合多个评分器的结果"""
        if not results:
            return GradingResult(grade=Grade.PASS, score=1.0, feedback="无评分器")

        total_weight = len(results)
        weighted_score = sum(r.score for r in results) / total_weight

        all_feedback = []
        for r in results:
            all_feedback.append(f"[{r.grade.value}] {r.feedback}")

        combined_feedback = "\n".join(all_feedback)

        if self.config.require_all_pass:
            all_pass = all(r.grade == Grade.PASS for r in results)
            grade = Grade.PASS if all_pass else Grade.FAIL
        else:
            if weighted_score >= self.config.pass_threshold:
                grade = Grade.PASS
            elif weighted_score >= 0.5:
                grade = Grade.NEEDS_REVISION
            else:
                grade = Grade.FAIL

        return GradingResult(
            grade=grade,
            score=weighted_score,
            feedback=combined_feedback,
            details={"individual_results": [
                {"grade": r.grade.value, "score": r.score, "feedback": r.feedback}
                for r in results
            ]}
        )

    def run(
        self,
        agent_fn,
        task_input: str,
        thread_id: str = "default"
    ) -> dict:
        """
        运行验证循环

        Args:
            agent_fn: Agent 执行函数，签名为 fn(task_input, feedback=None) -> str
            task_input: 任务描述
            thread_id: 线程 ID

        Returns:
            {
                "output": "最终输出",
                "grade": Grade,
                "score": float,
                "attempts": int,
                "history": [...]
            }
        """
        feedback = None
        best_result = None
        best_score = -1

        for attempt in range(1, self.config.max_retries + 1):
            print(f"\n{'='*60}")
            print(f" 验证循环 — 第 {attempt}/{self.config.max_retries} 次尝试")
            print(f"{'='*60}")

            # 1. Agent 执行
            if feedback:
                enhanced_input = f"{task_input}\n\n[上一轮反馈]\n{feedback}\n\n请根据反馈修改你的输出。"
            else:
                enhanced_input = task_input

            output = agent_fn(enhanced_input, thread_id=f"{thread_id}_attempt_{attempt}")

            # 2. 运行所有评分器
            all_results = []
            for grader in self.config.graders:
                try:
                    result = grader.check(output)
                    all_results.append(result)
                    print(f"  ✓ {grader.__class__.__name__}: {result.grade.value} (score={result.score:.2f})")
                except Exception as e:
                    print(f"  ✗ {grader.__class__.__name__}: 错误 - {e}")
                    all_results.append(GradingResult(
                        grade=Grade.FAIL, score=0, feedback=str(e)
                    ))

            # 3. 聚合结果
            aggregated = self._aggregate_results(all_results)
            print(f"  综合评分: {aggregated.grade.value} (score={aggregated.score:.2f})")

            # 记录历史
            self.run_history.append({
                "attempt": attempt,
                "output": output,
                "grade": aggregated.grade.value,
                "score": aggregated.score,
                "feedback": aggregated.feedback
            })

            # 跟踪最佳结果
            if aggregated.score > best_score:
                best_score = aggregated.score
                best_result = output

            # 4. 判断是否通过
            if aggregated.grade == Grade.PASS:
                return {
                    "output": output,
                    "grade": aggregated.grade,
                    "score": aggregated.score,
                    "attempts": attempt,
                    "history": self.run_history
                }

            # 5. 构建反馈
            feedback = f"输出未通过质量检查。\n{aggregated.feedback}\n请改进后重新提交。"

        # 超过最大重试次数，返回最佳结果
        return {
            "output": best_result,
            "grade": Grade.FAIL,
            "score": best_score,
            "attempts": self.config.max_retries,
            "history": self.run_history
        }


# ============================================================
# 5. 使用 LangChain 的 RubricMiddleware
# ============================================================

from langchain.agents.middleware import RubricMiddleware


def create_agent_with_rubric():
    """使用 LangChain 内置的 RubricMiddleware 创建带验证的 Agent"""
    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # 定义评分标准
    rubric = RubricMiddleware(
        rubric="""
评估标准（1-5 分）：
1. 文档包含清晰的标题和章节结构
2. 所有代码示例语法正确且可运行
3. 所有链接格式正确
4. 内容覆盖了原始请求中的所有要点
5. 语言风格符合项目文档规范

如果总分低于 4 分，请重试。
""",
        model=llm,
        max_retries=3,
    )

    agent = create_agent(
        model=llm,
        tools=[clone_repo, read_file, write_file, create_pr],
        middleware=[rubric],
    )

    return agent


# ============================================================
# 6. 使用 after_agent hook 手动实现
# ============================================================

def create_agent_with_after_hook():
    """使用 after_agent hook 实现验证循环"""
    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    def after_agent_hook(state: dict) -> dict:
        """Agent 执行后的验证钩子"""
        messages = state.get("messages", [])
        if not messages:
            return state

        # 获取最后一条 AI 消息
        last_msg = messages[-1]
        if not hasattr(last_msg, "content") or not last_msg.content:
            return state

        output = last_msg.content

        # 运行评分器
        link_checker = LinkChecker()
        code_validator = CodeBlockValidator()

        link_result = link_checker.check(output)
        code_result = code_validator.check(output)

        # 检查是否通过
        if link_result.grade != Grade.PASS or code_result.grade != Grade.FAIL:
            # 在 state 中添加反馈
            feedback = []
            if link_result.grade != Grade.PASS:
                feedback.append(link_result.feedback)
            if code_result.grade != Grade.PASS:
                feedback.append(code_result.feedback)

            state["verification_feedback"] = "\n".join(feedback)
            state["needs_retry"] = True

        return state

    agent = create_agent(
        model=llm,
        tools=[clone_repo, read_file, write_file, create_pr],
        after_agent=after_agent_hook,
    )

    return agent


# ============================================================
# 7. 完整使用示例
# ============================================================

if __name__ == "__main__":
    # 模拟 Agent 执行函数
    def mock_agent_fn(task_input: str, thread_id: str = "default") -> str:
        """模拟 Agent 输出（实际应调用真实 Agent）"""
        return f"""
# Getting Started with create_agent

This guide shows how to use `create_agent` to build your first agent.

## Installation

```python
pip install langchain langchain-openai

Basic Usage

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")
agent = create_agent(model=llm, tools=[...])

Link Check
For more info, see https://python.langchain.com/docs  and https://api.langchain.com .


    # 配置验证循环
    config = VerificationConfig(
        max_retries=3,
        graders=[
            LinkChecker(),
            CodeBlockValidator(),
            LLMJudge(model="gpt-4o"),
        ],
        pass_threshold=0.8,
        require_all_pass=False,
    )

    loop = VerificationLoop(config)

    result = loop.run(
        agent_fn=mock_agent_fn,
        task_input="请为 create_agent 编写一个快速入门指南，包含安装和基本用法示例。",
    )

    print(f"\n{'='*60}")
    print(f" 最终结果")
    print(f"{'='*60}")
    print(f"状态: {result['grade'].value}")
    print(f"评分: {result['score']:.2f}")
    print(f"尝试次数: {result['attempts']}")
    print(f"\n输出:\n{result['output'][:500]}")


Loop 3：事件驱动循环（Event-Driven Loop）
架构

事件触发（新文档/Slack消息/定时任务/Webhook）→ Agent 自动运行 → 更新系统

核心要点
●动机：Agent 开发中最关键的部分之一是集成层 —— 将 Agent 连接到生态系统中，使其能在后台持续运行。
●本质：Agent 不再是手动调用的东西，而是持续运行在更大系统中的组件。
●触发方式：新文档到达、定时任务（Cron）、Webhook、Slack 消息（如 #docs-plz 频道）
●LangChain 实现：LangSmith Deployment（cron/webhook）、Fleet（channels/schedules）、openclaw 的 “heartbeats”
完整案例：事件驱动的 Docs Agent

"""
Loop 3: Event-Driven Loop — 完整的事件驱动循环实现
通过多种触发器（Slack、定时任务、Webhook）自动运行 Agent
"""

import asyncio
import json
import os
import signal
import time
from datetime import datetime
from typing import Optional, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel


# ============================================================
# 1. 事件类型定义
# ============================================================

class EventType(str, Enum):
    SLACK_MESSAGE = "slack_message"
    CRON_SCHEDULE = "cron_schedule"
    WEBHOOK = "webhook"
    FILE_WATCHER = "file_watcher"
    GITHUB_EVENT = "github_event"


@dataclass
class AgentEvent:
    """Agent 事件"""
    event_type: EventType
    source: str  # 事件来源
    payload: dict  # 事件数据
    timestamp: datetime = field(default_factory=datetime.now)
    event_id: str = ""


# ============================================================
# 2. 触发器抽象
# ============================================================

class EventTrigger(ABC):
    """事件触发器基类"""

    def __init__(self, name: str, event_type: EventType):
        self.name = name
        self.event_type = event_type
        self._running = False
        self.callbacks: list[Callable[[AgentEvent], Awaitable[None]]] = []

    def on_event(self, callback: Callable[[AgentEvent], Awaitable[None]]):
        """注册事件回调"""
        self.callbacks.append(callback)

    @abstractmethod
    async def start(self):
        """启动触发器"""
        ...

    @abstractmethod
    async def stop(self):
        """停止触发器"""
        ...

    async def _emit(self, payload: dict):
        """发送事件"""
        event = AgentEvent(
            event_type=self.event_type,
            source=self.name,
            payload=payload,
        )
        for callback in self.callbacks:
            try:
                await callback(event)
            except Exception as e:
                print(f"[{self.name}] 回调错误: {e}")


# ============================================================
# 3. Slack 触发器
# ============================================================

class SlackTrigger(EventTrigger):
    """
    Slack 频道触发器：监听指定频道的消息，触发 Agent 运行

    模拟 Slack Events API 的集成
    """

    def __init__(self, channel: str, bot_token: str = None):
        super().__init__(name=f"slack:{channel}", event_type=EventType.SLACK_MESSAGE)
        self.channel = channel
        self.bot_token = bot_token or os.getenv("SLACK_BOT_TOKEN")
        self._ws = None

    async def start(self):
        """启动 Slack 监听"""
        self._running = True
        print(f"[{self.name}] 开始监听 Slack 频道: {self.channel}")

        # 实际环境中：
        # 1. 使用 slack_sdk 连接 Socket Mode
        # 2. 监听 message 事件
        # 3. 过滤目标频道
        # 这里用模拟实现演示架构

        async def simulate_messages():
            """模拟 Slack 消息（实际应使用 Slack SDK）"""
            import random
            mock_requests = [
                "请更新 /docs/api.md 的认证部分文档",
                "为 deployment 指南添加 Docker 配置节",
                "修复 /docs/tutorials/ 中的拼写错误",
                "在 README 中添加项目架构图",
            ]

            while self._running:
                await asyncio.sleep(random.randint(10, 30))  # 模拟消息间隔
                request = random.choice(mock_requests)

                await self._emit({
                    "channel": self.channel,
                    "user": "U123456",
                    "text": request,
                    "ts": str(time.time()),
                })

        asyncio.create_task(simulate_messages())

    async def stop(self):
        self._running = False
        print(f"[{self.name}] 已停止监听")


# ============================================================
# 4. 定时任务触发器（Cron）
# ============================================================

class CronTrigger(EventTrigger):
    """
    Cron 定时触发器：按计划自动运行 Agent

    支持：
    - 文档健康检查（每天）
    - 链接失效检测（每周）
    - 依赖更新检查（每 6 小时）
    """

    def __init__(self, name: str, interval_seconds: int, task_description: str):
        super().__init__(name=name, event_type=EventType.CRON_SCHEDULE)
        self.interval_seconds = interval_seconds
        self.task_description = task_description

    async def start(self):
        self._running = True
        print(f"[{self.name}] 启动定时任务: 每 {self.interval_seconds}s 执行")

        async def run_loop():
            while self._running:
                await self._emit({
                    "task": self.task_description,
                    "scheduled_at": datetime.now().isoformat(),
                    "interval": self.interval_seconds,
                })
                await asyncio.sleep(self.interval_seconds)

        asyncio.create_task(run_loop())

    async def stop(self):
        self._running = False
        print(f"[{self.name}] 已停止定时任务")


# ============================================================
# 5. Webhook 触发器
# ============================================================

from aiohttp import web  # 需要: pip install aiohttp


class WebhookTrigger(EventTrigger):
    """
    Webhook 触发器：接收 HTTP webhook 请求

    支持：
    - GitHub webhook（PR 创建、Issue 更新）
    - 自定义 API 调用
    - 第三方服务集成
    """

    def __init__(self, name: str, path: str, port: int = 8080):
        super().__init__(name=name, event_type=EventType.WEBHOOK)
        self.path = path
        self.port = port
        self._app = None
        self._runner = None

    async def start(self):
        self._running = True
        self._app = web.Application()
        self._app.router.add_post(self.path, self._handle_webhook)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "0.0.0.0", self.port)
        await site.start()

        print(f"[{self.name}] Webhook 监听: http://0.0.0.0:{self.port}{self.path}")

    async def _handle_webhook(self, request: web.Request):
        """处理 webhook 请求"""
        try:
            body = await request.json()
        except Exception:
            body = {"raw": await request.text()}

        # 验证签名（GitHub webhook 需要）
        # signature = request.headers.get("X-Hub-Signature-256")
        # if not self._verify_signature(body, signature):
        #     return web.Response(status=403)

        event_type = request.headers.get("X-GitHub-Event", "unknown")

        await self._emit({
            "event_type": event_type,
            "headers": dict(request.headers),
            "body": body,
            "source_ip": request.remote,
        })

        return web.Response(text="OK", status=200)

    async def stop(self):
        self._running = False
        if self._runner:
            await self._runner.cleanup()
        print(f"[{self.name}] Webhook 已停止")


# ============================================================
# 6. 事件驱动 Agent 引擎
# ============================================================

@dataclass
class AgentTask:
    """Agent 任务"""
    event: AgentEvent
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class EventDrivenAgent:
    """
    事件驱动的 Agent 引擎

    核心职责：
    1. 注册和管理触发器
    2. 接收事件 → 路由到合适的 Agent
    3. 管理 Agent 执行队列
    4. 记录执行历史和结果
    """

    def __init__(self):
        self.triggers: list[EventTrigger] = []
        self.tasks: list[AgentTask] = []
        self.running = False
        self._task_queue: asyncio.Queue = asyncio.Queue()
        self._workers: list[asyncio.Task] = []

        # 创建 Agent
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.agent = self._create_agent()

    def _create_agent(self):
        """创建 Docs Agent"""
        return create_agent(
            model=self.llm,
            tools=[clone_repo, read_file, write_file, create_pr],
            system_prompt="""你是一个由事件驱动的文档编写 Agent。
你会在后台持续运行，响应各种事件（Slack 消息、定时任务、Webhook）。
每个事件包含一个任务描述，你需要根据描述执行相应的文档操作。""",
        )

    def add_trigger(self, trigger: EventTrigger):
        """注册触发器"""
        trigger.on_event(self._handle_event)
        self.triggers.append(trigger)

    async def _handle_event(self, event: AgentEvent):
        """处理事件（放入队列）"""
        task = AgentTask(event=event)
        self.tasks.append(task)
        await self._task_queue.put(task)

        print(f"[Engine] 收到事件: {event.event_type.value} from {event.source}")

    async def _worker(self, worker_id: int):
        """工作线程：从队列取任务，执行 Agent"""
        while self.running:
            try:
                task = await asyncio.wait_for(
                    self._task_queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue

            print(f"[Worker-{worker_id}] 开始处理任务: {task.event.event_type.value}")

            try:
                task.status = "running"
                task.started_at = datetime.now()

                # 构建任务消息
                payload = task.event.payload
                if task.event.event_type == EventType.SLACK_MESSAGE:
                    user_message = payload.get("text", "")
                    channel = payload.get("channel", "unknown")
                    task_input = f"Slack 频道 #{channel} 收到请求:\n{user_message}"

                elif task.event.event_type == EventType.CRON_SCHEDULE:
                    task_input = f"定时任务: {payload.get('task', '')}"

                elif task.event.event_type == EventType.WEBHOOK:
                    body = payload.get("body", {})
                    task_input = f"Webhook 事件: {json.dumps(body, indent=2)}"

                else:
                    task_input = json.dumps(payload)

                # 执行 Agent
                result = self.agent.invoke({
                    "messages": [HumanMessage(content=task_input)]
                })

                # 提取结果
                for msg in reversed(result["messages"]):
                    if hasattr(msg, "content") and msg.content:
                        task.result = msg.content
                        break

                task.status = "completed"
                print(f"[Worker-{worker_id}] 任务完成")

            except Exception as e:
                task.status = "failed"
                task.result = str(e)
                print(f"[Worker-{worker_id}] 任务失败: {e}")

            finally:
                task.completed_at = datetime.now()

    async def start(self, num_workers: int = 3):
        """启动事件驱动引擎"""
        self.running = True
        print(f"[Engine] 启动事件驱动 Agent 引擎（{num_workers} 个工作线程）")

        # 启动工作线程
        for i in range(num_workers):
            worker = asyncio.create_task(self._worker(i))
            self._workers.append(worker)

        # 启动所有触发器
        for trigger in self.triggers:
            await trigger.start()

    async def stop(self):
        """停止引擎"""
        print("[Engine] 正在停止...")
        self.running = False

        # 停止触发器
        for trigger in self.triggers:
            await trigger.stop()

        # 等待工作线程
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)

        print("[Engine] 已停止")

    def get_status(self) -> dict:
        """获取引擎状态"""
        return {
            "running": self.running,
            "triggers": [t.name for t in self.triggers],
            "total_tasks": len(self.tasks),
            "pending": sum(1 for t in self.tasks if t.status == "pending"),
            "running": sum(1 for t in self.tasks if t.status == "running"),
            "completed": sum(1 for t in self.tasks if t.status == "completed"),
            "failed": sum(1 for t in self.tasks if t.status == "failed"),
        }


# ============================================================
# 7. 使用 LangSmith Deployment（生产级）
# ============================================================

def langsmith_deployment_example():
    """
    使用 LangSmith Deployment 配置生产级事件驱动

    LangSmith Deployment 支持：
    - Cron 定时任务
    - Webhook 触发
    - Fleet Channels（无代码 Agent 构建器）
    """
    deployment_config = {
        "name": "docs-agent-production",
        "agent": "docs_agent_v2",
        "triggers": [
            {
                "type": "cron",
                "schedule": "0 */6 * * *",  # 每 6 小时
                "task": "检查所有文档链接是否有效，修复失效链接"
            },
            {
                "type": "cron",
                "schedule": "0 9 * * 1",  # 每周一早上 9 点
                "task": "扫描文档仓库，生成文档健康报告"
            },
            {
                "type": "webhook",
                "path": "/github-webhook",
                "events": ["pull_request.opened", "pull_request.reopened"],
                "task": "检查新 PR 中是否包含文档变更，自动更新相关文档"
            },
            {
                "type": "fleet_channel",
                "channel": "#docs-plz",
                "platform": "slack",
                "task": "监听 Slack 频道的文档请求，自动执行"
            }
        ],
        "environment": {
            "GITHUB_TOKEN": "{{ secrets.GITHUB_TOKEN }}",
            "SLACK_BOT_TOKEN": "{{ secrets.SLACK_BOT_TOKEN }}"
        },
        "retry": {
            "max_attempts": 3,
            "backoff": "exponential"
        }
    }

    return deployment_config


# ============================================================
# 8. 使用示例
# ============================================================

async def main():
    """运行事件驱动 Agent 示例"""
    engine = EventDrivenAgent()

    # 注册触发器
    engine.add_trigger(SlackTrigger(channel="#docs-plz"))
    engine.add_trigger(CronTrigger(
        name="link-health-check",
        interval_seconds=3600,  # 每小时
        task_description="检查所有文档中的链接是否有效，修复失效链接"
    ))
    engine.add_trigger(CronTrigger(
        name="weekly-report",
        interval_seconds=86400,  # 每天
        task_description="生成文档更新周报，汇总本周所有文档变更"
    ))
    engine.add_trigger(WebhookTrigger(
        name="github-webhook",
        path="/webhook/github",
        port=8080
    ))

    # 启动引擎
    await engine.start(num_workers=3)

    # 运行 60 秒（演示目的）
    try:
        for i in range(12):
            await asyncio.sleep(5)
            status = engine.get_status()
            print(f"\n[状态] {json.dumps(status, indent=2)}")
    except KeyboardInterrupt:
        pass
    finally:
        await engine.stop()


if __name__ == "__main__":
    asyncio.run(main())


Loop 4：爬山循环（Hill Climbing Loop）
架构

生产运行 → 产生 Trace → 分析 Agent 检查 Trace → 发现问题 → 自动改进配置（prompt/工具/评分器）→ 回到 Loop 1

核心要点
●动机：前三个循环自动化工作，第四个自动化改进！
●核心机制：每次 Agent 运行产生 Trace → 分析 Agent 扫描 Trace → 根据发现重写 harness 配置（prompt、工具、评分器调整）
●LangChain 实现：LangSmith Engine（Trace 分析 Agent）
●关键洞察：返回箭头深入到内部直接更新 Agent 循环本身，外层循环每次迭代让内层循环更有效
●展望：可接入 RL 微调（开源模型）、改进辅助上下文（记忆和检索技能）
完整案例：爬山循环的自我改进 Agent

"""
Loop 4: Hill Climbing Loop — 完整的自我改进循环实现
通过分析生产 Trace，自动优化 Agent 的 prompt、工具和评分器配置
"""

import json
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from pydantic import BaseModel, Field


# ============================================================
# 1. Trace 数据结构
# ============================================================

class TraceStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    VERIFICATION_FAILED = "verification_failed"


@dataclass
class ToolCallTrace:
    """工具调用追踪"""
    tool_name: str
    input: dict
    output: str
    duration_ms: float
    error: Optional[str] = None
    retry_count: int = 0


@dataclass
class VerificationTrace:
    """验证追踪"""
    grader_name: str
    score: float
    grade: str
    feedback: str
    attempt_number: int


@dataclass
class AgentTrace:
    """Agent 运行追踪"""
    trace_id: str
    task_input: str
    status: TraceStatus
    final_output: Optional[str] = None
    error_message: Optional[str] = None
    tool_calls: List[ToolCallTrace] = field(default_factory=list)
    verifications: List[VerificationTrace] = field(default_factory=list)
    total_duration_ms: float = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    model_name: str = ""
    prompt_version: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


# ============================================================
# 2. Trace 存储和分析
# ============================================================

class TraceStore:
    """Trace 存储（模拟 LangSmith 的 Trace 存储）"""

    def __init__(self):
        self.traces: List[AgentTrace] = []

    def add(self, trace: AgentTrace):
        self.traces.append(trace)

    def query(
        self,
        status: Optional[TraceStatus] = None,
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> List[AgentTrace]:
        """查询 Trace"""
        results = self.traces

        if status:
            results = [t for t in results if t.status == status]
        if since:
            results = [t for t in results if t.timestamp >= since]

        return sorted(results, key=lambda t: t.timestamp, reverse=True)[:limit]

    def get_stats(self, since: Optional[datetime] = None) -> dict:
        """获取统计信息"""
        traces = self.query(since=since)

        if not traces:
            return {"total": 0}

        success = [t for t in traces if t.status == TraceStatus.SUCCESS]
        failed = [t for t in traces if t.status == TraceStatus.FAILED]
        verification_failed = [
            t for t in traces if t.status == TraceStatus.VERIFICATION_FAILED
        ]

        # 工具使用频率
        tool_usage = defaultdict(int)
        tool_errors = defaultdict(int)
        for t in traces:
            for tc in t.tool_calls:
                tool_usage[tc.tool_name] += 1
                if tc.error:
                    tool_errors[tc.tool_name] += 1

        # 验证失败模式
        grader_failures = defaultdict(int)
        for t in traces:
            for v in t.verifications:
                if v.grade != "PASS":
                    grader_failures[v.grader_name] += 1

        return {
            "total": len(traces),
            "success_rate": len(success) / len(traces),
            "failure_rate": len(failed) / len(traces),
            "verification_fail_rate": len(verification_failed) / len(traces),
            "avg_duration_ms": sum(t.total_duration_ms for t in traces) / len(traces),
            "avg_tokens": sum(t.total_tokens for t in traces) / len(traces),
            "avg_cost": sum(t.total_cost for t in traces) / len(traces),
            "tool_usage": dict(tool_usage),
            "tool_errors": dict(tool_errors),
            "grader_failures": dict(grader_failures),
            "common_failures": self._extract_common_failures(traces),
        }

    def _extract_common_failures(self, traces: List[AgentTrace], top_k: int = 5) -> list:
        """提取常见失败模式"""
        failure_patterns = defaultdict(int)

        for trace in traces:
            if trace.status == TraceStatus.FAILED:
                # 从错误信息中提取模式
                if trace.error_message:
                    patterns = [
                        "timeout", "connection refused", "permission denied",
                        "not found", "rate limit", "authentication failed",
                        "out of memory", "invalid format", "syntax error",
                    ]
                    for pattern in patterns:
                        if pattern in trace.error_message.lower():
                            failure_patterns[pattern] += 1
                            break
                    else:
                        failure_patterns["other"] += 1

            elif trace.status == TraceStatus.VERIFICATION_FAILED:
                for v in trace.verifications:
                    if v.grade != "PASS":
                        failure_patterns[f"grader:{v.grader_name}"] += 1

        return sorted(
            [{"pattern": k, "count": v} for k, v in failure_patterns.items()],
            key=lambda x: x["count"], reverse=True
        )[:top_k]


# ============================================================
# 3. 分析 Agent（Engine）
# ============================================================

class TraceAnalysisResult(BaseModel):
    """Trace 分析结果"""
    summary: str = Field(description="总体分析摘要")
    issues: List[Dict[str, Any]] = Field(description="发现的问题列表")
    recommendations: List[Dict[str, Any]] = Field(description="改进建议")
    priority: str = Field(description="优先级: critical/high/medium/low")


class TraceAnalyzer:
    """
    Trace 分析 Agent（LangSmith Engine 的模拟实现）

    职责：
    1. 分析 Trace 数据，识别失败模式
    2. 发现性能瓶颈
    3. 生成改进建议
    """

    ANALYSIS_PROMPT = """你是一个 Agent 性能分析专家。请分析以下 Agent 运行 Trace 数据，找出问题和改进机会。

分析维度：
1. **失败模式**：哪些类型的任务容易失败？失败原因是什么？
2. **工具使用**：哪些工具使用频率高但错误率高？是否缺少工具？
3. **验证问题**：评分器反复拒绝输出的原因是什么？
4. **性能瓶颈**：哪些步骤耗时最长？是否可以优化？
5. **Prompt 问题**：模型是否出现误解、幻觉或方向偏差？

请输出 JSON 格式的分析结果：
{
    "summary": "总体分析...",
    "issues": [
        {
            "id": "issue-001",
            "severity": "critical|high|medium|low",
            "category": "prompt|tool|grader|performance|other",
            "description": "问题描述",
            "evidence": ["trace_id_1", "trace_id_2"],
            "frequency": 5,
            "impact": "对成功率的影响描述"
        }
    ],
    "recommendations": [
        {
            "id": "rec-001",
            "target": "prompt|tool|grader|config",
            "action": "具体修改建议",
            "expected_improvement": "预期改进效果",
            "related_issue": "issue-001"
        }
    ],
    "priority": "critical|high|medium|low"
}
"""

    def __init__(self, model: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=model, temperature=0)

    def analyze(self, traces: List[AgentTrace], stats: dict) -> TraceAnalysisResult:
        """分析 Trace 数据"""
        # 构建分析输入
        trace_summaries = []
        for t in traces[:50]:  # 限制最近 50 条
            trace_summaries.append({
                "trace_id": t.trace_id,
                "status": t.status.value,
                "task": t.task_input[:200],
                "error": t.error_message[:200] if t.error_message else None,
                "duration_ms": t.total_duration_ms,
                "tool_calls": [
                    {"name": tc.tool_name, "error": tc.error}
                    for tc in t.tool_calls
                ],
                "verifications": [
                    {"grader": v.grader_name, "grade": v.grade, "score": v.score}
                    for v in t.verifications
                ],
            })

        analysis_input = f"""
统计数据：
{json.dumps(stats, indent=2)}

最近 Trace 摘要（{len(trace_summaries)} 条）：
{json.dumps(trace_summaries, indent=2)}

请分析以上数据，找出问题和改进建议。
"""

        messages = [
            SystemMessage(content=self.ANALYSIS_PROMPT),
            HumanMessage(content=analysis_input),
        ]

        response = self.llm.invoke(messages)

        # 解析响应
        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                data = json.loads(json_match.group())
                return TraceAnalysisResult(**data)
        except Exception as e:
            print(f"分析结果解析失败: {e}")

        return TraceAnalysisResult(
            summary="分析失败",
            issues=[],
            recommendations=[],
            priority="low"
        )


# ============================================================
# 4. 配置优化器
# ============================================================

@dataclass
class AgentConfig:
    """Agent 配置（可被爬山循环优化）"""
    prompt_version: str = "v1"
    system_prompt: str = ""
    tool_configs: Dict[str, dict] = field(default_factory=dict)
    grader_configs: Dict[str, dict] = field(default_factory=dict)
    max_retries: int = 3
    temperature: float = 0
    model_name: str = "gpt-4o"


class ConfigOptimizer:
    """
    配置优化器：根据分析结果自动改进 Agent 配置

    可优化的维度：
    1. Prompt 内容
    2. 工具描述和参数
    3. 评分器标准
    4. 重试策略
    5. 模型选择
    """

    OPTIMIZER_PROMPT = """你是一个 Agent 配置优化专家。根据 Trace 分析结果，请优化 Agent 的配置。

优化目标：
1. 提高任务成功率
2. 减少验证失败次数
3. 降低平均延迟
4. 减少 token 消耗

当前配置和问题：
{context}

请输出优化后的配置（JSON）：
{
    "system_prompt": "优化后的系统提示词...",
    "tool_improvements": {
        "tool_name": {
            "description": "改进后的工具描述",
            "suggested_changes": "具体修改说明"
        }
    },
    "grader_improvements": {
        "grader_name": {
            "rubric": "改进后的评分标准",
            "threshold": 0.85
        }
    },
    "config_changes": {
        "max_retries": 5,
        "temperature": 0.1
    },
    "change_log": "变更说明..."
}
"""

    def __init__(self, model: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=model, temperature=0)

    def optimize(
        self,
        current_config: AgentConfig,
        analysis: TraceAnalysisResult,
        stats: dict,
    ) -> Dict[str, Any]:
        """根据分析结果优化配置"""
        context = f"""
当前配置：
- Prompt 版本: {current_config.prompt_version}
- 系统提示词: {current_config.system_prompt[:500]}
- 最大重试: {current_config.max_retries}
- 模型: {current_config.model_name}

分析结果：
- 成功率: {stats.get('success_rate', 0):.1%}
- 验证失败率: {stats.get('verification_fail_rate', 0):.1%}
- 平均延迟: {stats.get('avg_duration_ms', 0):.0f}ms

发现的问题：
{json.dumps([i for i in analysis.issues], indent=2)}

改进建议：
{json.dumps([r for r in analysis.recommendations], indent=2)}
"""

        messages = [
            SystemMessage(content=self.OPTIMIZER_PROMPT.format(context=context)),
            HumanMessage(content="请生成优化后的配置。"),
        ]

        response = self.llm.invoke(messages)

        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            print(f"优化配置解析失败: {e}")

        return {"system_prompt": current_config.system_prompt, "change_log": "优化失败"}


# ============================================================
# 5. 爬山循环引擎
# ============================================================

class HillClimbingLoop:
    """
    爬山循环：持续分析 Trace，自动改进 Agent 配置

    工作流程：
    1. 收集生产 Trace
    2. 分析 Trace 数据（Engine）
    3. 生成优化配置
    4. 通过人工审核（可选）
    5. 部署新配置
    6. 回到步骤 1
    """

    def __init__(
        self,
        trace_store: TraceStore,
        analyzer: TraceAnalyzer,
        optimizer: ConfigOptimizer,
        require_human_review: bool = True,
        analysis_interval_hours: int = 24,
    ):
        self.trace_store = trace_store
        self.analyzer = analyzer
        self.optimizer = optimizer
        self.require_human_review = require_human_review
        self.analysis_interval_hours = analysis_interval_hours

        self.current_config = AgentConfig(
            prompt_version="v1",
            system_prompt="""你是一个专业的文档编写 Agent。
请确保：
1. 所有链接可解析
2. 代码示例语法正确
3. 文档结构清晰
4. 内容覆盖所有请求要点
""",
        )

        self.config_history: List[Dict[str, Any]] = []
        self.analysis_history: List[TraceAnalysisResult] = []

    async def run_cycle(self) -> Dict[str, Any]:
        """执行一次爬山循环"""
        print(f"\n{'='*60}")
        print(f" 爬山循环 — 开始分析周期")
        print(f"{'='*60}")

        # 1. 查询最近的 Trace
        since = datetime.now() - timedelta(hours=self.analysis_interval_hours)
        recent_traces = self.trace_store.query(since=since, limit=200)
        stats = self.trace_store.get_stats(since=since)

        print(f"  分析 {len(recent_traces)} 条 Trace")
        print(f"  当前成功率: {stats.get('success_rate', 0):.1%}")

        if len(recent_traces) < 10:
            print("  Trace 数量不足，跳过分析")
            return {"status": "skipped", "reason": "insufficient_traces"}

        # 2. 分析 Trace
        print("  正在分析 Trace...")
        analysis = self.analyzer.analyze(recent_traces, stats)
        self.analysis_history.append(analysis)

        print(f"  分析结果: {analysis.priority} 优先级")
        print(f"  发现 {len(analysis.issues)} 个问题")

        if not analysis.issues:
            print("  无问题发现，跳过优化")
            return {"status": "no_issues", "analysis": analysis}

        # 3. 生成优化建议
        print("  正在生成优化建议...")
        optimization = self.optimizer.optimize(
            self.current_config, analysis, stats
        )

        # 4. 人工审核（可选）
        if self.require_human_review:
            print("  等待人工审核...")
            approved = await self._human_review(analysis, optimization)
            if not approved:
                return {
                    "status": "rejected",
                    "analysis": analysis,
                    "optimization": optimization,
                }

        # 5. 应用新配置
        print("  正在应用新配置...")
        self._apply_optimization(optimization)

        return {
            "status": "applied",
            "analysis": analysis,
            "optimization": optimization,
            "new_config": {
                "prompt_version": self.current_config.prompt_version,
                "system_prompt": self.current_config.system_prompt[:500],
            },
        }

    async def _human_review(
        self,
        analysis: TraceAnalysisResult,
        optimization: Dict[str, Any],
    ) -> bool:
        """人工审核（模拟）"""
        # 实际环境中，这里会：
        # 1. 创建 GitHub Issue 或 PR
        # 2. 通知相关人员审核
        # 3. 等待审批结果

        print(f"\n  {'='*50}")
        print(f"  人工审核请求")
        print(f"  {'='*50}")
        print(f"  优先级: {analysis.priority}")
        print(f"  问题数量: {len(analysis.issues)}")
        print(f"  变更说明: {optimization.get('change_log', 'N/A')}")

        # 模拟：高优先级问题自动通过，其他需要审核
        if analysis.priority == "critical":
            print("  关键问题，自动批准")
            return True

        # 模拟审核通过（实际应等待人工确认）
        print("  模拟审核：通过")
        return True

    def _apply_optimization(self, optimization: Dict[str, Any]):
        """应用优化后的配置"""
        old_version = self.current_config.prompt_version

        # 更新 prompt
        if optimization.get("system_prompt"):
            self.current_config.system_prompt = optimization["system_prompt"]

        # 更新工具配置
        if optimization.get("tool_improvements"):
            self.current_config.tool_configs.update(optimization["tool_improvements"])

        # 更新评分器配置
        if optimization.get("grader_improvements"):
            self.current_config.grader_configs.update(optimization["grader_improvements"])

        # 更新其他配置
        if optimization.get("config_changes"):
            changes = optimization["config_changes"]
            if "max_retries" in changes:
                self.current_config.max_retries = changes["max_retries"]
            if "temperature" in changes:
                self.current_config.temperature = changes["temperature"]

        # 版本号递增
        version_num = int(old_version.replace("v", ""))
        self.current_config.prompt_version = f"v{version_num + 1}"

        # 记录历史
        self.config_history.append({
            "from_version": old_version,
            "to_version": self.current_config.prompt_version,
            "timestamp": datetime.now().isoformat(),
            "changes": optimization.get("change_log", ""),
        })

        print(f"  配置已更新: {old_version} → {self.current_config.prompt_version}")


# ============================================================
# 6. 使用 LangSmith Engine（生产级）
# ============================================================

def langsmith_engine_example():
    """
    使用 LangSmith Engine 进行生产级 Trace 分析

    LangSmith Engine 的核心能力：
    1. 自动分析生产 Trace
    2. 检测异常模式
    3. 生成改进建议
    4. 集成 GitHub Issues 进行跟踪
    """
    engine_config = {
        "name": "docs-agent-engine",
        "agent": "docs_agent_v2",
        "analysis": {
            "schedule": "daily",  # 每天分析一次
            "lookback_hours": 24,  # 分析最近 24 小时的 Trace
            "min_traces": 10,  # 最少 Trace 数量
        },
        "detection_rules": [
            {
                "name": "success_rate_drop",
                "condition": "success_rate < 0.8",
                "severity": "critical",
                "action": "file_issue"
            },
            {
                "name": "latency_spike",
                "condition": "avg_duration_ms > 30000",
                "severity": "high",
                "action": "file_issue"
            },
            {
                "name": "tool_error_rate",
                "condition": "tool_errors > 5",
                "severity": "medium",
                "action": "file_issue"
            },
            {
                "name": "verification_fail_spike",
                "condition": "verification_fail_rate > 0.3",
                "severity": "high",
                "action": "file_issue"
            }
        ],
        "auto_improvements": [
            "prompt_optimization",
            "tool_description_improvement",
            "grader_threshold_adjustment",
        ],
        "human_review_required": [
            "prompt_optimization",  # Prompt 修改需要人工审核
            "grader_threshold_adjustment",
        ],
        "notifications": {
            "slack_channel": "#agent-health",
            "email": "agent-team@company.com"
        }
    }

    return engine_config


# ============================================================
# 7. 完整使用示例
# ============================================================

def generate_mock_traces(store: TraceStore, count: int = 100):
    """生成模拟 Trace 数据"""
    import random
    import uuid

    tasks = [
        "更新 API 文档的认证部分",
        "为部署指南添加 Docker 配置",
        "修复教程中的拼写错误",
        "在 README 中添加项目架构图",
        "为 CLI 工具编写使用文档",
        "更新依赖列表和版本说明",
        "添加 FAQ 页面",
        "修复文档中的失效链接",
        "为 WebSocket API 编写文档",
        "更新贡献指南",
    ]

    tool_names = ["clone_repo", "read_file", "write_file", "create_pr", "search_docs"]
    grader_names = ["LinkChecker", "CodeBlockValidator", "LLMJudge"]

    for i in range(count):
        task = random.choice(tasks)

        # 模拟成功率：80% 成功，15% 验证失败，5% 完全失败
        rand = random.random()
        if rand < 0.80:
            status = TraceStatus.SUCCESS
        elif rand < 0.95:
            status = TraceStatus.VERIFICATION_FAILED
        else:
            status = TraceStatus.FAILED

        # 生成工具调用
        tool_calls = []
        for _ in range(random.randint(1, 5)):
            tn = random.choice(tool_names)
            tool_calls.append(ToolCallTrace(
                tool_name=tn,
                input={"dummy": "input"},
                output="tool output",
                duration_ms=random.uniform(100, 5000),
                error="timeout" if status == TraceStatus.FAILED and random.random() < 0.3 else None,
                retry_count=random.randint(0, 2),
            ))

        # 生成验证记录
        verifications = []
        for _ in range(random.randint(1, 3)):
            gn = random.choice(grader_names)
            if status == TraceStatus.VERIFICATION_FAILED and random.random() < 0.5:
                verifications.append(VerificationTrace(
                    grader_name=gn,
                    score=random.uniform(0.3, 0.7),
                    grade="FAIL",
                    feedback="需要改进",
                    attempt_number=random.randint(1, 3),
                ))
            else:
                verifications.append(VerificationTrace(
                    grader_name=gn,
                    score=random.uniform(0.8, 1.0),
                    grade="PASS",
                    feedback="通过",
                    attempt_number=1,
                ))

        trace = AgentTrace(
            trace_id=str(uuid.uuid4()),
            task_input=task,
            status=status,
            final_output="文档已更新" if status == TraceStatus.SUCCESS else None,
            error_message="timeout" if status == TraceStatus.FAILED else None,
            tool_calls=tool_calls,
            verifications=verifications,
            total_duration_ms=random.uniform(5000, 60000),
            total_tokens=random.randint(500, 5000),
            total_cost=random.uniform(0.01, 0.5),
            model_name="gpt-4o",
            prompt_version="v1",
            timestamp=datetime.now() - timedelta(hours=random.randint(0, 24)),
        )

        store.add(trace)


async def main():
    """运行爬山循环示例"""
    # 初始化
    trace_store = TraceStore()
    analyzer = TraceAnalyzer(model="gpt-4o")
    optimizer = ConfigOptimizer(model="gpt-4o")

    loop = HillClimbingLoop(
        trace_store=trace_store,
        analyzer=analyzer,
        optimizer=optimizer,
        require_human_review=False,  # 演示模式，跳过审核
        analysis_interval_hours=24,
    )

    # 生成模拟 Trace
    print("生成模拟 Trace 数据...")
    generate_mock_traces(trace_store, count=100)

    stats = trace_store.get_stats()
    print(f"当前统计: 成功率={stats.get('success_rate', 0):.1%}")

    # 运行爬山循环
    result = await loop.run_cycle()

    print(f"\n{'='*60}")
    print(f" 爬山循环结果")
    print(f"{'='*60}")
    print(f"状态: {result['status']}")

    if result['status'] == 'applied':
        print(f"新版本: {loop.current_config.prompt_version}")
        print(f"配置历史: {len(loop.config_history)} 次变更")

        if result.get('analysis'):
            print(f"\n发现问题:")
            for issue in result['analysis'].issues:
                print(f"  - [{issue.get('severity', '?')}] {issue.get('description', '')[:100]}")

        if result.get('optimization'):
            print(f"\n变更说明:")
            print(f"  {result['optimization'].get('change_log', 'N/A')}")


if __name__ == "__main__":
    asyncio.run(main())


人类监督与专家介入
文章强调：自动化不意味着把人排除在循环之外。 每一层都有自然的人类介入点：
循环层级	人类介入方式	实现方式
Agent 循环	在执行敏感操作 / 工具调用前要求人类确认	interrupt_before 钩子
验证循环	对于敏感工作流，人类可以充当评分器	人工 Grader
应用/事件循环	人类在输出返回给最终用户前审批	审批中间件
爬山循环	Harness 改进在部署前经过人类审查	人工审核流程

"""
Human-in-the-Loop — 完整的人类介入实现
"""

from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt


# ============================================================
# 1. Agent 循环：敏感操作前要求人类确认
# ============================================================

def create_agent_with_human_approval():
    """创建带人类审批的 Agent"""

    SENSITIVE_TOOLS = {"create_pr", "write_file", "delete_file"}

    def before_tool_hook(state: dict, tool_name: str) -> dict:
        """工具调用前的人类确认钩子"""
        if tool_name in SENSITIVE_TOOLS:
            # 中断执行，等待人类确认
            approval = interrupt({
                "message": f"Agent 即将调用敏感工具: {tool_name}",
                "tool_input": state.get("pending_tool_input"),
                "options": ["approve", "reject", "modify"],
            })

            if approval == "reject":
                raise Exception(f"人类拒绝了 {tool_name} 调用")
            elif approval == "modify":
                state["pending_tool_input"] = interrupt({
                    "message": "请提供修改后的工具输入",
                })

        return state

    agent = create_agent(
        model=ChatOpenAI(model="gpt-4o"),
        tools=[clone_repo, read_file, write_file, create_pr],
        before_tool=before_tool_hook,
        checkpointer=MemorySaver(),
    )

    return agent


# ============================================================
# 2. 验证循环：人类作为评分器
# ============================================================

class HumanGrader:
    """人类评分器：将输出发送给人类审核"""

    def __init__(self, review_channel: str = "#agent-review"):
        self.review_channel = review_channel

    def check(self, output: str) -> GradingResult:
        """请求人类审核"""
        # 将输出发送到审核频道，等待人工反馈
        review_request = {
            "channel": self.review_channel,
            "output": output,
            "timestamp": datetime.now().isoformat(),
            "review_id": f"review-{int(time.time())}",
        }

        # 模拟：发送审核请求并等待响应
        print(f"\n[人类审核] 输出已发送到 {self.review_channel}")
        print(f"[人类审核] 输出预览: {output[:200]}...")

        # 在真实系统中，这里会：
        # 1. 发送 Slack 消息到审核频道
        # 2. 等待人类点击 Approve/Reject/Modify 按钮
        # 3. 返回审核结果

        # 模拟审核通过
        return GradingResult(
            grade=Grade.PASS,
            score=1.0,
            feedback="人类审核通过",
            details={"review_id": review_request["review_id"]}
        )


# ============================================================
# 3. 应用循环：输出审批
# ============================================================

def create_output_approval_middleware():
    """创建输出审批中间件"""

    def after_agent_hook(state: dict) -> dict:
        """Agent 执行后的输出审批"""
        messages = state.get("messages", [])
        if not messages:
            return state

        last_msg = messages[-1]
        output = getattr(last_msg, "content", "")

        if not output:
            return state

        # 请求人类审批
        approval = interrupt({
            "message": "Agent 产生了输出，请审批",
            "output_preview": output[:500],
            "options": ["approve", "reject", "request_changes"],
        })

        if approval == "reject":
            state["messages"].append(
                HumanMessage(content="输出被拒绝。请重新生成。")
            )
        elif approval == "request_changes":
            changes = interrupt({
                "message": "请描述需要的修改",
            })
            state["messages"].append(
                HumanMessage(content=f"请根据以下要求修改: {changes}")
            )

        return state

    return after_agent_hook


# ============================================================
# 4. 爬山循环：改进前的人类审查
# ============================================================

class HumanReviewGate:
    """人类审查门：爬山循环的改进建议需要人类审批"""

    def __init__(self, reviewers: list[str] = None):
        self.reviewers = reviewers or ["agent-team@company.com"]

    def review(self, analysis: TraceAnalysisResult, optimization: dict) -> dict:
        """提交审查并等待结果"""
        review = {
            "id": f"review-{int(time.time())}",
            "analysis": analysis.dict(),
            "optimization": optimization,
            "reviewers": self.reviewers,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
        }

        # 在真实系统中：
        # 1. 创建 GitHub Issue 或 PR
        # 2. 分配审查者
        # 3. 等待审批

        print(f"\n[审查] 改进建议已提交")
        print(f"[审查] 审查者: {', '.join(self.reviewers)}")
        print(f"[审查] 优先级: {analysis.priority}")

        # 模拟审查通过
        review["status"] = "approved"
        review["approved_by"] = self.reviewers[0]
        review["approved_at"] = datetime.now().isoformat()

        return review


四层循环总览表
循环	做什么	影响	LangChain 原语	关键代码
1. Agent 循环	模型反复调用工具直到任务完成	自动化工作	create_agent，任意 LangChain 支持的模型	create_agent(model=llm, tools=[...])
2. 验证循环	Agent 运行，输出按标准评分，不合格则带反馈重试	保证工作质量和正确性	RubricMiddleware	RubricMiddleware(rubric="...", max_retries=3)
3. 事件驱动循环	事件触发 Agent 运行并更新真实系统	规模化自动工作	LangSmith Deployment（cron/webhook）或 Fleet Channels	SlackTrigger, CronTrigger, WebhookTrigger
4. 爬山循环	生产 Trace 输入分析 Agent，改进 harness 配置	Harness 持续改进	LangSmith Engine	TraceAnalyzer + ConfigOptimizer

核心结论
1.“AI 领导者如 Steipete、Boris、Andrej 都得出同一结论：Agent 的潜力在于你围绕它们构建的循环。”
2.“Loop 1 和 Loop 2 我们已经思考了一段时间。但焦点应该转向 Loop 3 和 Loop 4 —— 价值通过将 Agent 嵌入你的生态系统并持续根据你的标准改进而复合增长。”
3.Satya 的组织层面观点：“早期构建学习循环的公司 —— 人类判断和 token 资本共同复合增长 —— 将建立难以复制的优势。”
4.循环是模式，优化什么取决于你：Prompt 和工具配置是最容易改进的，但也可以扩展到 RL 微调（开源模型）、记忆和检索技能优化等。

参考资源
●https://docs.langchain.com/oss/python/deepagents/quickstart 
●https://docs.langchain.com/oss/python/langchain/agents 
●https://docs.langchain.com/oss/python/langchain/middleware 
●https://docs.langchain.com/langsmith/deployments 
●https://docs.langchain.com/langsmith/engine 
●https://docs.langchain.com/langsmith/fleet 

总结：这篇文章本质上提出了一套 Agent 工程的成熟度模型：
●Level 1：你能让 Agent 干活（基本能力）
●Level 2：你能保证 Agent 干对活（质量保证）
●Level 3：你能让 Agent 自动干活（规模化）
●Level 4：你能让 Agent 越干越好（持续进化）
最关键的洞见是：真正拉开差距的不是模型本身，而是 Loop 3（集成）和 Loop 4（自我改进） —— 这两个循环让 Agent 从"一次性工具"变成"持续进化的系统组件"，价值随时间复合增长。


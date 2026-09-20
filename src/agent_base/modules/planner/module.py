"""planner 模块的 AgentModule 契约实现（B6）。"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from agent_base.core.contracts import Graph, ModuleContext
from agent_base.modules.planner.graph import build_planner_graph


class PlannerModule:
    """plan-execute-replan 模块（M9）：拆解、逐条执行、预算内重规划。"""

    name = "planner"
    description = "把复杂请求拆解为子任务并逐条执行（plan-execute-replan）"

    def build_graph(self, ctx: ModuleContext) -> Graph:
        return build_planner_graph(ctx)

    def get_tools(self) -> list[BaseTool]:
        # 复用共享工具池；本模块不贡献私有工具。
        return []


module = PlannerModule()

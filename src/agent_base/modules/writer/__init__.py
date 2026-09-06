"""``writer`` 样板模块——第二个内置 agent（阶段 4）。

它有两个用途：
1. 作为第二个 supervisor sub-agent，让多 Agent 模板可以演示性地在专家
   之间路由（chat vs writer）；
2. 证明新增模块遵循五步 module guide 且零基座改动：一个目录、一份契约
   实现、以及一条 AGENT_MODULES 条目。
"""

from agent_base.modules.writer.module import WriterModule

module = WriterModule()

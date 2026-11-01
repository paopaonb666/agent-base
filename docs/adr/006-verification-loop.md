# ADR-006：Agent 输出验证循环作为基座质量契约（Loop 2）

- 状态：已采纳（0.3.x）
- 日期：2026-11-01
- 相关：ADR-002 模块契约、`docs/agent-dev/agent-dev-knowledge.md` §Loop 2 验证循环、
  `docs/agent-dev/integration-review.md` §3.3、`modules/planner/`（check/replan 先例）、
  `examples/custom_memory_port.py`（Port 注入模式先例）

## 背景

循环工程方法论的四层循环中，Loop 2（验证循环）回答"怎么保证 Agent 干对活"：
给 Agent 输出加评分器（Grader），不合格则带反馈重试，重试耗尽返回历史最佳。
本项目的成本治理（T4）只回答"花多少钱"，不回答"值不值"——LLM 调用成功 ≠
输出合格。

项目内部已有两处局部实践：
- `modules/planner/` 的 check → replan 节点（计划质量校验 + 重规划）；
- 记忆形成管线的整合裁决（`CONSOLIDATION_PROMPT` 输出封闭操作集，失败降级 NONE）。

但两者都是内联逻辑，没有可复用的评分契约与循环引擎。

## 决策

1. **新增 `core/verification.py`**，提供四个原语（全部基座级、零业务绑定）：
   - `Grade` 枚举：`PASS / FAIL / NEEDS_REVISION`（照外部方法论三态）；
   - `GradingResult` frozen dataclass：`grade / score(0..1) / feedback / components`；
   - `Grader` 协议：`async check(output: str) -> GradingResult`——评分器的最小面；
   - `VerificationLoop`：`__init__(graders, *, max_retries=3, pass_threshold=0.8,
     require_all_pass=False)`；`run(task, agent_fn)` 流程照外部方法论——
     逐轮执行（失败轮反馈拼入输入）→ 全部评分器打分 → 加权聚合（阈值判 PASS /
     ≥0.5 判 NEEDS_REVISION）→ 通过即返回；耗尽重试返回**历史最佳结果**
     （best_score 跟踪），并附完整 `attempts` 历史。
2. **评分器两类都支持**：确定性评分器（正则/语法检查等纯代码）直接实现协议；
   代理式评分器提供 `LLMJudgeGrader` 助手——prompt 强制 JSON verdict
   （`{overall_score, verdict, feedback}`），正则提取解析，解析失败一律 FAIL
   （宁可错杀不放水）。LLM 经构造注入（任何 `ainvoke(messages)` 对象，
   与 `ChatInvokerPort` 同形），不绑具体模型类。
3. **可选能力，不强制接线**：模块自行 import 取用（planner 不改造——它已有
   check/replan；记忆形成管线不改造——它有自己的封闭决策集）。本决策只固定
   契约与引擎，接线由后续模块按需进行。
4. **聚合语义**：`require_all_pass=False`（默认）时按算术平均分对照阈值分级；
   `True` 时全部 PASS 才 PASS。评分器异常按 FAIL 计（照外部方法论），不中断
   其他评分器。

## 被否方案

- **基于 langchain `RubricMiddleware` / `AgentMiddleware`**：项目不用
  AgentMiddleware（自研 `build_model_input` 钩子已覆盖模型调用前拦截），
  引入中间件框架会形成两套钩子体系（见评审报告 §3.6）。
- **把验证循环接进共享图工厂**（每个模块自动获得输出校验）：模块输出形态各异
  （chat 文本 / planner 计划 / 工具结果），默认评分器无从定义；强制接线只会
  产出"永远 PASS"的摆设评分器。
- **同步重试语义放在 SSE 层**（HTTP 请求内循环调 LLM 直到合格）：与流式契约
  冲突且失控重试会放大成本；验证循环作为独立组件由调用方控制预算更安全。

## 后果

正面：

- "花多少钱"（T4）之外补上"值不值"的度量原语，二者组合才是完整质量治理；
- 契约先行：后续模块（如知识库摄取质检、报告生成校验）可直接实现 `Grader`；
- 与外部方法论语义一一对应，参考资料可直译。

代价（明示）：

- 短期内是"建而未用"的契约库（靠测试与文档维持），首个真实接线出现前存在
  设计偏差风险；
- `VerificationLoop.run` 的 `agent_fn` 形参约定为 `async (task: str, feedback:
  str | None) -> str`，非图执行形态——接 LangGraph 图需要调用方自行包装，
  本决策不提供通用图包装器（各模块拓扑差异过大）。

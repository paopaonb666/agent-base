package io.github.paopaonb666.agentbase.entrypoints.server.routes;

import java.util.ArrayList;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;

import io.github.paopaonb666.agentbase.core.AgentRuntime;
import io.github.paopaonb666.agentbase.core.ThreadId;
import io.github.paopaonb666.agentbase.core.text.CharMetrics;
import io.github.paopaonb666.agentbase.core.context.ContextEngine;
import io.github.paopaonb666.agentbase.core.graph.CompiledGraph;
import io.github.paopaonb666.agentbase.core.graph.GraphState;
import io.github.paopaonb666.agentbase.core.graph.RunConfig;
import io.github.paopaonb666.agentbase.core.llm.LlmFactory;
import io.github.paopaonb666.agentbase.core.prompt.PromptComposer;
import io.github.paopaonb666.agentbase.core.storage.ThreadIndexRecord;
import io.github.paopaonb666.agentbase.entrypoints.server.Attachments;
import io.github.paopaonb666.agentbase.entrypoints.server.AuthBackends;
import io.github.paopaonb666.agentbase.entrypoints.server.Background;
import io.github.paopaonb666.agentbase.entrypoints.server.ErrorCode;
import io.github.paopaonb666.agentbase.entrypoints.server.HttpError;
import io.github.paopaonb666.agentbase.entrypoints.server.MessageSerializers;
import io.github.paopaonb666.agentbase.entrypoints.server.RateLimiters;
import io.github.paopaonb666.agentbase.entrypoints.server.RunGate;
import io.github.paopaonb666.agentbase.entrypoints.server.SseStreams;
import io.github.paopaonb666.agentbase.extensions.costmeter.CostBudgetExceededException;
import io.github.paopaonb666.agentbase.extensions.costmeter.CostGovernor;
import io.github.paopaonb666.agentbase.extensions.filestore.UploadedFileInfo;
import io.github.paopaonb666.agentbase.extensions.filestore.UploadedFileStore;

/**
 * 对话与会话线程端点：invoke（SSE）、线程列表/历史/删除、工具审计、
 * 计划状态。S1 线程作用域：每个线程在 thread_index 表里有属主记录
 * （invoke 时 upsert）；列表只返回当前用户的线程；历史/删除/审计端点
 * 校验属主（404 掩蔽）。索引缺席（后端不可用）时：<b>production 拒绝
 * 降级服务</b>（#82 fail-safe 而非 fail-open——无索引即无属主校验，
 * 功能降级不能悄悄变成越权数据暴露面，503 + 修复指引）；development
 * 保留旧行为（全库扫描 + 无属主校验，本地开发豁免）并 WARN。
 *
 * <p>对应 Python 侧 {@code routes/agents.py}，端点/状态码/文案逐条对齐。
 */
@RestController
@RequestMapping("/v1/agents")
public class AgentsRoutes {

    private static final org.slf4j.Logger LOG =
            org.slf4j.LoggerFactory.getLogger(AgentsRoutes.class);

    // 线程列表端点的扫描/数量上限（旧版回退路径仍用）。
    private static final int THREAD_LIST_SCAN_LIMIT = 2000;
    private static final int THREAD_LIST_TOTAL_LIMIT = 20_000;
    private static final int THREAD_LIST_PAGE_MAX = 200;
    /** 单次取数上限：offset+limit 的扫描预算（#204）。 */
    private static final int THREAD_LIST_FETCH_MAX = 1000;

    /** mode 单源（#220）：chat=直答、plan=任务拆解、auto=难度分流——校验、错误文案与 /v1/modules 发现面共用。 */
    public static final List<String> MODES = List.of("chat", "plan", "auto");

    private final AgentRuntime runtime;
    private final AuthBackends.AuthBackend auth;
    private final CostGovernor governor;
    /** #78 per-user 令牌桶（invoke 入口 enforcement；0 = 不限）。 */
    private final RateLimiters.PerUserRateLimiter rateLimiter;
    /** #79 全局 run 并发闸门（0 = 不限）。 */
    private final RunGate runGate;

    public AgentsRoutes(AgentRuntime runtime, AuthBackends.AuthBackend auth,
                        org.springframework.beans.factory.ObjectProvider<CostGovernor> governorProvider) {
        this.runtime = runtime;
        this.auth = auth;
        // COST_ENABLED=false 时 governor bean 缺席（null）——可选注入。
        this.governor = governorProvider.getIfAvailable();
        // #78 per-user 限流：0 = 不限（本机开发），production 由
        // ensureProductionReady 强制显式配置。
        this.rateLimiter = new RateLimiters.PerUserRateLimiter(runtime.settings().rateLimitPerMinute);
        // #79 全局 run 并发闸门：默认 32，0 = 不限。
        this.runGate = new RunGate(runtime.settings().agentRunConcurrency);
    }

    private String user(String userIdHeader, String sigHeader, String tsHeader, String nonceHeader) {
        return auth.authenticate(userIdHeader, sigHeader, tsHeader, nonceHeader, runtime.settings());
    }

    /** 按模块取图；未知模块给干净的 404。 */
    private CompiledGraph graphOr404(String module) {
        try {
            return runtime.graph(module);
        } catch (AgentRuntime.UnknownModuleError exc) {
            throw new HttpError(404, ErrorCode.NOT_FOUND, String.valueOf(exc.getMessage()));
        }
    }

    private boolean knownModule(String module) {
        return runtime.modules().containsKey(module) || AgentRuntime.SUPERVISOR_MODULE.equals(module);
    }

    /**
     * 降级守门（#82）：线程索引缺席且 env=production 时以 503 拒绝——
     * 降级路径没有属主信息，静默放宽安全边界会把「功能降级」变成
     * 「越权数据暴露」；development 显式豁免（本地开发）并 WARN。
     */
    private void requireIndexForDegradedPath(String operation) {
        if ("production".equals(runtime.settings().env)) {
            throw new HttpError(503, ErrorCode.STORAGE_UNAVAILABLE, "线程索引不可用，生产环境拒绝降级" + operation
                    + "：无索引即无属主校验（fail-safe 而非 fail-open）；"
                    + "请修复索引存储后重启，本地调试请用 ENV=development");
        }
        LOG.warn("线程索引缺席，{} 走无属主校验的降级路径（仅限本地开发，"
                + "production 会拒绝服务）", operation);
    }

    /** 线程属主校验（S1）：索引在位时，非属主一律 404 掩蔽。 */
    private void threadOwnerOr404(String module, String threadId, String userId) {
        var index = runtime.threadIndex();
        if (index == null) {
            requireIndexForDegradedPath("线程读取端点");
            return;
        }
        var entry = index.getThreadIndex(new ThreadId(module, threadId).toString());
        if (entry == null || !entry.userId().equals(userId)) {
            throw new HttpError(404, ErrorCode.NOT_FOUND, "线程不存在：" + threadId);
        }
    }

    // ── invoke 请求体（校验在 invoke 内手写 → 422，对齐 FastAPI 语义） ────

    public record InvokeRequest(String message, String threadId, List<String> attachments,
                                String mode, String profile,
                                @com.fasterxml.jackson.annotation.JsonProperty("continue_plan")
                                Boolean continuePlan) {
        /** 五参便利构造（continue_plan 缺省 false；测试与旧调用方用）。 */
        public InvokeRequest(String message, String threadId, List<String> attachments,
                             String mode, String profile) {
            this(message, threadId, attachments, mode, profile, null);
        }
    }

    @PostMapping("/{module}/invoke")
    public SseEmitter invoke(@PathVariable("module") String module,
                             @RequestBody InvokeRequest body,
                             jakarta.servlet.http.HttpServletResponse response,
                             @RequestHeader(value = "X-User-Id", required = false) String userIdHeader,
                             @RequestHeader(value = "X-User-Sig", required = false) String sigHeader,
                             @RequestHeader(value = "X-User-Timestamp", required = false) String tsHeader,
                             @RequestHeader(value = "X-User-Nonce", required = false) String nonceHeader) {
        // 请求体校验（422 优先，对齐 FastAPI 校验层）。长度按码点计数
        // （Pydantic max_length 语义），代理对内容与 Python 侧同界。
        if (body.message() == null || body.message().isEmpty()
                || CharMetrics.len(body.message()) > 100_000) {
            throw new HttpError(422, ErrorCode.VALIDATION, "message 必须是 1..100000 字符");
        }
        if (body.threadId() != null && (body.threadId().length() > 128
                // (?U)：Python re 的 \w 是 unicode 语义（中文 thread id 合法），
                // Java 默认 ASCII——不加标志会把 Python 接受的 id 422 掉。
                || !body.threadId().matches("(?U)^[\\w.-]+$"))) {
            throw new HttpError(422, ErrorCode.VALIDATION, "thread_id 格式非法（[\\w.-]{1,128}）");
        }
        if (body.attachments() != null && body.attachments().size() > 5) {
            throw new HttpError(422, ErrorCode.VALIDATION, "attachments 最多 5 个");
        }
        String mode = body.mode() == null ? "chat" : body.mode();
        String profile = body.profile() == null ? "main" : body.profile();
        if (!MODES.contains(mode)) {
            throw new HttpError(422, ErrorCode.VALIDATION, "mode 必须是 " + String.join("|", MODES));
        }
        if (!LlmFactory.PROFILES.contains(profile)) {
            throw new HttpError(422, ErrorCode.VALIDATION, "profile 必须是 " + String.join("|", LlmFactory.PROFILES));
        }
        // 预算熔断（T4.2）：在建流之前检查——429 优先于半截 SSE。
        if (governor != null) {
            try {
                governor.check();
            } catch (CostBudgetExceededException exc) {
                throw new HttpError(429, ErrorCode.BUDGET_BLOCKED, String.valueOf(exc.getMessage()));
            }
        }
        // 难度分流（M9）：mode 只决定调哪张图——thread 前缀、记忆 agent_id、
        // 属主、附件绑定、形成钩子全部维持请求模块不变（决策 5）。
        boolean usePlan = "plan".equals(mode)
                || ("auto".equals(mode) && (CharMetrics.len(body.message()) > 200
                        || (body.attachments() != null && !body.attachments().isEmpty())));
        if (usePlan && !runtime.modules().containsKey("planner")) {
            if ("plan".equals(mode)) {
                throw new HttpError(503, ErrorCode.MODULE_UNAVAILABLE, "planner 模块未启用（AGENT_MODULES 不含 planner）");
            }
            usePlan = false; // auto：静默降级
        }
        // 模型档位（T1.2）：快档视图共享 checkpointer/memory，未配置显式 400。
        AgentRuntime view = runtime.profileView(profile);
        if (view == null) {
            throw new HttpError(400, ErrorCode.BAD_REQUEST, "LLM_FAST_* 未配置：profile=\"fast\" 不可用（配置快速档或使用默认档）");
        }
        CompiledGraph graph = graphOr404(usePlan ? "planner" : module);
        String userThreadId = body.threadId() != null && !body.threadId().isEmpty()
                ? body.threadId()
                : io.github.paopaonb666.agentbase.extensions.Observability.newRequestId();
        String userId = user(userIdHeader, sigHeader, tsHeader, nonceHeader);
        // per-user 限流（#78）：认证之后、任何副作用之前——429 优先于半截
        // SSE（与预算熔断 429 同哲学）。无鉴权部署（#196：secret 空，全部
        // 请求落 default 用户）用共享降级桶——X-User-Id 此时客户端可控，
        // 按"用户"分桶等于给攻击者发无限个桶（review R1-F2）。
        String limitKey = AuthBackends.unauthenticatedDeployment(runtime.settings())
                ? "__unauthenticated__"
                : userId;
        if (!rateLimiter.tryAcquire(limitKey, System.nanoTime())) {
            response.setHeader("Retry-After", String.valueOf(60));
            throw new HttpError(429, ErrorCode.RATE_LIMITED, "请求过于频繁：每用户上限 " + rateLimiter.perMinute()
                    + " 次/分钟，请稍后重试（RATE_LIMIT_PER_MINUTE）");
        }
        String fullThreadId = new ThreadId(module, userThreadId).toString();
        // 属主预检（S1/P0）：已认证 ≠ 有权访问该线程。invoke 的全部副作用
        // （busy 探针、plan 快照读取、记忆注入、附件绑定、checkpoint 写入）
        // 都必须在通过之后。404 文案与 threadOwnerOr404 一致，不泄露存在性
        // （顺带堵住 409/400 对非属主线程的存在性枚举通道）。
        var threadIndex = runtime.threadIndex();
        if (threadIndex != null) {
            var existing = threadIndex.getThreadIndex(fullThreadId);
            if (existing != null && !existing.userId().equals(userId)) {
                throw new HttpError(404, ErrorCode.NOT_FOUND, "线程不存在：" + userThreadId);
            }
        }
        // 同 thread 并发 invoke → 409（#197）：权威判定在图执行的 tryLock，
        // 探针只把报错时机提前到响应提交前（SSE 开始后只能以 error 事件收口）。
        if (graph.isThreadBusy(fullThreadId)) {
            response.setHeader("Retry-After", "5");
            throw new HttpError(409, ErrorCode.THREAD_BUSY, "线程忙：该 thread 已有进行中的对话（" + userThreadId
                    + "），请等待当前轮完成或稍后重试");
        }
        RunConfig config = new RunConfig(fullThreadId, userId, runtime.settings().agentRecursionLimit);
        // 跨轮 plan 延续（#66）：continue_plan=true 载入快照，planner 从
        // cursor 直接续跑（plan 节点检测到已注入 tasks 即跳过重拆）。
        Map<String, Object> planSeed = planContinuationSeed(runtime, body, usePlan, fullThreadId);
        List<ChatMessage> messagesInput = new ArrayList<>();
        // 记忆上下文注入（M6d + T2.1）：冻结段在前，召回段紧随其后；任何
        // 一段失败都静默降级。
        var memory = runtime.memory();
        if (memory != null) {
            String contextBlock = memory.composeContext(userId, module, fullThreadId, body.message());
            String recallBlock = memory.recallContext(userId, body.message());
            // 条目 13（PromptComposer）：注入层只在日志侧标注层 id/版本——
            // composeContext/recallContext 返回值与下方 SystemMessage 消息
            // 构造逐字节不动（冻结段快照缓存语义不受影响）。graph 字段=发起
            // 模块名（usePlan 分流到 planner 图时注入内容仍是该模块的记忆）。
            if (contextBlock != null && !contextBlock.isEmpty()) {
                PromptComposer.logLayer(PromptComposer.Layer.memoryFrozen(contextBlock), module);
            }
            if (recallBlock != null && !recallBlock.isEmpty()) {
                PromptComposer.logLayer(PromptComposer.Layer.memoryRecall(recallBlock), module);
            }
            // Arrays.asList 允许 null 元素：记忆缺席（null）= 该段静默降级。
            for (String block : java.util.Arrays.asList(contextBlock, recallBlock)) {
                if (block != null && !block.isEmpty()) {
                    messagesInput.add(SystemMessage.from(block));
                }
            }
        }
        // 附件（M4b）：校验属主 → 绑定线程 → 注入。
        if (body.attachments() != null && !body.attachments().isEmpty()) {
            UploadedFileStore fileStore = runtime.fileStore();
            if (fileStore == null) {
                throw new HttpError(503, ErrorCode.STORAGE_UNAVAILABLE, "附件存储未启用（存储后端不可用）");
            }
            List<UploadedFileInfo> attachmentInfos = fileStore.getMany(body.attachments());
            List<String> missingOrUnauthorized = new ArrayList<>();
            for (String fileId : body.attachments()) {
                boolean exists = attachmentInfos.stream().anyMatch(i -> i.fileId().equals(fileId));
                boolean owned = attachmentInfos.stream()
                        .anyMatch(i -> i.fileId().equals(fileId)
                                && (i.userId().isEmpty() || i.userId().equals(userId)));
                if (!exists || !owned) {
                    missingOrUnauthorized.add(fileId);
                }
            }
            if (!missingOrUnauthorized.isEmpty()) {
                // 缺失与越权同文案同状态码——不给第三方探测 file_id 存在性的通道。
                List<String> sorted = new ArrayList<>(missingOrUnauthorized);
                java.util.Collections.sort(sorted);
                throw new HttpError(400, ErrorCode.BAD_REQUEST, "附件不存在或无权访问：" + String.join("，", sorted)
                        + "；请重新上传");
            }
            fileStore.bindThread(body.attachments(), fullThreadId);
            List<String> blocks = attachmentBlocks(attachmentInfos);
            List<UploadedFileInfo> imageParts = new ArrayList<>();
            for (UploadedFileInfo info : attachmentInfos) {
                if (MessageSerializers.IMAGE_STORED_FORMATS.contains(info.format())) {
                    imageParts.add(info);
                }
            }
            if (!blocks.isEmpty()) {
                // 条目 13（PromptComposer）：附件注入段只在日志侧标注
                // input.attachment 层——拼接出的文本与收口前逐字节相同，
                // SystemMessage 消息构造不动。
                String attachmentBlock = ContextEngine.ATTACHMENT_CONTEXT_PREFIX
                        + "，供回答时参考：\n\n" + String.join("\n\n", blocks);
                PromptComposer.logLayer(PromptComposer.Layer.inputAttachment(attachmentBlock), module);
                messagesInput.add(SystemMessage.from(attachmentBlock));
            }
            // 图片：多模态 content blocks（vision 模型直接看图）。
            List<Map<String, Object>> imageMeta = new ArrayList<>();
            for (UploadedFileInfo img : imageParts) {
                Map<String, Object> meta = new LinkedHashMap<>();
                meta.put("content", img.content());
                meta.put("format", img.format());
                imageMeta.add(meta);
            }
            UserMessage human = Attachments.humanMessage(body.message(), imageMeta);
            messagesInput.add(human);
            // 附件元数据落到图状态 extras（历史回放由 human 消息承载）。
            List<Map<String, Object>> attachmentsMeta = new ArrayList<>();
            for (UploadedFileInfo info : attachmentInfos) {
                attachmentsMeta.add(Attachments.metaOf(info.fileId(), info.filename(),
                        info.format(), info.pages(), info.truncated(), info.textLen(), info.warning()));
            }
            // 元数据随本轮输入合并进状态（execute 后由 GraphState extras 承载）。
            Map<String, Object> inputExtra = new LinkedHashMap<>();
            inputExtra.put("attachments", Map.of(String.valueOf(messagesInput.size() - 1), attachmentsMeta));
            inputExtra.putAll(planSeed);
            return startStream(graph, messagesInput, inputExtra, config, userThreadId, userId, module, response);
        }
        messagesInput.add(UserMessage.from(body.message()));
        // 会话索引 upsert（S1）由 startStream 统一执行（#141：非附件分支
        // 不再先行写一次——双写既放大写入量，两次写之间失败还会留下
        // 不一致标题；Python 侧恰一次）。
        return startStream(graph, messagesInput, planSeed, config, userThreadId, userId, module, response);
    }

    /**
     * 跨轮 plan 延续的种子（#66）：{@code continue_plan=true} 时从
     * plan_snapshots 载入 tasks/cursor/replans 作为图状态注入。校验：
     * 非 planner 执行（mode=chat 或 auto 未分流）→ 400；无快照 → 400
     * （显式要求延续而快照缺席是用户可理解的错误，静默重拆反而误导）。
     */
    private Map<String, Object> planContinuationSeed(AgentRuntime runtime, InvokeRequest body,
                                                     boolean usePlan, String fullThreadId) {
        if (body.continuePlan() == null || !body.continuePlan()) {
            return Map.of();
        }
        if (!usePlan) {
            throw new HttpError(400, ErrorCode.BAD_REQUEST,
                    "continue_plan 需要与 planner 一起使用（mode=plan，或 mode=auto 且本轮分流到 planner）");
        }
        var planStore = runtime.planStore();
        Map<String, Object> snapshot = planStore == null ? null : planStore.getPlan(fullThreadId);
        if (snapshot == null || !(snapshot.get("tasks") instanceof List<?> tasks) || tasks.isEmpty()) {
            throw new HttpError(400, ErrorCode.BAD_REQUEST,
                    "没有可延续的计划快照（该线程首轮或快照已清理）；请正常发起 plan");
        }
        Map<String, Object> seed = new LinkedHashMap<>();
        seed.put(io.github.paopaonb666.agentbase.modules.planner.graph.PlannerState.TASKS, tasks);
        seed.put(io.github.paopaonb666.agentbase.modules.planner.graph.PlannerState.CURSOR,
                snapshot.get("cursor"));
        seed.put(io.github.paopaonb666.agentbase.modules.planner.graph.PlannerState.REPLANS,
                snapshot.get("replans"));
        return seed;
    }

    /** 打开 SSE 流（挂记忆形成钩子；索引 upsert 的唯一落点，所有分支共用）。 */
    private SseEmitter startStream(CompiledGraph graph, List<ChatMessage> messagesInput,
                                   Map<String, Object> extra, RunConfig config,
                                   String userThreadId, String userId, String module,
                                   jakarta.servlet.http.HttpServletResponse response) {
        // SSE 代理头（对应 Python sse.py _SSE_HEADERS 挂在 StreamingResponse
        // 上）：no-cache 防中间层缓存事件流；X-Accel-Buffering 关掉 nginx 的
        // proxy_buffering——缺了它 delta 会被缓冲到流结束才一次性下发。
        response.setHeader("Cache-Control", "no-cache");
        response.setHeader("X-Accel-Buffering", "no");
        // 全局 run 并发闸门（#79，review R1-F6）：判定提前到线程索引 claim
        // 之前——429 拒绝的请求不再认领/刷新线程索引（幽灵活跃痕迹消失）。
        // 限流令牌豁免不做（裁决）：#78 限流仍在流式入口先行，被闸门 429
        // 的请求令牌已消耗，计数按「请求发出」口径语义保持。饱和快速失败
        // （429+Retry-After），不排队。
        if (!runGate.tryEnter()) {
            response.setHeader("Retry-After", "5");
            throw new HttpError(429, ErrorCode.RUN_GATE_SATURATED, "服务并发已达上限（AGENT_RUN_CONCURRENCY="
                    + runGate.maxConcurrent() + "），请稍后重试");
        }
        // tryEnter 成功后到开流之间仍有唯一常规抛出路径（下方 claim 属主
        // 不符 404）：opened 标志保配对——开流成功前抛出在 finally 还回许可；
        // 开流成功后退出权移交 SSE 终态钩子（produce 的 finally，正常/失败/
        // 取消全路径恰好一次）。
        boolean opened = false;
        try {
            // 会话索引原子认领（S1）：标题取首条 human 消息前 60 字符。缺席则
            // 按本用户插入，在席且属主一致则仅刷新 updated_at，属主不符则不
            // 改动并以 404 拒绝——invoke 开头的预检与这里的认领之间属主被
            // 并发改变（删除后被他人首发）由权威行兜底，属主永不翻转。
            if (runtime.threadIndex() != null) {
                String title = messagesInput.stream()
                        .filter(m -> m instanceof UserMessage)
                        .map(m -> MessageSerializers.extractText(m))
                        .filter(t -> !t.isEmpty())
                        .findFirst()
                        .map(t -> t.substring(0, Math.min(60, t.length())))
                        .orElse(userThreadId);
                var claimed = runtime.threadIndex().claimThreadIndex(
                        io.github.paopaonb666.agentbase.core.storage.ThreadIndexRecord.create(
                                config.threadId(), userId, module, title));
                if (!claimed.userId().equals(userId)) {
                    throw new HttpError(404, ErrorCode.NOT_FOUND, "线程不存在：" + userThreadId);
                }
            }
            // 记忆形成（M6c）：轮次正常完成后在后台跑抽取/整合/画像/摘要。
            Runnable onComplete = buildCaptureHook(module, userThreadId, graph, config);
            // 附件元数据并入状态 extras（对应 Python 把 meta 挂在
            // HumanMessage.additional_kwargs 随 checkpoint 持久化；langchain4j
            // 消息对象无元数据承载，改由 extras 承载）。键用“绝对消息下标”
            // （历史消息数 + 本轮输入下标）——按本轮下标键控会在第二轮起错位；
            // attachments 键整表替换（extras 通道语义），所以先并入历史表。
            Map<String, Object> attachmentsByIndex = new LinkedHashMap<>();
            GraphState prior = graph.getState(config);
            int base = prior == null ? 0 : prior.messages().size();
            if (prior != null && prior.extra().get("attachments") instanceof Map<?, ?> priorAtt) {
                for (Map.Entry<?, ?> e : priorAtt.entrySet()) {
                    attachmentsByIndex.put(String.valueOf(e.getKey()), e.getValue());
                }
            }
            if (extra.get("attachments") instanceof Map<?, ?> newAtt) {
                for (Map.Entry<?, ?> e : newAtt.entrySet()) {
                    int inputIdx;
                    try {
                        inputIdx = Integer.parseInt(String.valueOf(e.getKey()));
                    } catch (NumberFormatException nfe) {
                        continue;
                    }
                    attachmentsByIndex.put(String.valueOf(base + inputIdx), e.getValue());
                }
            }
            Map<String, Object> effectiveExtra = new LinkedHashMap<>();
            if (!attachmentsByIndex.isEmpty()) {
                effectiveExtra.put("attachments", attachmentsByIndex);
            }
            // 非附件的种子键（#66 的 planner 延续）原样透传进图状态
            for (Map.Entry<String, Object> e : extra.entrySet()) {
                if (!"attachments".equals(e.getKey())) {
                    effectiveExtra.putIfAbsent(e.getKey(), e.getValue());
                }
            }
            SseEmitter emitter = SseStreams.open(graph, messagesInput, effectiveExtra, config, userThreadId,
                    onComplete, runGate::exit);
            opened = true;
            return emitter;
        } finally {
            if (!opened) {
                runGate.exit();
            }
        }
    }

    /** 构造轮次完成后的记忆形成回调（M6c）；管线不可用时返回 null。 */
    private Runnable buildCaptureHook(String module, String userThreadId, CompiledGraph graph,
                                      RunConfig config) {
        var memory = runtime.memory();
        if (memory == null || !memory.captureReady()) {
            return null;
        }
        return wrapContext(() -> {
            try {
                GraphState snapshot = graph.getState(config);
                List<ChatMessage> messages = snapshot == null ? List.of() : snapshot.messages();
                if (messages.isEmpty()) {
                    return;
                }
                String userId = config.userId() == null || config.userId().isEmpty()
                        ? "default" : config.userId();
                memory.captureTurn(userId, module,
                        new ThreadId(module, userThreadId).toString(), messages);
            } catch (Exception e) {
                org.slf4j.LoggerFactory.getLogger(AgentsRoutes.class)
                        .warn("memory: 后台记忆形成失败（不影响对话）", e);
            }
        });
    }

    private static Runnable wrapContext(Runnable body) {
        return io.github.paopaonb666.agentbase.extensions.Observability.wrap(body);
    }

    // ── GET /{module}/threads：列出当前用户的线程 ────────────────────────

    @GetMapping("/{module}/threads")
    public Map<String, Object> listThreads(@PathVariable("module") String module,
                                           @RequestParam(value = "limit", defaultValue = "50") int limit,
                                           @RequestParam(value = "offset", defaultValue = "0") int offset,
                                           @RequestHeader(value = "X-User-Id", required = false) String userIdHeader,
                                           @RequestHeader(value = "X-User-Sig", required = false) String sigHeader,
                                           @RequestHeader(value = "X-User-Timestamp", required = false) String tsHeader,
                                            @RequestHeader(value = "X-User-Nonce", required = false) String nonceHeader) {
        // 鉴权先行（对应 Python get_current_user 依赖）。
        String userId = user(userIdHeader, sigHeader, tsHeader, nonceHeader);
        if (!knownModule(module)) {
            throw new HttpError(404, ErrorCode.NOT_FOUND, "unknown module '" + module + "'");
        }
        // 分页（#204，Java 侧 wire 扩展）：limit/offset 可选，缺省 = Python
        // 同构（50 条、零偏移）。limit 钳 1..200，offset 非负，取数上限
        // 1000 行（防止 offset 放大扫描——要更深的页面前端应该收窄模块）。
        int pageLimit = Math.max(1, Math.min(limit, THREAD_LIST_PAGE_MAX));
        int pageOffset = Math.max(0, offset);
        // +1 探针行：满页时还能区分"恰好满页"与"后面还有"（无 COUNT 查询）。
        int fetch = Math.min(pageOffset + pageLimit + 1, THREAD_LIST_FETCH_MAX);
        var index = runtime.threadIndex();
        if (index != null) {
            List<ThreadIndexRecord> entries = index.listThreadIndexes(userId, module, fetch);
            boolean hasMore = entries.size() > pageOffset + pageLimit;
            List<Map<String, Object>> threads = new ArrayList<>();
            for (int i = pageOffset; i < entries.size() && threads.size() < pageLimit; i++) {
                var entry = entries.get(i);
                // 索引行总是命名空间形态；异常行整串回退。
                ThreadId parsed = ThreadId.tryParse(entry.threadId());
                String shortId = parsed != null ? parsed.raw() : entry.threadId();
                Map<String, Object> view = new LinkedHashMap<>();
                view.put("thread_id", shortId);
                view.put("module", entry.module().isEmpty() ? module : entry.module());
                view.put("title", entry.title().isEmpty() ? entry.threadId() : entry.title());
                view.put("updated_at", (long) (entry.updatedAt() * 1000));
                threads.add(view);
                if (threads.size() >= pageLimit) {
                    break;
                }
            }
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("threads", threads);
            if (hasMore) {
                out.put("next_offset", pageOffset + pageLimit);
            }
            return out;
        }
        // 降级：checkpointer 扫描（#82：production 拒绝，development 豁免）。
        requireIndexForDegradedPath("线程列表");
        if (runtime.checkpointer() == null) {
            return Map.of("threads", List.of());
        }
        String prefix = module + ":";
        Map<String, Map<String, Object>> threads = new LinkedHashMap<>();
        int scanned = 0;
        int total = 0;
        for (String fullId : runtime.checkpointer().listThreadIds()) {
            total++;
            if (total > THREAD_LIST_TOTAL_LIMIT) {
                break;
            }
            if (!fullId.startsWith(prefix)) {
                continue;
            }
            scanned++;
            if (scanned > THREAD_LIST_SCAN_LIMIT) {
                break;
            }
            String shortId = fullId.substring(prefix.length());
            if (shortId.isEmpty() || threads.containsKey(shortId)) {
                continue;
            }
            GraphState state = runtime.checkpointer().get(fullId);
            String title = shortId;
            if (state != null) {
                for (ChatMessage m : state.messages()) {
                    if (m instanceof UserMessage) {
                        String text = MessageSerializers.extractText(m);
                        title = text.substring(0, Math.min(60, text.length()));
                        break;
                    }
                }
            }
            Map<String, Object> view = new LinkedHashMap<>();
            view.put("thread_id", shortId);
            view.put("module", module);
            view.put("title", title);
            // 时间语义恢复（#142）：checkpointer 记录最后落盘时刻——
            // 解析不了才退 0（对齐 Python checkpoint.ts 的降级口径）。
            view.put("updated_at", runtime.checkpointer().updatedAt(fullId));
            threads.put(shortId, view);
            if (threads.size() >= fetch) {
                break;
            }
        }
        List<Map<String, Object>> ordered = new ArrayList<>(threads.values());
        ordered.sort((a, b) -> Long.compare((long) b.get("updated_at"), (long) a.get("updated_at")));
        List<Map<String, Object>> page = ordered.subList(Math.min(pageOffset, ordered.size()),
                Math.min(pageOffset + pageLimit, ordered.size()));
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("threads", page);
        if (ordered.size() > pageOffset + pageLimit) {
            out.put("next_offset", pageOffset + pageLimit);
        }
        return out;
    }

    // ── GET /{module}/threads/{thread_id}：持久化消息历史 ────────────────
    // 可选 ?include_system=true|1（#154，Java 单边增强，Python 端点未跟进
    // ——差异登记）：system 消息（附件注入块、记忆上下文）默认不上屏
    // （现行为零变化），prompt 组装审计时显式传 true 才以
    // {"role":"system","content":…} 上屏（审计视图按码点截断）。

    @GetMapping("/{module}/threads/{thread_id}")
    public Map<String, Object> threadHistory(@PathVariable("module") String module,
                                             @PathVariable("thread_id") String threadId,
                                             @RequestHeader(value = "X-User-Id", required = false) String userIdHeader,
                                             @RequestHeader(value = "X-User-Sig", required = false) String sigHeader,
                                             @RequestHeader(value = "X-User-Timestamp", required = false) String tsHeader,
                                             @RequestHeader(value = "X-User-Nonce", required = false) String nonceHeader,
                                             @RequestParam(value = "include_system", required = false) String includeSystem) {
        // 缺省/非 true 值一律视为 false（system 不上屏）；"1" 是惯用简写。
        // 与 deleteThread 的 purge_memories 解析同款。
        boolean withSystem = "true".equalsIgnoreCase(includeSystem) || "1".equals(includeSystem);
        return threadHistory(module, threadId, userIdHeader, sigHeader, tsHeader, nonceHeader, withSystem);
    }

    /** 六参形态（include_system 缺省 false）：既有调用方/测试的兼容入口。 */
    public Map<String, Object> threadHistory(String module, String threadId,
                                             String userIdHeader, String sigHeader,
                                             String tsHeader, String nonceHeader) {
        return threadHistory(module, threadId, userIdHeader, sigHeader, tsHeader, nonceHeader, false);
    }

    public Map<String, Object> threadHistory(String module, String threadId,
                                             String userIdHeader, String sigHeader,
                                             String tsHeader, String nonceHeader,
                                             boolean includeSystem) {
        // 鉴权先行（对应 Python get_current_user 依赖先于一切处理逻辑）。
        String userId = user(userIdHeader, sigHeader, tsHeader, nonceHeader);
        validateThreadId(threadId);
        CompiledGraph graph = graphOr404(module);
        threadOwnerOr404(module, threadId, userId);
        GraphState snapshot = graph.getState(new RunConfig(new ThreadId(module, threadId).toString(),
                userId, runtime.settings().agentRecursionLimit));
        List<Map<String, Object>> messages = new ArrayList<>();
        if (snapshot != null) {
            Map<String, Object> extra = snapshot.extra();
            Map<?, ?> attachmentsByIndex = extra.get("attachments") instanceof Map<?, ?> m ? m : Map.of();
            int idx = 0;
            for (ChatMessage message : snapshot.messages()) {
                Object meta = attachmentsByIndex.get(String.valueOf(idx));
                var serialized = MessageSerializers.serializeMessage(message, meta, includeSystem);
                if (serialized != null) {
                    messages.add(serialized);
                }
                idx++;
            }
        }
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("thread_id", threadId);
        out.put("module", module);
        out.put("messages", messages);
        return out;
    }

    // ── GET /{module}/threads/{thread_id}/tool-calls：审计记录 ───────────

    @GetMapping("/{module}/threads/{thread_id}/tool-calls")
    public Map<String, Object> toolCalls(@PathVariable("module") String module,
                                         @PathVariable("thread_id") String threadId,
                                         @RequestParam(value = "limit", defaultValue = "100") int limit,
                                         @RequestHeader(value = "X-User-Id", required = false) String userIdHeader,
                                         @RequestHeader(value = "X-User-Sig", required = false) String sigHeader,
                                         @RequestHeader(value = "X-User-Timestamp", required = false) String tsHeader,
                                         @RequestHeader(value = "X-User-Nonce", required = false) String nonceHeader) {
        // 鉴权先行（对应 Python get_current_user 依赖）。
        String userId = user(userIdHeader, sigHeader, tsHeader, nonceHeader);
        validateThreadId(threadId);
        if (!knownModule(module)) {
            throw new HttpError(404, ErrorCode.NOT_FOUND, "unknown module '" + module + "'");
        }
        threadOwnerOr404(module, threadId, userId);
        var recorder = runtime.toolRecorder();
        if (recorder == null) {
            return Map.of("tool_calls", List.of());
        }
        return Map.of("tool_calls", recorder.listForThread(
                new ThreadId(module, threadId).toString(), Math.max(1, Math.min(limit, 500))));
    }

    // ── GET /{module}/threads/{thread_id}/plan：只读计划状态（M10 读表） ──

    @GetMapping("/{module}/threads/{thread_id}/plan")
    public Map<String, Object> threadPlan(@PathVariable("module") String module,
                                          @PathVariable("thread_id") String threadId,
                                          @RequestHeader(value = "X-User-Id", required = false) String userIdHeader,
                                          @RequestHeader(value = "X-User-Sig", required = false) String sigHeader,
                                          @RequestHeader(value = "X-User-Timestamp", required = false) String tsHeader,
                                          @RequestHeader(value = "X-User-Nonce", required = false) String nonceHeader) {
        // 鉴权先行（对应 Python get_current_user 依赖）。
        String userId = user(userIdHeader, sigHeader, tsHeader, nonceHeader);
        validateThreadId(threadId);
        if (!knownModule(module)) {
            throw new HttpError(404, ErrorCode.NOT_FOUND, "unknown module '" + module + "'");
        }
        if (!runtime.modules().containsKey("planner")) {
            throw new HttpError(503, ErrorCode.MODULE_UNAVAILABLE, "planner 模块未启用，无计划状态可读");
        }
        threadOwnerOr404(module, threadId, userId);
        String fullThreadId = new ThreadId(module, threadId).toString();
        if (runtime.planStore() != null) {
            Map<String, Object> stored = runtime.planStore().getPlan(fullThreadId);
            if (stored != null) {
                Map<String, Object> out = new LinkedHashMap<>();
                out.put("thread_id", threadId);
                out.put("module", stored.getOrDefault("module", module));
                out.put("tasks", stored.getOrDefault("tasks", List.of()));
                out.put("cursor", numberOr(stored.get("cursor"), 0));
                out.put("replans", numberOr(stored.get("replans"), 0));
                return out;
            }
        }
        // 表缺席时回退读 planner 图状态；从未跑过 plan 返回空默认值。
        GraphState snapshot = runtime.graph("planner")
                .getState(new RunConfig(fullThreadId, userId, runtime.settings().agentRecursionLimit));
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("thread_id", threadId);
        out.put("module", module);
        out.put("tasks", snapshot != null ? snapshot.getOrDefault("tasks", List.of()) : List.of());
        out.put("cursor", snapshot != null ? numberOr(snapshot.get("cursor"), 0) : 0);
        out.put("replans", snapshot != null ? numberOr(snapshot.get("replans"), 0) : 0);
        return out;
    }

    // ── DELETE /{module}/threads/{thread_id}：级联删除 ───────────────────
    // 可选 ?purge_memories=true|1（#172，Java 单边增强，Python 端点未跟进
    // ——行为分歧登记 #315）：删除线程默认保留跨会话记忆（删线程 ≠ 删记忆，
    // memories 出处字段 source_thread_id 原样保留）；显式传 true 才连长期
    // 记忆一起清（GDPR 式彻底删除）。

    @DeleteMapping("/{module}/threads/{thread_id}")
    public Map<String, Object> deleteThread(@PathVariable("module") String module,
                                            @PathVariable("thread_id") String threadId,
                                            @RequestHeader(value = "X-User-Id", required = false) String userIdHeader,
                                            @RequestHeader(value = "X-User-Sig", required = false) String sigHeader,
                                            @RequestHeader(value = "X-User-Timestamp", required = false) String tsHeader,
                                            @RequestHeader(value = "X-User-Nonce", required = false) String nonceHeader,
                                            @RequestParam(value = "purge_memories", required = false) String purgeMemories) {
        // 缺省/非 true 值一律视为 false（保留记忆）；"1" 是惯用简写。
        boolean purge = "true".equalsIgnoreCase(purgeMemories) || "1".equals(purgeMemories);
        return deleteThread(module, threadId, userIdHeader, sigHeader, tsHeader, nonceHeader, purge);
    }

    /** 六参形态（purge_memories 缺省 false）：既有调用方/测试的兼容入口。 */
    public Map<String, Object> deleteThread(String module, String threadId,
                                            String userIdHeader, String sigHeader,
                                            String tsHeader, String nonceHeader) {
        return deleteThread(module, threadId, userIdHeader, sigHeader, tsHeader, nonceHeader, false);
    }

    public Map<String, Object> deleteThread(String module, String threadId,
                                            String userIdHeader, String sigHeader,
                                            String tsHeader, String nonceHeader,
                                            boolean purgeMemories) {
        // 鉴权先行（对应 Python get_current_user 依赖）。
        String userId = user(userIdHeader, sigHeader, tsHeader, nonceHeader);
        validateThreadId(threadId);
        if (!knownModule(module)) {
            throw new HttpError(404, ErrorCode.NOT_FOUND, "unknown module '" + module + "'");
        }
        threadOwnerOr404(module, threadId, userId);
        if (runtime.checkpointer() == null) {
            return Map.of("deleted", false, "reason", "checkpointer unconfigured");
        }
        String fullThreadId = new ThreadId(module, threadId).toString();
        runtime.checkpointer().deleteThread(fullThreadId);
        if (runtime.fileStore() != null) {
            runtime.fileStore().deleteForThread(fullThreadId);
        }
        if (runtime.threadIndex() != null) {
            // 索引行随线程消亡；计划快照同属线程作用域数据面（M10）。
            runtime.threadIndex().deleteThreadIndex(fullThreadId);
            runtime.threadIndex().deletePlan(fullThreadId);
        }
        var memory = runtime.memory();
        if (memory != null) {
            // 记忆级联（M6）：清摘要与知识库分块；跨会话记忆刻意保留
            // （#172 隐私语义：删线程默认保留长期记忆）——purgeMemories=true
            // 才按 source_thread_id 连 memories 一起删。
            memory.purgeThread(fullThreadId, purgeMemories);
        }
        return Map.of("deleted", true);
    }

    private static int numberOr(Object value, int fallback) {
        return value instanceof Number n ? n.intValue() : fallback;
    }

    private static void validateThreadId(String threadId) {
        // (?U)：Python re 的 \w 是 unicode 语义（中文 thread id 合法），
        // Java 默认 ASCII——不加标志会把 invoke 创建的中文 id 在读侧 422 掉
        // （创建后即失联）。与 invoke 请求体侧的校验（本文件上方）同标志。
        if (threadId.length() > 128 || !threadId.matches("(?U)^[\\w.-]+$")) {
            throw new HttpError(422, ErrorCode.VALIDATION, "thread_id 格式非法（[\\w.-]{1,128}）");
        }
    }

    /**
     * 附件注入块（文本类；图片走多模态分支不在此列）。
     *
     * <p>#250：截断发生在「附件全文注入」时用户与模型都不知情（模型拿到
     * 半份文档）——truncated=true 时在块尾追加指路提示（完整内容走
     * knowledge_search），而不是让模型把半份文档当成全文回答。
     */
    static List<String> attachmentBlocks(List<UploadedFileInfo> infos) {
        List<String> blocks = new ArrayList<>();
        for (UploadedFileInfo info : infos) {
            if (MessageSerializers.IMAGE_STORED_FORMATS.contains(info.format())) {
                continue;
            }
            if (!info.extractedText().strip().isEmpty()) {
                String pages = info.pages() != null ? info.pages() + " 页，" : "";
                String truncatedNote = info.truncated()
                        ? "\n\n（注：以上内容因长度已截断，完整内容请用 knowledge_search 检索）"
                        : "";
                blocks.add("[附件文件：" + info.filename() + "（" + info.format() + "，" + pages
                        + info.textLen() + " 字符）]\n" + info.extractedText() + truncatedNote);
            } else {
                String note = info.warning() == null || info.warning().isEmpty()
                        ? "无法提取文本" : info.warning();
                blocks.add("[附件文件：" + info.filename() + "]（" + note + "）");
            }
        }
        return blocks;
    }
}

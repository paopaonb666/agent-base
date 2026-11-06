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
import io.github.paopaonb666.agentbase.entrypoints.server.Attachments;
import io.github.paopaonb666.agentbase.entrypoints.server.AuthBackends;
import io.github.paopaonb666.agentbase.entrypoints.server.Background;
import io.github.paopaonb666.agentbase.entrypoints.server.HttpError;
import io.github.paopaonb666.agentbase.entrypoints.server.MessageSerializers;
import io.github.paopaonb666.agentbase.entrypoints.server.SseStreams;
import io.github.paopaonb666.agentbase.extensions.costmeter.CostBudgetExceededException;
import io.github.paopaonb666.agentbase.extensions.costmeter.CostGovernor;
import io.github.paopaonb666.agentbase.extensions.filestore.UploadedFileInfo;
import io.github.paopaonb666.agentbase.extensions.filestore.UploadedFileStore;

/**
 * 对话与会话线程端点：invoke（SSE）、线程列表/历史/删除、工具审计、
 * 计划状态。S1 线程作用域：每个线程在 thread_index 表里有属主记录
 * （invoke 时 upsert）；列表只返回当前用户的线程；历史/删除/审计端点
 * 校验属主（404 掩蔽）。索引缺席（后端不可用）时降级为旧行为——全库
 * 扫描 + 无属主校验，仅限本地开发。
 *
 * <p>对应 Python 侧 {@code routes/agents.py}，端点/状态码/文案逐条对齐。
 */
@RestController
@RequestMapping("/v1/agents")
public class AgentsRoutes {

    // 线程列表端点的扫描/数量上限（旧版回退路径仍用）。
    private static final int THREAD_LIST_SCAN_LIMIT = 2000;
    private static final int THREAD_LIST_TOTAL_LIMIT = 20_000;
    private static final int THREAD_LIST_LIMIT = 50;

    private final AgentRuntime runtime;
    private final AuthBackends.AuthBackend auth;
    private final CostGovernor governor;

    public AgentsRoutes(AgentRuntime runtime, AuthBackends.AuthBackend auth,
                        org.springframework.beans.factory.ObjectProvider<CostGovernor> governorProvider) {
        this.runtime = runtime;
        this.auth = auth;
        // COST_ENABLED=false 时 governor bean 缺席（null）——可选注入。
        this.governor = governorProvider.getIfAvailable();
    }

    private String user(String userIdHeader, String sigHeader, String tsHeader, String nonceHeader) {
        return auth.authenticate(userIdHeader, sigHeader, tsHeader, nonceHeader, runtime.settings());
    }

    /** 按模块取图；未知模块给干净的 404。 */
    private CompiledGraph graphOr404(String module) {
        try {
            return runtime.graph(module);
        } catch (AgentRuntime.UnknownModuleError exc) {
            throw new HttpError(404, String.valueOf(exc.getMessage()));
        }
    }

    private boolean knownModule(String module) {
        return runtime.modules().containsKey(module) || AgentRuntime.SUPERVISOR_MODULE.equals(module);
    }

    /** 线程属主校验（S1）：索引在位时，非属主一律 404 掩蔽。 */
    private void threadOwnerOr404(String module, String threadId, String userId) {
        var index = runtime.threadIndex();
        if (index == null) {
            return;
        }
        var entry = index.getThreadIndex(new ThreadId(module, threadId).toString());
        if (entry == null || !entry.userId().equals(userId)) {
            throw new HttpError(404, "线程不存在：" + threadId);
        }
    }

    // ── invoke 请求体（校验在 invoke 内手写 → 422，对齐 FastAPI 语义） ────

    public record InvokeRequest(String message, String threadId, List<String> attachments,
                                String mode, String profile) {}

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
            throw new HttpError(422, "message 必须是 1..100000 字符");
        }
        if (body.threadId() != null && (body.threadId().length() > 128
                // (?U)：Python re 的 \w 是 unicode 语义（中文 thread id 合法），
                // Java 默认 ASCII——不加标志会把 Python 接受的 id 422 掉。
                || !body.threadId().matches("(?U)^[\\w.-]+$"))) {
            throw new HttpError(422, "thread_id 格式非法（[\\w.-]{1,128}）");
        }
        if (body.attachments() != null && body.attachments().size() > 5) {
            throw new HttpError(422, "attachments 最多 5 个");
        }
        String mode = body.mode() == null ? "chat" : body.mode();
        String profile = body.profile() == null ? "main" : body.profile();
        if (!List.of("chat", "plan", "auto").contains(mode)) {
            throw new HttpError(422, "mode 必须是 chat|plan|auto");
        }
        if (!List.of("main", "fast").contains(profile)) {
            throw new HttpError(422, "profile 必须是 main|fast");
        }
        // 预算熔断（T4.2）：在建流之前检查——429 优先于半截 SSE。
        if (governor != null) {
            try {
                governor.check();
            } catch (CostBudgetExceededException exc) {
                throw new HttpError(429, String.valueOf(exc.getMessage()));
            }
        }
        // 难度分流（M9）：mode 只决定调哪张图——thread 前缀、记忆 agent_id、
        // 属主、附件绑定、形成钩子全部维持请求模块不变（决策 5）。
        boolean usePlan = "plan".equals(mode)
                || ("auto".equals(mode) && (CharMetrics.len(body.message()) > 200
                        || (body.attachments() != null && !body.attachments().isEmpty())));
        if (usePlan && !runtime.modules().containsKey("planner")) {
            if ("plan".equals(mode)) {
                throw new HttpError(503, "planner 模块未启用（AGENT_MODULES 不含 planner）");
            }
            usePlan = false; // auto：静默降级
        }
        // 模型档位（T1.2）：快档视图共享 checkpointer/memory，未配置显式 400。
        AgentRuntime view = runtime.profileView(profile);
        if (view == null) {
            throw new HttpError(400, "LLM_FAST_* 未配置：profile=\"fast\" 不可用（配置快速档或使用默认档）");
        }
        CompiledGraph graph = graphOr404(usePlan ? "planner" : module);
        String userThreadId = body.threadId() != null && !body.threadId().isEmpty()
                ? body.threadId()
                : io.github.paopaonb666.agentbase.extensions.Observability.newRequestId();
        String userId = user(userIdHeader, sigHeader, tsHeader, nonceHeader);
        String fullThreadId = new ThreadId(module, userThreadId).toString();
        RunConfig config = new RunConfig(fullThreadId, userId, runtime.settings().agentRecursionLimit);
        List<ChatMessage> messagesInput = new ArrayList<>();
        // 记忆上下文注入（M6d + T2.1）：冻结段在前，召回段紧随其后；任何
        // 一段失败都静默降级。
        var memory = runtime.memory();
        if (memory != null) {
            String contextBlock = memory.composeContext(userId, module, fullThreadId, body.message());
            String recallBlock = memory.recallContext(userId, body.message());
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
                throw new HttpError(503, "附件存储未启用（存储后端不可用）");
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
                throw new HttpError(400, "附件不存在或无权访问：" + String.join("，", sorted)
                        + "；请重新上传");
            }
            fileStore.bindThread(body.attachments(), fullThreadId);
            List<String> blocks = new ArrayList<>();
            List<UploadedFileInfo> imageParts = new ArrayList<>();
            for (UploadedFileInfo info : attachmentInfos) {
                if (MessageSerializers.IMAGE_STORED_FORMATS.contains(info.format())) {
                    imageParts.add(info);
                    continue;
                }
                if (!info.extractedText().strip().isEmpty()) {
                    String pages = info.pages() != null ? info.pages() + " 页，" : "";
                    blocks.add("[附件文件：" + info.filename() + "（" + info.format() + "，" + pages
                            + info.textLen() + " 字符）]\n" + info.extractedText());
                } else {
                    String note = info.warning() == null || info.warning().isEmpty()
                            ? "无法提取文本" : info.warning();
                    blocks.add("[附件文件：" + info.filename() + "]（" + note + "）");
                }
            }
            if (!blocks.isEmpty()) {
                messagesInput.add(SystemMessage.from(ContextEngine.ATTACHMENT_CONTEXT_PREFIX
                        + "，供回答时参考：\n\n" + String.join("\n\n", blocks)));
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
            return startStream(graph, messagesInput, inputExtra, config, userThreadId, userId, module, response);
        }
        messagesInput.add(UserMessage.from(body.message()));
        // 会话索引 upsert（S1）：属主/模块/标题（首条用户消息前 60 字符）。
        upsertThreadIndex(module, fullThreadId, userId, body.message());
        return startStream(graph, messagesInput, Map.of(), config, userThreadId, userId, module, response);
    }

    /** 打开 SSE 流（挂记忆形成钩子；附件分支的索引 upsert 在此共用）。 */
    private SseEmitter startStream(CompiledGraph graph, List<ChatMessage> messagesInput,
                                   Map<String, Object> extra, RunConfig config,
                                   String userThreadId, String userId, String module,
                                   jakarta.servlet.http.HttpServletResponse response) {
        // SSE 代理头（对应 Python sse.py _SSE_HEADERS 挂在 StreamingResponse
        // 上）：no-cache 防中间层缓存事件流；X-Accel-Buffering 关掉 nginx 的
        // proxy_buffering——缺了它 delta 会被缓冲到流结束才一次性下发。
        response.setHeader("Cache-Control", "no-cache");
        response.setHeader("X-Accel-Buffering", "no");
        // 会话索引 upsert（S1）：标题取首条 human 消息前 60 字符。
        if (runtime.threadIndex() != null) {
            String title = messagesInput.stream()
                    .filter(m -> m instanceof UserMessage)
                    .map(m -> MessageSerializers.extractText(m))
                    .filter(t -> !t.isEmpty())
                    .findFirst()
                    .map(t -> t.substring(0, Math.min(60, t.length())))
                    .orElse(userThreadId);
            var existing = runtime.threadIndex().getThreadIndex(config.threadId());
            runtime.threadIndex().upsertThreadIndex(existing == null
                    ? io.github.paopaonb666.agentbase.core.storage.ThreadIndexRecord.create(
                            config.threadId(), userId, module, title)
                    : new io.github.paopaonb666.agentbase.core.storage.ThreadIndexRecord(
                            existing.threadId(), existing.userId(), existing.module(),
                            existing.title(), existing.createdAt(),
                            System.currentTimeMillis() / 1000.0));
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
        return SseStreams.open(graph, messagesInput, effectiveExtra, config, userThreadId, onComplete);
    }

    /** 会话索引 upsert（S1）：属主/模块/标题（首条用户消息前 60 字符）。 */
    private void upsertThreadIndex(String module, String fullThreadId, String userId, String message) {
        if (runtime.threadIndex() == null) {
            return;
        }
        var existing = runtime.threadIndex().getThreadIndex(fullThreadId);
        if (existing == null) {
            runtime.threadIndex().upsertThreadIndex(
                    io.github.paopaonb666.agentbase.core.storage.ThreadIndexRecord.create(
                            fullThreadId, userId, module,
                            message.substring(0, Math.min(60, message.length()))));
        } else {
            runtime.threadIndex().upsertThreadIndex(
                    new io.github.paopaonb666.agentbase.core.storage.ThreadIndexRecord(
                            existing.threadId(), existing.userId(), existing.module(),
                            existing.title(), existing.createdAt(),
                            System.currentTimeMillis() / 1000.0));
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
                                           @RequestHeader(value = "X-User-Id", required = false) String userIdHeader,
                                           @RequestHeader(value = "X-User-Sig", required = false) String sigHeader,
                                           @RequestHeader(value = "X-User-Timestamp", required = false) String tsHeader,
                                            @RequestHeader(value = "X-User-Nonce", required = false) String nonceHeader) {
        // 鉴权先行（对应 Python get_current_user 依赖）。
        String userId = user(userIdHeader, sigHeader, tsHeader, nonceHeader);
        if (!knownModule(module)) {
            throw new HttpError(404, "unknown module '" + module + "'");
        }
        var index = runtime.threadIndex();
        if (index != null) {
            List<Map<String, Object>> threads = new ArrayList<>();
            for (var entry : index.listThreadIndexes(userId, module, THREAD_LIST_LIMIT)) {
                // 索引行总是命名空间形态；异常行整串回退。
                ThreadId parsed = ThreadId.tryParse(entry.threadId());
                String shortId = parsed != null ? parsed.raw() : entry.threadId();
                Map<String, Object> view = new LinkedHashMap<>();
                view.put("thread_id", shortId);
                view.put("module", entry.module().isEmpty() ? module : entry.module());
                view.put("title", entry.title().isEmpty() ? entry.threadId() : entry.title());
                view.put("updated_at", (long) (entry.updatedAt() * 1000));
                threads.add(view);
            }
            return Map.of("threads", threads);
        }
        // 降级：checkpointer 扫描（仅本地开发语义）。
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
            view.put("updated_at", 0L);
            threads.put(shortId, view);
            if (threads.size() >= THREAD_LIST_LIMIT) {
                break;
            }
        }
        List<Map<String, Object>> ordered = new ArrayList<>(threads.values());
        ordered.sort((a, b) -> Long.compare((long) b.get("updated_at"), (long) a.get("updated_at")));
        return Map.of("threads", ordered);
    }

    // ── GET /{module}/threads/{thread_id}：持久化消息历史 ────────────────

    @GetMapping("/{module}/threads/{thread_id}")
    public Map<String, Object> threadHistory(@PathVariable("module") String module,
                                             @PathVariable("thread_id") String threadId,
                                             @RequestHeader(value = "X-User-Id", required = false) String userIdHeader,
                                             @RequestHeader(value = "X-User-Sig", required = false) String sigHeader,
                                             @RequestHeader(value = "X-User-Timestamp", required = false) String tsHeader,
                                             @RequestHeader(value = "X-User-Nonce", required = false) String nonceHeader) {
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
                var serialized = MessageSerializers.serializeMessage(message, meta);
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
            throw new HttpError(404, "unknown module '" + module + "'");
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
            throw new HttpError(404, "unknown module '" + module + "'");
        }
        if (!runtime.modules().containsKey("planner")) {
            throw new HttpError(503, "planner 模块未启用，无计划状态可读");
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

    @DeleteMapping("/{module}/threads/{thread_id}")
    public Map<String, Object> deleteThread(@PathVariable("module") String module,
                                            @PathVariable("thread_id") String threadId,
                                            @RequestHeader(value = "X-User-Id", required = false) String userIdHeader,
                                            @RequestHeader(value = "X-User-Sig", required = false) String sigHeader,
                                            @RequestHeader(value = "X-User-Timestamp", required = false) String tsHeader,
                                            @RequestHeader(value = "X-User-Nonce", required = false) String nonceHeader) {
        // 鉴权先行（对应 Python get_current_user 依赖）。
        String userId = user(userIdHeader, sigHeader, tsHeader, nonceHeader);
        validateThreadId(threadId);
        if (!knownModule(module)) {
            throw new HttpError(404, "unknown module '" + module + "'");
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
            // 记忆级联（M6）：清摘要与知识库分块；跨会话记忆刻意保留。
            memory.purgeThread(fullThreadId);
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
            throw new HttpError(422, "thread_id 格式非法（[\\w.-]{1,128}）");
        }
    }
}

package io.github.paopaonb666.agentbase.memory.api;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import io.github.paopaonb666.agentbase.core.AgentModule;
import io.github.paopaonb666.agentbase.core.AgentRuntime;
import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.entrypoints.server.AuthBackends;
import io.github.paopaonb666.agentbase.entrypoints.server.HttpError;
import io.github.paopaonb666.agentbase.memory.service.MemoryService;
import io.github.paopaonb666.agentbase.memory.store.MemoryStore;
import io.github.paopaonb666.agentbase.memory.store.Models.MemoryBlock;
import io.github.paopaonb666.agentbase.memory.store.Models.MemoryRecord;
import io.github.paopaonb666.agentbase.memory.tools.BlockLimits;

import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PatchMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

/**
 * /v1/memory 管理端点（对应 Python 侧 {@code agent_base_memory/api.py}，
 * 11 个端点逐条对齐）。全部经统一的身份解析（AuthBackend）+ 属主比对
 * （404 掩蔽而非 403）；记忆服务未启用时 503。
 */
@RestController
@RequestMapping("/v1/memory")
public class MemoryApiController {

    private final AgentRuntime runtime;
    private final AuthBackends.AuthBackend auth;

    public MemoryApiController(AgentRuntime runtime, AuthBackends.AuthBackend auth) {
        this.runtime = runtime;
        this.auth = auth;
    }

    private MemoryService service() {
        AgentModule.MemoryServicePort port = runtime.memory();
        if (port == null) {
            throw new HttpError(503, "记忆系统未启用（MEMORY_ENABLED=false）");
        }
        return (MemoryService) port;
    }

    private Settings settings() {
        return runtime.settings();
    }

    /** 身份解析唯一路径（对应 deps.get_current_user）。 */
    private String user(String userIdHeader, String sigHeader, String tsHeader, String nonceHeader) {
        return auth.authenticate(userIdHeader, sigHeader, tsHeader, nonceHeader, settings());
    }

    /** 属主校验（404 掩蔽，不泄露存在性）。 */
    private MemoryRecord owned(MemoryService memory, String memoryId, String userId) {
        MemoryRecord record = memory.getMemory(memoryId);
        if (record == null || !record.userId().equals(userId)) {
            throw new HttpError(404, "记忆不存在：" + memoryId);
        }
        return record;
    }

    // ── POST /v1/memory：手工写入（落库前尽力向量化） ───────────────────

    public record MemoryCreateRequest(String content, String kind, List<String> tags,
                                      String agentId, Double salience) {}

    @PostMapping
    public Map<String, Object> create(@RequestBody MemoryCreateRequest body,
                                      @RequestHeader(value = "X-User-Id", required = false) String userIdHeader,
                                      @RequestHeader(value = "X-User-Sig", required = false) String sigHeader,
                                      @RequestHeader(value = "X-User-Timestamp", required = false) String tsHeader,
                                      @RequestHeader(value = "X-User-Nonce", required = false) String nonceHeader) {
        MemoryService memory = service();
        String userId = user(userIdHeader, sigHeader, tsHeader, nonceHeader);
        if (body.content() == null || body.content().isEmpty() || body.content().length() > 20_000) {
            throw new HttpError(422, "content 必须是 1..20000 字符");
        }
        String kind = body.kind() == null ? "semantic" : body.kind();
        // 对应 Python api.py create_memory：kind 白名单校验（400）。
        if (!io.github.paopaonb666.agentbase.memory.store.Models.KNOWN_MEMORY_KINDS.contains(kind)) {
            throw new HttpError(400, "kind 非法：'" + kind
                    + "'；允许 ['semantic', 'episodic', 'procedural']");
        }
        List<String> tags = body.tags() == null ? List.of() : body.tags();
        if (tags.size() > 10) {
            throw new HttpError(422, "tags 最多 10 个");
        }
        if (body.agentId() != null && body.agentId().length() > 64) {
            throw new HttpError(422, "agent_id 最多 64 字符");
        }
        String agentId = body.agentId() == null || body.agentId().isEmpty() ? "*" : body.agentId();
        double salience = body.salience() == null ? 0.5 : body.salience();
        // Pydantic ge=0.0 le=1.0：越界 422，而不是静默钳制。
        if (salience < 0.0 || salience > 1.0) {
            throw new HttpError(422, "salience 必须在 0.0..1.0 之间");
        }
        MemoryRecord record = memory.addMemory(userId, agentId, body.content(), kind, tags, salience, "");
        return record.meta();
    }

    // ── GET /v1/memory：q 有→混合检索；无→按状态浏览 ────────────────────

    @GetMapping
    public Map<String, Object> list(
            @RequestParam(value = "q", required = false) String q,
            @RequestParam(value = "module", required = false) String module,
            @RequestParam(value = "kind", required = false) String kind,
            @RequestParam(value = "status", required = false) String status,
            @RequestParam(value = "limit", defaultValue = "20") int limit,
            @RequestHeader(value = "X-User-Id", required = false) String userIdHeader,
            @RequestHeader(value = "X-User-Sig", required = false) String sigHeader,
            @RequestHeader(value = "X-User-Timestamp", required = false) String tsHeader,
            @RequestHeader(value = "X-User-Nonce", required = false) String nonceHeader) {
        MemoryService memory = service();
        String userId = user(userIdHeader, sigHeader, tsHeader, nonceHeader);
        // 对应 Python search_memory：kind/status 白名单校验（400）——null 视为
        // 缺省，其余任何值（含空串）都必须在已知集合内。
        if (kind != null && !io.github.paopaonb666.agentbase.memory.store.Models.KNOWN_MEMORY_KINDS.contains(kind)) {
            throw new HttpError(400, "kind 非法：'" + kind
                    + "'；允许 ['semantic', 'episodic', 'procedural']");
        }
        if (status != null && !io.github.paopaonb666.agentbase.memory.store.Models.KNOWN_MEMORY_STATUSES.contains(status)) {
            throw new HttpError(400, "status 非法：'" + status
                    + "'；允许 ['active', 'superseded', 'archived']");
        }
        // 对应 Python api.py search_memory：limit 两个分支共用同一钳制
        // （1..100），q 分支同样吃 module（agent 作用域收窄）与 kind 过滤。
        int clampedLimit = Math.max(1, Math.min(limit, 100));
        Map<String, Object> out = new LinkedHashMap<>();
        if (q != null && !q.isEmpty()) {
            out.put("query", q);
            List<Map<String, Object>> memories = new ArrayList<>();
            for (var item : memory.searchScored(userId, module, q, clampedLimit)) {
                if (kind != null && !kind.isEmpty() && !kind.equals(item.record().kind())) {
                    continue;
                }
                Map<String, Object> view = new LinkedHashMap<>(item.record().meta());
                view.put("score", Math.round(item.score() * 10000.0) / 10000.0);
                view.put("expired", memory.isExpired(item.record()));
                memories.add(view);
            }
            out.put("memories", memories);
            return out;
        }
        List<String> statuses = status == null || status.isEmpty() ? null : List.of(status);
        List<String> kinds = kind == null || kind.isEmpty() ? null : List.of(kind);
        List<Map<String, Object>> memories = new ArrayList<>();
        for (MemoryRecord record : memory.listMemories(userId, module, true, kinds, statuses,
                clampedLimit, true)) {
            Map<String, Object> view = new LinkedHashMap<>(record.meta());
            view.put("expired", memory.isExpired(record));
            memories.add(view);
        }
        out.put("memories", memories);
        return out;
    }

    // ── PATCH /v1/memory/{id}：部分更新（content 变化重向量化） ──────────

    public record MemoryPatchRequest(String content, List<String> tags, Double salience,
                                     String status) {}

    @PatchMapping("/{memory_id}")
    public Map<String, Object> patch(@PathVariable("memory_id") String memoryId,
                                     @RequestBody MemoryPatchRequest body,
                                     @RequestHeader(value = "X-User-Id", required = false) String userIdHeader,
                                     @RequestHeader(value = "X-User-Sig", required = false) String sigHeader,
                                     @RequestHeader(value = "X-User-Timestamp", required = false) String tsHeader,
                                     @RequestHeader(value = "X-User-Nonce", required = false) String nonceHeader) {
        // 字段约束对应 Python MemoryPatchRequest（Pydantic）：
        // content 1..20000、tags ≤ 10、salience 0..1——越界 422 而不是静默
        // 接受（content:"" 曾会覆盖成空内容）。
        if (body.content() != null && (body.content().isEmpty() || body.content().length() > 20_000)) {
            throw new HttpError(422, "content 必须是 1..20000 字符");
        }
        if (body.tags() != null && body.tags().size() > 10) {
            throw new HttpError(422, "tags 最多 10 个");
        }
        if (body.salience() != null && (body.salience() < 0.0 || body.salience() > 1.0)) {
            throw new HttpError(422, "salience 必须在 0.0..1.0 之间");
        }
        MemoryService memory = service();
        String userId = user(userIdHeader, sigHeader, tsHeader, nonceHeader);
        owned(memory, memoryId, userId);
        MemoryRecord updated = memory.updateMemory(memoryId, body.content(), body.tags(),
                body.salience(), body.status(), "update");
        if (updated == null) {
            throw new HttpError(404, "记忆不存在：" + memoryId);
        }
        return updated.meta();
    }

    // ── DELETE /v1/memory/{id}：硬删（版本墓碑与审计保留） ───────────────

    @DeleteMapping("/{memory_id}")
    public Map<String, Object> delete(@PathVariable("memory_id") String memoryId,
                                      @RequestHeader(value = "X-User-Id", required = false) String userIdHeader,
                                      @RequestHeader(value = "X-User-Sig", required = false) String sigHeader,
                                      @RequestHeader(value = "X-User-Timestamp", required = false) String tsHeader,
                                      @RequestHeader(value = "X-User-Nonce", required = false) String nonceHeader) {
        MemoryService memory = service();
        String userId = user(userIdHeader, sigHeader, tsHeader, nonceHeader);
        owned(memory, memoryId, userId);
        return Map.of("deleted", memory.deleteMemory(memoryId));
    }

    // ── GET /v1/memory/{id}/versions：内容版本史 ────────────────────────

    @GetMapping("/{memory_id}/versions")
    public Map<String, Object> versions(@PathVariable("memory_id") String memoryId,
                                        @RequestParam(value = "limit", defaultValue = "50") int limit,
                                        @RequestHeader(value = "X-User-Id", required = false) String userIdHeader,
                                        @RequestHeader(value = "X-User-Sig", required = false) String sigHeader,
                                        @RequestHeader(value = "X-User-Timestamp", required = false) String tsHeader,
                                        @RequestHeader(value = "X-User-Nonce", required = false) String nonceHeader) {
        MemoryService memory = service();
        String userId = user(userIdHeader, sigHeader, tsHeader, nonceHeader);
        owned(memory, memoryId, userId);
        List<Map<String, Object>> versions = new ArrayList<>();
        for (var version : memory.listVersions(memoryId, Math.min(Math.max(limit, 1), 200))) {
            versions.add(version.toDict());
        }
        return Map.of("versions", versions);
    }

    // ── POST /v1/memory/{id}/versions/{vid}/restore：恢复到指定版本 ──────

    @PostMapping("/{memory_id}/versions/{version_id}/restore")
    public Map<String, Object> restore(@PathVariable("memory_id") String memoryId,
                                       @PathVariable("version_id") String versionId,
                                       @RequestHeader(value = "X-User-Id", required = false) String userIdHeader,
                                       @RequestHeader(value = "X-User-Sig", required = false) String sigHeader,
                                       @RequestHeader(value = "X-User-Timestamp", required = false) String tsHeader,
                                       @RequestHeader(value = "X-User-Nonce", required = false) String nonceHeader) {
        MemoryService memory = service();
        String userId = user(userIdHeader, sigHeader, tsHeader, nonceHeader);
        owned(memory, memoryId, userId);
        MemoryRecord restored = memory.restoreVersion(memoryId, versionId);
        if (restored == null) {
            // 400 而非 404：对应 Python api.py restore_memory_version。
            throw new HttpError(400, "版本不可恢复：version_id 无效或对应删除墓碑");
        }
        return restored.meta();
    }

    // ── GET /v1/memory/blocks?module：列出常驻记忆块 ────────────────────

    @GetMapping("/blocks")
    public Map<String, Object> blocks(@RequestParam(value = "module", required = false) String module,
                                      @RequestHeader(value = "X-User-Id", required = false) String userIdHeader,
                                      @RequestHeader(value = "X-User-Sig", required = false) String sigHeader,
                                      @RequestHeader(value = "X-User-Timestamp", required = false) String tsHeader,
                                      @RequestHeader(value = "X-User-Nonce", required = false) String nonceHeader) {
        MemoryService memory = service();
        String userId = user(userIdHeader, sigHeader, tsHeader, nonceHeader);
        String agentId = module == null || module.isEmpty() ? "*" : module;
        List<Map<String, Object>> blocks = memory.listBlocks(userId, agentId).stream()
                .map(MemoryBlock::toDict).toList();
        return Map.of("blocks", blocks);
    }

    // ── PUT /v1/memory/blocks/{label}：手工写块（覆盖式，版本递增） ──────

    public record MemoryBlockPutRequest(String content, String module, Integer charLimit) {}

    @PutMapping("/blocks/{label}")
    public Map<String, Object> putBlock(@PathVariable("label") String label,
                                        @RequestBody MemoryBlockPutRequest body,
                                        @RequestHeader(value = "X-User-Id", required = false) String userIdHeader,
                                        @RequestHeader(value = "X-User-Sig", required = false) String sigHeader,
                                        @RequestHeader(value = "X-User-Timestamp", required = false) String tsHeader,
                                        @RequestHeader(value = "X-User-Nonce", required = false) String nonceHeader) {
        MemoryService memory = service();
        String userId = user(userIdHeader, sigHeader, tsHeader, nonceHeader);
        // #178：三路径上限收口于 BlockLimits（工具侧 4000 更保守，见该类 javadoc 矩阵）。
        if (body.content() == null || body.content().isEmpty()
                || body.content().length() > BlockLimits.CONTENT_MAX_API) {
            throw new HttpError(422, "content 必须是 1.." + BlockLimits.CONTENT_MAX_API + " 字符");
        }
        int charLimit = body.charLimit() == null ? BlockLimits.CHAR_LIMIT_DEFAULT : body.charLimit();
        if (charLimit < BlockLimits.CHAR_LIMIT_MIN || charLimit > BlockLimits.CHAR_LIMIT_MAX) {
            throw new HttpError(422, "char_limit 必须在 " + BlockLimits.CHAR_LIMIT_MIN + ".."
                    + BlockLimits.CHAR_LIMIT_MAX + " 之间");
        }
        String agentId = body.module() == null || body.module().isEmpty() ? "*" : body.module();
        return memory.putBlock(userId, agentId, label, body.content(), charLimit).toDict();
    }

    // ── DELETE /v1/memory/blocks/{label}?module ─────────────────────────

    @DeleteMapping("/blocks/{label}")
    public Map<String, Object> deleteBlock(@PathVariable("label") String label,
                                           @RequestParam(value = "module", required = false) String module,
                                           @RequestHeader(value = "X-User-Id", required = false) String userIdHeader,
                                           @RequestHeader(value = "X-User-Sig", required = false) String sigHeader,
                                           @RequestHeader(value = "X-User-Timestamp", required = false) String tsHeader,
                                           @RequestHeader(value = "X-User-Nonce", required = false) String nonceHeader) {
        MemoryService memory = service();
        String userId = user(userIdHeader, sigHeader, tsHeader, nonceHeader);
        String agentId = module == null || module.isEmpty() ? "*" : module;
        return Map.of("deleted", memory.deleteBlock(userId, agentId, label));
    }

    // ── GET /v1/memory/profile：结构化画像 ──────────────────────────────

    @GetMapping("/profile")
    public Map<String, Object> profile(
            @RequestHeader(value = "X-User-Id", required = false) String userIdHeader,
            @RequestHeader(value = "X-User-Sig", required = false) String sigHeader,
            @RequestHeader(value = "X-User-Timestamp", required = false) String tsHeader,
            @RequestHeader(value = "X-User-Nonce", required = false) String nonceHeader) {
        MemoryService memory = service();
        String userId = user(userIdHeader, sigHeader, tsHeader, nonceHeader);
        return Map.of("profile", memory.getProfile(userId));
    }

    // ── GET /v1/memory/audit：记忆操作审计 ──────────────────────────────

    /**
     * #160/#176：可选 {@code thread}（thread_id 精确匹配）、{@code since}
     * （epoch 秒，含边界）与 {@code before}（翻页游标，epoch 秒——取上一页
     * 响应的 next_before，语义是严格早于该时刻）。取满 limit 时响应追加
     * {@code next_before}（末行的 created_at 秒值）；不足 limit 不带。
     */
    @GetMapping("/audit")
    public Map<String, Object> audit(@RequestParam(value = "limit", defaultValue = "50") int limit,
                                     @RequestParam(value = "thread", required = false) String thread,
                                     @RequestParam(value = "since", required = false) Double since,
                                     @RequestParam(value = "before", required = false) Double before,
                                     @RequestHeader(value = "X-User-Id", required = false) String userIdHeader,
                                     @RequestHeader(value = "X-User-Sig", required = false) String sigHeader,
                                     @RequestHeader(value = "X-User-Timestamp", required = false) String tsHeader,
                                     @RequestHeader(value = "X-User-Nonce", required = false) String nonceHeader) {
        MemoryService memory = service();
        String userId = user(userIdHeader, sigHeader, tsHeader, nonceHeader);
        int effectiveLimit = Math.min(Math.max(limit, 1), 200);
        // 游标只有秒值（API 契约），同刻并列的余量由 store 层的时间戳截断
        // 兜底（见 MemoryStore.OpsCursor）。
        MemoryStore.OpsCursor cursor = before == null ? null
                : new MemoryStore.OpsCursor(before, null);
        List<io.github.paopaonb666.agentbase.memory.store.Models.MemoryOp> ops =
                memory.listOps(userId, thread, since, cursor, effectiveLimit);
        List<Map<String, Object>> items = ops.stream()
                .map(io.github.paopaonb666.agentbase.memory.store.Models.MemoryOp::toDict)
                .toList();
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("ops", items);
        if (ops.size() >= effectiveLimit) {
            out.put("next_before", ops.get(ops.size() - 1).createdAt());
        }
        return out;
    }

    // ── 向量回填（#164：换 embedding 模型后的迁移入口） ─────────────────

    /**
     * 启动异步回填（进程内单飞；重复启动 409）。维护级操作：鉴权与其他
     * 记忆端点同一层（HMAC 头），无额外 per-user 作用域——回填对象是
     * 全库向量，不是发起用户的数据。
     */
    @PostMapping("/backfill")
    public Map<String, Object> startBackfill(@RequestParam(value = "batch", required = false) Integer batch,
                                             @RequestHeader(value = "X-User-Id", required = false) String userIdHeader,
                                             @RequestHeader(value = "X-User-Sig", required = false) String sigHeader,
                                             @RequestHeader(value = "X-User-Timestamp", required = false) String tsHeader,
                                             @RequestHeader(value = "X-User-Nonce", required = false) String nonceHeader) {
        MemoryService memory = service();
        user(userIdHeader, sigHeader, tsHeader, nonceHeader);
        boolean started = memory.startBackfill(batch == null ? 64 : batch);
        if (!started) {
            throw new HttpError(409, "回填任务已在运行中（见 GET /v1/memory/backfill/status）");
        }
        return Map.of("started", true, "status", memory.backfillStatus());
    }

    /** 回填进度查询。 */
    @GetMapping("/backfill/status")
    public Map<String, Object> backfillStatus(
            @RequestHeader(value = "X-User-Id", required = false) String userIdHeader,
            @RequestHeader(value = "X-User-Sig", required = false) String sigHeader,
            @RequestHeader(value = "X-User-Timestamp", required = false) String tsHeader,
            @RequestHeader(value = "X-User-Nonce", required = false) String nonceHeader) {
        MemoryService memory = service();
        user(userIdHeader, sigHeader, tsHeader, nonceHeader);
        return memory.backfillStatus();
    }
}

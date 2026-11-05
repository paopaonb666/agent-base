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
import io.github.paopaonb666.agentbase.memory.store.Models.MemoryBlock;
import io.github.paopaonb666.agentbase.memory.store.Models.MemoryRecord;

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
        List<String> tags = body.tags() == null ? List.of() : body.tags();
        if (tags.size() > 10) {
            throw new HttpError(422, "tags 最多 10 个");
        }
        String agentId = body.agentId() == null || body.agentId().isEmpty() ? "*" : body.agentId();
        double salience = body.salience() == null ? 0.5 : body.salience();
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
            @RequestParam(value = "limit", defaultValue = "50") int limit,
            @RequestHeader(value = "X-User-Id", required = false) String userIdHeader,
            @RequestHeader(value = "X-User-Sig", required = false) String sigHeader,
            @RequestHeader(value = "X-User-Timestamp", required = false) String tsHeader,
            @RequestHeader(value = "X-User-Nonce", required = false) String nonceHeader) {
        MemoryService memory = service();
        String userId = user(userIdHeader, sigHeader, tsHeader, nonceHeader);
        Map<String, Object> out = new LinkedHashMap<>();
        if (q != null && !q.isEmpty()) {
            out.put("query", q);
            List<Map<String, Object>> memories = new ArrayList<>();
            for (var item : memory.searchScored(userId, null, q, limit)) {
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
                Math.min(Math.max(limit, 1), 500), true)) {
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
            throw new HttpError(404, "版本不存在或不可恢复：" + versionId);
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
        if (body.content() == null || body.content().isEmpty() || body.content().length() > 8000) {
            throw new HttpError(422, "content 必须是 1..8000 字符");
        }
        int charLimit = body.charLimit() == null ? 2000 : body.charLimit();
        if (charLimit < 50 || charLimit > 8000) {
            throw new HttpError(422, "char_limit 必须在 50..8000 之间");
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

    @GetMapping("/audit")
    public Map<String, Object> audit(@RequestParam(value = "limit", defaultValue = "100") int limit,
                                     @RequestHeader(value = "X-User-Id", required = false) String userIdHeader,
                                     @RequestHeader(value = "X-User-Sig", required = false) String sigHeader,
                                     @RequestHeader(value = "X-User-Timestamp", required = false) String tsHeader,
                                     @RequestHeader(value = "X-User-Nonce", required = false) String nonceHeader) {
        MemoryService memory = service();
        String userId = user(userIdHeader, sigHeader, tsHeader, nonceHeader);
        List<Map<String, Object>> ops = memory.listOps(userId, Math.min(Math.max(limit, 1), 200))
                .stream().map(io.github.paopaonb666.agentbase.memory.store.Models.MemoryOp::toDict)
                .toList();
        return Map.of("ops", ops);
    }
}

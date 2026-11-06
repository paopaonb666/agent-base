package io.github.paopaonb666.agentbase.memory.store;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/**
 * 记忆存储的数据模型：记录/块/摘要/分块/审计/版本 + 向量编解码。
 *
 * <p>时间戳在数据类里统一为 epoch 秒（double），落库形式由后端决定
 * （sqlite 存字符串，MySQL 存 DATETIME）。向量以 float32 小端 BLOB 编码
 * （紧凑、跨后端一致的二进制形态），{@link #encodeEmbedding} /
 * {@link #decodeEmbedding} 负责编解码。
 *
 * <p>对应 Python 侧 {@code agent_base_memory/store/models.py}。
 */
public final class Models {

    private Models() {}

    /** 记忆的合法类别（LangMem 分类法）。 */
    public static final List<String> KNOWN_MEMORY_KINDS = List.of("semantic", "episodic", "procedural");
    /** 整合管线的生命周期状态。 */
    public static final List<String> KNOWN_MEMORY_STATUSES = List.of("active", "superseded", "archived");

    /**
     * 用户画像的确定性 memory_id 前缀：画像复用 memories 表（agent_id="*"、
     * tags=["profile"]），以 id 前缀与普通记忆区分——检索时被排除（画像由
     * 上下文组装单独注入），管理端点里可见可删。
     */
    public static final String PROFILE_ID_PREFIX = "profile:";

    /** 新生成的记忆 id（uuid hex 截断，与 file_id 同风格）。 */
    public static String newMemoryId() {
        return UUID.randomUUID().toString().replace("-", "").substring(0, 32);
    }

    /** 某用户画像记录的确定性 id。 */
    public static String profileMemoryId(String userId) {
        return PROFILE_ID_PREFIX + userId;
    }

    /** 存储层的快速失败：非法类别/状态等数据错误在写入边界被拒绝。 */
    public static final class MemoryStoreError extends IllegalArgumentException {
        public MemoryStoreError(String message) { super(message); }
    }

    /** 把向量编码为 float32 小端 BLOB。 */
    public static byte[] encodeEmbedding(List<Float> vector) {
        ByteBuffer buf = ByteBuffer.allocate(vector.size() * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (float v : vector) {
            buf.putFloat(v);
        }
        return buf.array();
    }

    /** 把 float32 小端 BLOB 解码回向量；空输入返回空表。 */
    public static List<Float> decodeEmbedding(byte[] blob) {
        if (blob == null || blob.length == 0) {
            return List.of();
        }
        ByteBuffer buf = ByteBuffer.wrap(blob).order(ByteOrder.LITTLE_ENDIAN);
        Float[] values = new Float[blob.length / 4];
        for (int i = 0; i < values.length; i++) {
            values[i] = buf.getFloat();
        }
        return List.of(values);
    }

    /** 一条跨会话长期记忆。 */
    public record MemoryRecord(
            String memoryId,
            String userId,
            String agentId, // 模块名；"*" 表示全模块共享
            String kind, // semantic | episodic | procedural
            String content,
            List<String> tags,
            byte[] embedding, // float32 小端 BLOB；null = 未向量化
            Integer embeddingDim,
            double salience,
            String status, // active | superseded | archived
            String sourceThreadId,
            List<String> sourceRefs,
            double createdAt,
            double updatedAt,
            Double lastAccessedAt,
            int accessCount) {

        public MemoryRecord {
            tags = tags == null ? List.of() : List.copyOf(tags);
            sourceRefs = sourceRefs == null ? List.of() : List.copyOf(sourceRefs);
            embedding = embedding == null ? new byte[0] : embedding;
        }

        /** 全默认便捷构造（对应 Python dataclass 默认值）。 */
        public MemoryRecord(String memoryId, String userId, String agentId, String kind, String content) {
            this(memoryId, userId, agentId, kind, content, List.of(), null, null,
                    0.5, "active", "", List.of(), now(), now(), null, 0);
        }

        /** 对外元信息（不含 embedding 字节与审计字段）。 */
        public Map<String, Object> meta() {
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("memory_id", memoryId);
            out.put("user_id", userId);
            out.put("agent_id", agentId);
            out.put("kind", kind);
            out.put("content", content);
            out.put("tags", tags);
            out.put("salience", salience);
            out.put("status", status);
            out.put("source_thread_id", sourceThreadId);
            out.put("has_embedding", embedding.length > 0);
            out.put("access_count", accessCount);
            out.put("created_at", createdAt);
            out.put("updated_at", updatedAt);
            return out;
        }

        public MemoryRecord withEmbedding(byte[] vector, int dim) {
            return new MemoryRecord(memoryId, userId, agentId, kind, content, tags, vector, dim,
                    salience, status, sourceThreadId, sourceRefs, createdAt, updatedAt,
                    lastAccessedAt, accessCount);
        }

        public MemoryRecord withContent(String newContent, double updatedAt) {
            return new MemoryRecord(memoryId, userId, agentId, kind, newContent, tags, null, null,
                    salience, status, sourceThreadId, sourceRefs, createdAt, updatedAt,
                    lastAccessedAt, accessCount);
        }
    }

    /** 一个常驻上下文的记忆块（Letta 式，agent 可用工具自我编辑）。 */
    public record MemoryBlock(
            String userId,
            String agentId,
            String label, // persona | human | 自定义标签
            String content,
            int charLimit,
            int version,
            double updatedAt) {

        public MemoryBlock(String userId, String agentId, String label, String content) {
            this(userId, agentId, label, content, 2000, 0, now());
        }

        public Map<String, Object> toDict() {
            // 键集与 Python api.py 的 blocks 载荷逐项一致（不含 user_id）。
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("agent_id", agentId);
            out.put("label", label);
            out.put("content", content);
            out.put("char_limit", charLimit);
            out.put("version", version);
            out.put("updated_at", updatedAt);
            return out;
        }
    }

    /** 一个线程的滚动摘要（短期压缩的载体，覆盖已修剪的旧历史）。 */
    public record SessionSummary(
            String userId,
            String threadId, // 带模块前缀的完整 thread id（模块名已含于其中）
            String summary,
            String agentId,
            int coveredMessageCount,
            double updatedAt) {

        public SessionSummary(String userId, String threadId, String summary) {
            this(userId, threadId, summary, "", 0, now());
        }
    }

    /**
     * 文档知识库的一个分块（上传文件的文本切片 + 可选 embedding）。
     * {@code offsets} 是 {@code ((start,end),…)}——各段在原文
     * （extracted_text）中的字符区间，切片可视化高亮边界用；段落打包块
     * 是多区间，固定窗口切片是单区间。旧数据为 null（前端降级为纯卡片）。
     */
    public record DocChunk(
            String chunkId,
            String fileId,
            String threadId,
            String userId,
            String agentId,
            int ordinal,
            String text,
            List<long[]> offsets, // 每个 long[]{start,end}；null = 旧数据
            byte[] embedding,
            Integer embeddingDim,
            double createdAt) {

        public DocChunk {
            offsets = offsets == null ? List.of() : List.copyOf(offsets);
            embedding = embedding == null ? new byte[0] : embedding;
        }

        public DocChunk(String chunkId, String fileId, String threadId, String userId,
                        String agentId, int ordinal, String text) {
            this(chunkId, fileId, threadId, userId, agentId, ordinal, text, null, null, null, now());
        }
    }

    /** 一次记忆系统操作的审计记录（成功与失败都记录）。 */
    public record MemoryOp(
            String opId,
            String op, // extract | consolidate | summary | profile | ingest | manual | ...
            String userId,
            String agentId,
            String threadId,
            Map<String, Object> detail,
            String status, // ok | error
            String errorText,
            int durationMs,
            double createdAt) {

        public MemoryOp {
            detail = detail == null ? new LinkedHashMap<>() : new LinkedHashMap<>(detail);
        }

        public MemoryOp(String opId, String op) {
            this(opId, op, "", "", "", new LinkedHashMap<>(), "ok", "", 0, now());
        }

        public Map<String, Object> toDict() {
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("op_id", opId);
            out.put("op", op);
            out.put("user_id", userId);
            out.put("agent_id", agentId);
            out.put("thread_id", threadId);
            out.put("detail", detail);
            out.put("status", status);
            out.put("error_text", errorText.isEmpty() ? null : errorText);
            out.put("duration_ms", durationMs);
            out.put("created_at", createdAt);
            return out;
        }
    }

    /**
     * 一条记忆的内容版本快照（版本史）。op 语义：create（首版）/
     * update（内容或属性变更，content 为新值、previous_content 为旧值）/
     * delete（墓碑：content 为空串，previous_content 保留最后内容）/
     * restore（从历史版本恢复）。
     */
    public record MemoryVersion(
            String versionId,
            String memoryId,
            String userId,
            String op, // create | update | delete | restore
            String content,
            String previousContent,
            String status,
            double createdAt) {

        public MemoryVersion(String versionId, String memoryId, String userId, String op,
                             String content, String previousContent) {
            this(versionId, memoryId, userId, op, content, previousContent, "active", now());
        }

        /** 带真实 status 的版本快照（对应 Python 各记录点的 status=… 语义）。 */
        public MemoryVersion(String versionId, String memoryId, String userId, String op,
                             String content, String previousContent, String status) {
            this(versionId, memoryId, userId, op, content, previousContent, status, now());
        }

        public Map<String, Object> toDict() {
            // 键集与 Python api.py 的 versions 载荷逐项一致（不含
            // memory_id/user_id；previous_content 空串归 null）。
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("version_id", versionId);
            out.put("op", op);
            out.put("content", content);
            out.put("previous_content", previousContent == null || previousContent.isEmpty()
                    ? null : previousContent);
            out.put("status", status);
            out.put("created_at", createdAt);
            return out;
        }
    }

    /** 当前 epoch 秒（double）。 */
    public static double now() {
        return System.currentTimeMillis() / 1000.0;
    }
}

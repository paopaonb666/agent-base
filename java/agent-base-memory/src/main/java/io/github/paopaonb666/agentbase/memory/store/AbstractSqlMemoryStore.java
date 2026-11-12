package io.github.paopaonb666.agentbase.memory.store;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeFormatterBuilder;
import java.time.temporal.ChronoField;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.StringJoiner;

/**
 * sqlite / MySQL 记忆后端共享的 DML 与行编解码（对应 Python 侧
 * {@code agent_base_memory/store/sql_base.py} 的 {@code _SqlMemoryStoreBase}）。
 *
 * <p>所有 SQL 用 JDBC {@code ?} 占位符书写——sqlite 与 MySQL 的 JDBC 驱动
 * 都认 {@code ?}，无需 Python 侧 {@code ?} → {@code %s} 的执行前整体替换
 * （本项目语句不含字面问号的前提同样成立）。整行覆盖式写入沿用 sqlite 的
 * {@code INSERT OR REPLACE} / MySQL 的 {@code REPLACE INTO}——同为"遇冲突
 * 删旧插新"的原子语义，要求提供全列值。
 *
 * <p>子类只需提供四个连接原语（{@link #execute} / {@link #fetchAll} /
 * 时间戳互转 {@link #toSqlTs} / {@link #fromSqlTs}）、同秒并列的排序 tiebreak
 * 与建表自举；所有高层仓储方法全部在此共享。
 *
 * <p><b>不移植 plan_snapshots</b>：Python 侧 sql_base.py 挂了
 * save_plan/get_plan/delete_plan 三个方法，Java 侧该表归基座
 * {@code core.storage}（{@code AbstractSqlCoreStore} 已实现
 * savePlan/getPlan/deletePlan），记忆子系统不重复持有。
 */
abstract class AbstractSqlMemoryStore implements MemoryStore {

    /** tags_json / source_refs_json / offsets_json / detail_json 的编解码。 */
    static final ObjectMapper JSON = new ObjectMapper();

    /**
     * 宽松时间戳解析：{@code yyyy-MM-dd HH:mm:ss} 带可选的微秒小数
     * （对应 Python {@code datetime.fromisoformat} 的兼容面）。
     */
    static final DateTimeFormatter FLEX_TS = new DateTimeFormatterBuilder()
            .appendPattern("yyyy-MM-dd HH:mm:ss")
            .optionalStart()
            .appendFraction(ChronoField.MICRO_OF_SECOND, 0, 6, true)
            .optionalEnd()
            .toFormatter();

    /** 微秒精度写入格式：memory_ops 的同秒并列依赖它稳定排序。 */
    static final DateTimeFormatter TS_WRITE =
            DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.SSSSSS");

    /** 整行覆盖式 upsert 的动词：sqlite 与 MySQL 各有拼写。 */
    private final String replaceVerb;

    /**
     * @param replaceVerb {@code "INSERT OR REPLACE"}（sqlite）或
     *                    {@code "REPLACE"}（MySQL），SQL 里后接 {@code INTO}
     */
    protected AbstractSqlMemoryStore(String replaceVerb) {
        this.replaceVerb = replaceVerb;
    }

    // ── 连接原语（子类实现） ────────────────────────────────────────

    /** 执行一条写语句，返回受影响行数；SQLException 由实现归一为运行时异常。 */
    protected abstract int execute(String sql, Object... params);

    /** 执行一条查询，返回按列顺序展开的行数组列表。 */
    protected abstract List<Object[]> fetchAll(String sql, Object... params);

    /** epoch 秒 → 存储时间戳（sqlite 是微秒 ISO 文本；MySQL 是 LocalDateTime）。 */
    protected abstract Object toSqlTs(double epochSeconds);

    /** 存储时间戳 → epoch 秒；解析失败按当前时间兜底并记 WARNING。 */
    protected abstract double fromSqlTs(Object value);

    /**
     * memory_ops 的同秒并列排序：sqlite 用 {@code rowid DESC}，MySQL 用
     * 自增 {@code seq DESC}（同一秒内多条 op 的稳定次序）。
     */
    protected abstract String opsTiebreak();

    /**
     * {@link #opsTiebreak()} 的裸列名（两实现均为 {@code "<col> DESC"}）：
     * 游标的同秒并列下推要在 WHERE 里比较该列（#160）。
     */
    protected final String opsTiebreakColumn() {
        String tiebreak = opsTiebreak();
        return tiebreak.substring(0, tiebreak.indexOf(' '));
    }

    /** memory_versions 的同秒并列排序（同 {@link #opsTiebreak()}）。 */
    protected abstract String versionsTiebreak();

    // ── 共享小工具 ──────────────────────────────────────────────────

    /** 整行覆盖式 upsert 的 SQL（占位符数量与列数一致）。 */
    private String replaceInto(String table, String... columns) {
        StringJoiner marks = new StringJoiner(", ");
        for (int i = 0; i < columns.length; i++) {
            marks.add("?");
        }
        return "%s INTO %s (%s) VALUES (%s)".formatted(
                replaceVerb, table, String.join(", ", columns), marks);
    }

    /** 绑定参数（全部用 setObject：本仓储的参数只会是字符串/数值/BLOB/时间戳）。 */
    protected static void bind(PreparedStatement ps, Object[] params) throws SQLException {
        for (int i = 0; i < params.length; i++) {
            ps.setObject(i + 1, params[i]);
        }
    }

    /** epoch 秒 → 本地时区的 LocalDateTime（对应 Python datetime.fromtimestamp）。 */
    static LocalDateTime toLocalDateTime(double epochSeconds) {
        long whole = (long) Math.floor(epochSeconds);
        long nanos = (long) Math.round((epochSeconds - whole) * 1e9);
        return LocalDateTime.ofInstant(Instant.ofEpochSecond(whole, nanos), ZoneId.systemDefault());
    }

    /** LocalDateTime → epoch 秒（对应 Python datetime.timestamp()，微秒精度）。 */
    static double toEpochSeconds(LocalDateTime dateTime) {
        // 写入是微秒精度（TS_WRITE），读取也必须回到微秒——toEpochMilli()
        // 会把亚毫秒位抹掉，updated_at 往返一次丢精度（#301 实测）。
        java.time.Instant instant = dateTime.atZone(ZoneId.systemDefault()).toInstant();
        return instant.getEpochSecond() + instant.getNano() / 1_000_000_000.0;
    }

    /** 当前时刻的 epoch 秒（对应 Python time.time()）。 */
    static double epochNow() {
        return System.currentTimeMillis() / 1000.0;
    }

    private static String str(Object value) {
        return value == null ? "" : value.toString();
    }

    /** 数值宽容转换（对应 Python int(row[x] or 0)：null/空/坏值一律 0）。 */
    private static int toInt(Object value) {
        if (value instanceof Number number) {
            return number.intValue();
        }
        if (value instanceof String text && !text.isEmpty()) {
            try {
                return Integer.parseInt(text.trim());
            } catch (NumberFormatException ignored) {
                // 落到兜底 0。
            }
        }
        return 0;
    }

    /** REAL/DOUBLE 列的宽容转换；坏值按 fallback 兜底。 */
    private static double toDouble(Object value, double fallback) {
        if (value instanceof Number number) {
            return number.doubleValue();
        }
        if (value instanceof String text && !text.isEmpty()) {
            try {
                return Double.parseDouble(text.trim());
            } catch (NumberFormatException ignored) {
                // 落到兜底。
            }
        }
        return fallback;
    }

    /** BLOB 列读取；非字节值按缺失处理（空数组 = 未向量化）。 */
    private static byte[] toBlob(Object value) {
        return value instanceof byte[] bytes ? bytes : null;
    }

    private static final Logger LOG = LoggerFactory.getLogger(AbstractSqlMemoryStore.class);

    /**
     * 容错解析 JSON 字符串列表：空值返回空表；畸形/非数组（损坏的
     * tags/引用列）返回空表但 WARN（#169）——静默丢 tags 让记忆行以
     * 「无标签」形态继续使用，是查不到原因的数据退化。读路径不抛错
     * （一条坏行不该打断检索）。对应 Python 侧 {@code _json_list}。
     */
    private static List<String> jsonList(Object value, String memoryId, String column) {
        if (value == null || value.toString().isEmpty()) {
            return List.of();
        }
        try {
            JsonNode parsed = JSON.readTree(value.toString());
            if (!parsed.isArray()) {
                LOG.warn("memory store: {} 列不是 JSON 数组（memory_id={}），按空处理：{}",
                        column, memoryId, value);
                return List.of();
            }
            List<String> out = new ArrayList<>();
            for (JsonNode item : parsed) {
                // 对应 Python str(item)：标量取文本形态，容器取 JSON 形态。
                out.add(item.isValueNode() ? item.asText() : item.toString());
            }
            return out;
        } catch (JsonProcessingException e) {
            LOG.warn("memory store: {} 列 JSON 解析失败（memory_id={}），按空处理：{}",
                    column, memoryId, value);
            return List.of();
        }
    }

    /** JSON 字符串列表序列化（对应 json.dumps(..., ensure_ascii=False)）。 */
    private static String writeStringList(List<String> values) {
        try {
            return JSON.writeValueAsString(values == null ? List.of() : values);
        } catch (JsonProcessingException e) {
            throw new IllegalStateException("JSON 列表序列化失败", e);
        }
    }

    /** offsets → JSON 数组（{@code [[start,end],…]}）；空/缺失写 NULL。 */
    private static String encodeOffsets(List<long[]> offsets) {
        if (offsets == null || offsets.isEmpty()) {
            return null;
        }
        StringBuilder sb = new StringBuilder("[");
        for (long[] pair : offsets) {
            if (sb.length() > 1) {
                sb.append(',');
            }
            sb.append('[').append(pair[0]).append(',').append(pair[1]).append(']');
        }
        return sb.append(']').toString();
    }

    /**
     * offsets_json → {@code ((start,end),…)}；畸形/非数组一律 null（前端
     * 降级为纯卡片视图）。对应 Python {@code _decode_offsets}。
     */
    private static List<long[]> decodeOffsets(Object value) {
        if (value == null || value.toString().isEmpty()) {
            return null;
        }
        try {
            JsonNode parsed = JSON.readTree(value.toString());
            if (!parsed.isArray()) {
                return null;
            }
            List<long[]> out = new ArrayList<>();
            for (JsonNode seg : parsed) {
                if (seg.isArray() && seg.size() == 2) {
                    out.add(new long[]{seg.get(0).asLong(), seg.get(1).asLong()});
                }
            }
            // 对应 Python `tuple(...) or None`：空结果按缺失处理。
            return out.isEmpty() ? null : out;
        } catch (JsonProcessingException e) {
            return null;
        }
    }

    /**
     * detail 字典序列化：Python 用 {@code default=str} 把不可序列化的值
     * 转成字符串，这里先对非 Jackson 原生类型的值做一层 String.valueOf
     * 兜底（语义一致）；极端嵌套场景序列化仍失败时按空字典兜底并告警。
     */
    private static String writeDetail(Map<String, Object> detail) {
        Map<String, Object> safe = new LinkedHashMap<>();
        for (Map.Entry<String, Object> e : detail.entrySet()) {
            Object v = e.getValue();
            if (v == null || v instanceof String || v instanceof Number
                    || v instanceof Boolean || v instanceof Map || v instanceof java.util.Collection) {
                safe.put(e.getKey(), v);
            } else {
                safe.put(e.getKey(), String.valueOf(v));
            }
        }
        try {
            return JSON.writeValueAsString(safe);
        } catch (JsonProcessingException e) {
            return "{}";
        }
    }

    /** detail_json 容错解析：畸形/非字典一律空字典（对应 Python _op_from_row）。 */
    private static Map<String, Object> parseDetail(Object value) {
        if (value == null || value.toString().isEmpty()) {
            return new LinkedHashMap<>();
        }
        try {
            JsonNode parsed = JSON.readTree(value.toString());
            if (!parsed.isObject()) {
                return new LinkedHashMap<>();
            }
            return JSON.convertValue(parsed, new TypeReference<Map<String, Object>>() {});
        } catch (JsonProcessingException e) {
            return new LinkedHashMap<>();
        }
    }

    // ── 行编解码 ────────────────────────────────────────────────────

    /** memories 行的写入参数（时间戳列与 access_count 由 upsert 单独追加）。 */
    private Object[] recordParams(Models.MemoryRecord record) {
        return new Object[]{
                record.memoryId(),
                record.userId(),
                record.agentId(),
                record.kind(),
                record.content(),
                writeStringList(record.tags()),
                record.embedding().length == 0 ? null : record.embedding(),
                record.embeddingDim(),
                record.salience(),
                record.status(),
                record.sourceThreadId(),
                writeStringList(record.sourceRefs()),
        };
    }

    /** 一行 memories 的列清单（读取顺序与 {@link #recordFromRow} 对齐）。 */
    private static final String MEMORY_COLUMNS =
            "memory_id, user_id, agent_id, kind, content, tags_json, embedding, embedding_dim,"
                    + " salience, status, source_thread_id, source_refs_json, created_at, updated_at,"
                    + " last_accessed_at, access_count";

    private Models.MemoryRecord recordFromRow(Object[] row) {
        return new Models.MemoryRecord(
                str(row[0]),
                str(row[1]),
                str(row[2]),
                str(row[3]),
                str(row[4]),
                jsonList(row[5], str(row[0]), "tags"),
                toBlob(row[6]),
                row[7] == null ? null : Integer.valueOf(toInt(row[7])),
                toDouble(row[8], 0.5),
                str(row[9]),
                str(row[10]),
                jsonList(row[11], str(row[0]), "source_refs"),
                fromSqlTs(row[12]),
                fromSqlTs(row[13]),
                row[14] == null ? null : Double.valueOf(fromSqlTs(row[14])),
                toInt(row[15]));
    }

    private static Object[] blockParams(Models.MemoryBlock block) {
        return new Object[]{
                block.userId(), block.agentId(), block.label(), block.content(),
                block.charLimit(), block.version(),
        };
    }

    private static final String BLOCK_COLUMNS =
            "user_id, agent_id, label, content, char_limit, version, updated_at";

    private Models.MemoryBlock blockFromRow(Object[] row) {
        return new Models.MemoryBlock(
                str(row[0]),
                str(row[1]),
                str(row[2]),
                str(row[3]),
                toInt(row[4]),
                toInt(row[5]),
                fromSqlTs(row[6]));
    }

    private static final String SUMMARY_COLUMNS =
            "user_id, thread_id, summary, agent_id, covered_message_count, updated_at";

    private Models.SessionSummary summaryFromRow(Object[] row) {
        return new Models.SessionSummary(
                str(row[0]),
                str(row[1]),
                str(row[2]),
                str(row[3]),
                toInt(row[4]),
                fromSqlTs(row[5]));
    }

    private static final String CHUNK_COLUMNS =
            "chunk_id, file_id, thread_id, user_id, agent_id, ordinal, text, offsets_json,"
                    + " embedding, embedding_dim, created_at";

    private Models.DocChunk chunkFromRow(Object[] row) {
        return new Models.DocChunk(
                str(row[0]),
                str(row[1]),
                str(row[2]),
                str(row[3]),
                str(row[4]),
                toInt(row[5]),
                str(row[6]),
                decodeOffsets(row[7]),
                toBlob(row[8]),
                row[9] == null ? null : Integer.valueOf(toInt(row[9])),
                fromSqlTs(row[10]));
    }

    private static final String OP_COLUMNS =
            "op_id, op, user_id, agent_id, thread_id, detail_json, status, error_text,"
                    + " duration_ms, created_at";

    private Models.MemoryOp opFromRow(Object[] row) {
        Map<String, Object> detail = parseDetail(row[5]);
        return new Models.MemoryOp(
                str(row[0]),
                str(row[1]),
                str(row[2]),
                str(row[3]),
                str(row[4]),
                detail,
                row[6] == null ? "ok" : str(row[6]),
                str(row[7]),
                toInt(row[8]),
                fromSqlTs(row[9]));
    }

    private static final String VERSION_COLUMNS =
            "version_id, memory_id, user_id, op, content, previous_content, status, created_at";

    private Models.MemoryVersion versionFromRow(Object[] row) {
        return new Models.MemoryVersion(
                str(row[0]),
                str(row[1]),
                str(row[2]),
                str(row[3]),
                str(row[4]),
                str(row[5]),
                row[6] == null ? "active" : str(row[6]),
                fromSqlTs(row[7]));
    }

    // ── memories：跨会话长期记忆 ─────────────────────────────────────

    @Override
    public void upsertMemory(Models.MemoryRecord record) {
        if (!Models.KNOWN_MEMORY_KINDS.contains(record.kind())) {
            throw new Models.MemoryStoreError(
                    "unknown memory kind '%s'; expected one of %s"
                            .formatted(record.kind(), Models.KNOWN_MEMORY_KINDS));
        }
        if (!Models.KNOWN_MEMORY_STATUSES.contains(record.status())) {
            throw new Models.MemoryStoreError(
                    "unknown memory status '%s'; expected one of %s"
                            .formatted(record.status(), Models.KNOWN_MEMORY_STATUSES));
        }
        // 时间戳列与 access_count 不在 recordParams 里：它们由后端的
        // 时间格式决定，逐个追加（null 的 last_accessed_at 保持 NULL）。
        execute(replaceInto("memories",
                        "memory_id", "user_id", "agent_id", "kind", "content", "tags_json",
                        "embedding", "embedding_dim", "salience", "status", "source_thread_id",
                        "source_refs_json", "created_at", "updated_at", "last_accessed_at",
                        "access_count"),
                concat(recordParams(record), new Object[]{
                        toSqlTs(record.createdAt()),
                        toSqlTs(record.updatedAt()),
                        record.lastAccessedAt() == null ? null : toSqlTs(record.lastAccessedAt()),
                        record.accessCount(),
                }));
        // 关键词索引同步（#326）：sqlite 覆写同步 FTS 表；MySQL FULLTEXT
        // 引擎自维护（默认 no-op）。
        indexAfterUpsert(record);
    }

    /** 关键词索引同步钩子（#326）：REPLACE 覆盖写后重建该行的索引项。 */
    protected void indexAfterUpsert(Models.MemoryRecord record) {
    }

    /** 关键词索引同步钩子（#326）：行删除后清索引项。 */
    protected void indexAfterDelete(String memoryId) {
    }

    /**
     * 关键词索引同步钩子（#326）：按线程清理前调用——sqlite 需在
     * memories 行仍可见时按 source_thread_id 子查询定位索引项。
     */
    protected void indexBeforeThreadDelete(String threadId) {
    }

    /** 拼接两段参数（模拟 Python 的元组解包追加）。 */
    private static Object[] concat(Object[] head, Object[] tail) {
        Object[] out = new Object[head.length + tail.length];
        System.arraycopy(head, 0, out, 0, head.length);
        System.arraycopy(tail, 0, out, head.length, tail.length);
        return out;
    }

    @Override
    public Models.MemoryRecord getMemory(String memoryId) {
        List<Object[]> rows = fetchAll(
                "SELECT " + MEMORY_COLUMNS + " FROM memories WHERE memory_id = ?", memoryId);
        return rows.isEmpty() ? null : recordFromRow(rows.get(0));
    }

    @Override
    public boolean deleteMemory(String memoryId) {
        boolean deleted = execute("DELETE FROM memories WHERE memory_id = ?", memoryId) > 0;
        if (deleted) {
            indexAfterDelete(memoryId);
        }
        return deleted;
    }

    @Override
    public List<Models.MemoryRecord> listMemories(String userId, String agentId,
                                                  boolean includeGlobal, List<String> kinds,
                                                  List<String> statuses, int limit,
                                                  boolean excludeProfile) {
        return listMemories(userId, agentId, includeGlobal, kinds, statuses, limit,
                excludeProfile, null);
    }

    /** #180：episodic TTL 下推——过期行在 SQL 层被排除，不占候选名额。 */
    @Override
    public List<Models.MemoryRecord> listMemories(String userId, String agentId,
                                                  boolean includeGlobal, List<String> kinds,
                                                  List<String> statuses, int limit,
                                                  boolean excludeProfile,
                                                  Double episodicCutoffBefore) {
        StringBuilder sql = new StringBuilder("SELECT ").append(MEMORY_COLUMNS)
                .append(" FROM memories WHERE user_id = ?");
        List<Object> params = new ArrayList<>();
        params.add(userId);
        // 浏览/检索列表默认排除画像：画像由上下文组装与画像页签单独
        // 呈现，混进记忆列表既是实现细节泄漏也是重复展示。
        if (excludeProfile) {
            sql.append(" AND memory_id NOT LIKE 'profile:%'");
        }
        if (statuses != null && !statuses.isEmpty()) {
            sql.append(" AND status IN (").append(placeholders(statuses.size())).append(')');
            params.addAll(statuses);
        }
        if (kinds != null && !kinds.isEmpty()) {
            sql.append(" AND kind IN (").append(placeholders(kinds.size())).append(')');
            params.addAll(kinds);
        }
        if (episodicCutoffBefore != null) {
            // 时间口径与列写入一致（toSqlTs），边界取 >=（与内存过滤一致：
            // updatedAt >= cutoff 保留）。
            sql.append(" AND (kind != 'episodic' OR updated_at >= ?)");
            params.add(toSqlTs(episodicCutoffBefore));
        }
        if (agentId != null) {
            if (includeGlobal) {
                sql.append(" AND (agent_id = ? OR agent_id = '*')");
            } else {
                sql.append(" AND agent_id = ?");
            }
            params.add(agentId);
        }
        // 同刻并列的确定性次序键（review B-P3）：updated_at 是毫秒粒度
        // （Models.now()），一轮 capture 写出的多条记忆常同值；MySQL
        // TIMESTAMP(0) 更把同秒全部压成同值——sqlite 的 INSERT OR REPLACE
        // 翻 rowid，会让同刻并列组的返回序在两次查询间互换，RRF 通道排名/
        // top-k 边界随之漂移（候选序是 sortByScoreDescending 与 RRF 并列
        // 判定的底序）。追加 memory_id 作确定性 tiebreak（与 ops 的
        // opsTiebreak 同款收口）。
        sql.append(" ORDER BY updated_at DESC, memory_id DESC LIMIT ?");
        params.add(Math.max(1, limit));
        List<Object[]> rows = fetchAll(sql.toString(), params.toArray());
        List<Models.MemoryRecord> out = new ArrayList<>();
        for (Object[] row : rows) {
            out.add(recordFromRow(row));
        }
        return out;
    }

    /** N 个 {@code ?} 占位符（IN 子句用）。 */
    private static String placeholders(int n) {
        StringJoiner marks = new StringJoiner(", ");
        for (int i = 0; i < n; i++) {
            marks.add("?");
        }
        return marks.toString();
    }

    @Override
    public void touchMemories(List<String> memoryIds) {
        if (memoryIds == null || memoryIds.isEmpty()) {
            return;
        }
        execute("UPDATE memories SET last_accessed_at = ?, access_count = access_count + 1"
                        + " WHERE memory_id IN (" + placeholders(memoryIds.size()) + ")",
                concat(new Object[]{toSqlTs(epochNow())}, memoryIds.toArray()));
    }

    /**
     * 回填查询：active 且向量缺失/维度不符的非画像记忆（锐评 #3）。
     *
     * <p>画像记录（profile:*）刻意排除——它们整体注入不走向量召回，且
     * 永远没有向量，纳入会让回填死循环。
     */
    @Override
    public List<Models.MemoryRecord> listMemoriesNeedingEmbedding(int dims, int limit) {
        List<Object[]> rows = fetchAll(
                "SELECT " + MEMORY_COLUMNS + " FROM memories WHERE status = 'active'"
                        + " AND memory_id NOT LIKE 'profile:%'"
                        + " AND (embedding IS NULL OR embedding_dim IS NULL OR embedding_dim != ?)"
                        + " ORDER BY updated_at DESC LIMIT ?",
                dims, Math.max(1, limit));
        List<Models.MemoryRecord> out = new ArrayList<>();
        for (Object[] row : rows) {
            out.add(recordFromRow(row));
        }
        return out;
    }

    // ── memory_blocks：常驻上下文的记忆块 ────────────────────────────

    @Override
    public void upsertBlock(Models.MemoryBlock block) {
        execute(replaceInto("memory_blocks",
                        "user_id", "agent_id", "label", "content", "char_limit", "version",
                        "updated_at"),
                concat(blockParams(block), new Object[]{toSqlTs(block.updatedAt())}));
    }

    @Override
    public Models.MemoryBlock getBlock(String userId, String agentId, String label) {
        List<Object[]> rows = fetchAll("SELECT " + BLOCK_COLUMNS + " FROM memory_blocks"
                + " WHERE user_id = ? AND agent_id = ? AND label = ?", userId, agentId, label);
        return rows.isEmpty() ? null : blockFromRow(rows.get(0));
    }

    @Override
    public List<Models.MemoryBlock> listBlocks(String userId, String agentId) {
        List<Object[]> rows = fetchAll("SELECT " + BLOCK_COLUMNS + " FROM memory_blocks"
                + " WHERE user_id = ? AND agent_id = ? ORDER BY label ASC", userId, agentId);
        List<Models.MemoryBlock> out = new ArrayList<>();
        for (Object[] row : rows) {
            out.add(blockFromRow(row));
        }
        return out;
    }

    @Override
    public boolean deleteBlock(String userId, String agentId, String label) {
        return execute("DELETE FROM memory_blocks WHERE user_id = ? AND agent_id = ? AND label = ?",
                userId, agentId, label) > 0;
    }

    // ── session_summaries：线程滚动摘要 ──────────────────────────────

    @Override
    public Models.SessionSummary getSummary(String userId, String threadId) {
        List<Object[]> rows = fetchAll("SELECT " + SUMMARY_COLUMNS + " FROM session_summaries"
                + " WHERE user_id = ? AND thread_id = ?", userId, threadId);
        return rows.isEmpty() ? null : summaryFromRow(rows.get(0));
    }

    @Override
    public void upsertSummary(Models.SessionSummary summary) {
        execute(replaceInto("session_summaries",
                        "user_id", "thread_id", "summary", "agent_id", "covered_message_count",
                        "updated_at"),
                summary.userId(), summary.threadId(), summary.summary(), summary.agentId(),
                summary.coveredMessageCount(), toSqlTs(summary.updatedAt()));
    }

    // ── doc_chunks：文档知识库分块 ───────────────────────────────────

    @Override
    public void putChunks(List<Models.DocChunk> chunks) {
        for (Models.DocChunk chunk : chunks) {
            execute(replaceInto("doc_chunks",
                            "chunk_id", "file_id", "thread_id", "user_id", "agent_id", "ordinal",
                            "text", "offsets_json", "embedding", "embedding_dim", "created_at"),
                    chunk.chunkId(), chunk.fileId(), chunk.threadId(), chunk.userId(),
                    chunk.agentId(), chunk.ordinal(), chunk.text(),
                    encodeOffsets(chunk.offsets()),
                    chunk.embedding().length == 0 ? null : chunk.embedding(),
                    chunk.embeddingDim(),
                    toSqlTs(chunk.createdAt()));
        }
    }

    @Override
    public List<Models.DocChunk> listChunks(String userId, String agentId, String fileId,
                                            int limit) {
        StringBuilder sql = new StringBuilder("SELECT ").append(CHUNK_COLUMNS)
                .append(" FROM doc_chunks WHERE user_id = ?");
        List<Object> params = new ArrayList<>();
        params.add(userId);
        if (agentId != null) {
            sql.append(" AND (agent_id = ? OR agent_id = '*')");
            params.add(agentId);
        }
        if (fileId != null) {
            sql.append(" AND file_id = ?");
            params.add(fileId);
        }
        sql.append(" ORDER BY file_id ASC, ordinal ASC LIMIT ?");
        params.add(Math.max(1, limit));
        List<Object[]> rows = fetchAll(sql.toString(), params.toArray());
        List<Models.DocChunk> out = new ArrayList<>();
        for (Object[] row : rows) {
            out.add(chunkFromRow(row));
        }
        return out;
    }

    @Override
    public int countChunksForFile(String fileId) {
        List<Object[]> rows = fetchAll(
                "SELECT COUNT(*) FROM doc_chunks WHERE file_id = ?", fileId);
        return rows.isEmpty() ? 0 : toInt(rows.get(0)[0]);
    }

    @Override
    public int deleteChunksForFile(String fileId) {
        return execute("DELETE FROM doc_chunks WHERE file_id = ?", fileId);
    }

    @Override
    public Map<String, Integer> chunkCountsByFile(String userId) {
        List<Object[]> rows = fetchAll(
                "SELECT file_id, COUNT(*) FROM doc_chunks WHERE user_id = ? GROUP BY file_id",
                userId);
        Map<String, Integer> out = new LinkedHashMap<>();
        for (Object[] row : rows) {
            out.put(str(row[0]), toInt(row[1]));
        }
        return out;
    }

    /** 回填查询：向量缺失或维度不符的知识库分块（与 memories 的回填同语义）。 */
    @Override
    public List<Models.DocChunk> listChunksNeedingEmbedding(int dims, int limit) {
        List<Object[]> rows = fetchAll(
                "SELECT " + CHUNK_COLUMNS + " FROM doc_chunks"
                        + " WHERE embedding IS NULL OR embedding_dim IS NULL OR embedding_dim != ?"
                        + " ORDER BY created_at DESC LIMIT ?",
                dims, Math.max(1, limit));
        List<Models.DocChunk> out = new ArrayList<>();
        for (Object[] row : rows) {
            out.add(chunkFromRow(row));
        }
        return out;
    }

    // ── 线程级联 ────────────────────────────────────────────────────

    @Override
    public void deleteForThread(String threadId) {
        // memories 刻意不删：跨会话记忆独立于线程存活（出处仍在
        // source_thread_id）；随线程消亡的是它的摘要与知识库分块
        // （会话索引与计划快照归基座 core.storage 管理）。
        execute("DELETE FROM doc_chunks WHERE thread_id = ?", threadId);
        execute("DELETE FROM session_summaries WHERE thread_id = ?", threadId);
    }

    /** #172：显式按线程清理跨会话记忆（purge_memories=true 才会走到）。 */
    @Override
    public int deleteMemoriesForThread(String threadId) {
        // 索引清理先于行删除（#326）：sqlite 需按仍可见的 memories 行定位
        // FTS 索引项；MySQL 无独立索引表（no-op）。
        indexBeforeThreadDelete(threadId);
        return execute("DELETE FROM memories WHERE source_thread_id = ?", threadId);
    }

    // ── 操作审计：memory_ops ─────────────────────────────────────────

    @Override
    public void recordOp(Models.MemoryOp op) {
        execute("INSERT INTO memory_ops (" + OP_COLUMNS + ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                op.opId(),
                op.op(),
                op.userId(),
                op.agentId(),
                op.threadId(),
                writeDetail(op.detail()),
                op.status(),
                op.errorText(),
                op.durationMs(),
                toSqlTs(op.createdAt()));
    }

    @Override
    public List<Models.MemoryOp> listOps(String userId, int limit) {
        return listOps(userId, null, null, null, limit);
    }

    /**
     * #160/#176：thread/since/游标全部在 SQL 层下推。游标 {@code before}
     * 的语义是严格早于边界行——{@code created_at < 边界} 或（同刻并列时）
     * 按 opsTiebreak 前移；边界行的 rowid/seq 由其 op_id 反查（op_id 是
     * 主键，子查询恒单值；游标未带 op_id 时退化为纯时间戳截断，同刻并列
     * 行的余量不返回——API 侧 next_before 只有秒值的既有约束）。
     */
    @Override
    public List<Models.MemoryOp> listOps(String userId, String threadId,
                                         Double sinceEpochSeconds, MemoryStore.OpsCursor before,
                                         int limit) {
        StringBuilder sql = new StringBuilder("SELECT ").append(OP_COLUMNS)
                .append(" FROM memory_ops");
        List<Object> params = new ArrayList<>();
        if (userId != null) {
            sql.append(" WHERE user_id = ?");
            params.add(userId);
        }
        if (threadId != null) {
            // WHERE/AND 按「是否已有 WHERE」推进（review B-P5）：userId 缺席
            // + threadId 在场的组合此前生成 `FROM memory_ops AND thread_id = ?`
            // 的非法 SQL——当前接线不可达（调用方 userId 来自认证非空），
            // 纯潜伏；InMemory 版对同组合语义正常（null=跨用户）。
            sql.append(userId == null ? " WHERE thread_id = ?" : " AND thread_id = ?");
            params.add(threadId);
        }
        if (sinceEpochSeconds != null) {
            // 时间口径与列写入一致（toSqlTs，#301 时间戳矩阵）。
            sql.append(" AND created_at >= ?");
            params.add(toSqlTs(sinceEpochSeconds));
        }
        if (before != null) {
            Object boundary = toSqlTs(before.beforeEpochSeconds());
            if (before.beforeOpId() != null && !before.beforeOpId().isEmpty()) {
                // 带边界行 op_id：同刻并列按 tiebreak 前移（子查询恒单值，
                // op_id 不存在时子查询为 NULL、整支不匹配——退化为纯截断）。
                String column = opsTiebreakColumn();
                sql.append(" AND (created_at < ? OR (created_at = ?");
                params.add(boundary);
                params.add(boundary);
                sql.append(" AND ").append(column).append(" < (SELECT ").append(column)
                        .append(" FROM memory_ops WHERE op_id = ?)");
                params.add(before.beforeOpId());
                sql.append("))");
            } else {
                // 只带秒值（API 契约）：严格早于该时刻，同刻并列行的余量
                // 不再返回（见 MemoryStore.OpsCursor 的说明）。
                sql.append(" AND created_at < ?");
                params.add(boundary);
            }
        }
        sql.append(" ORDER BY created_at DESC, ").append(opsTiebreak()).append(" LIMIT ?");
        params.add(Math.max(1, limit));
        List<Object[]> rows = fetchAll(sql.toString(), params.toArray());
        List<Models.MemoryOp> out = new ArrayList<>();
        for (Object[] row : rows) {
            out.add(opFromRow(row));
        }
        return out;
    }

    // ── 版本史：memory_versions ──────────────────────────────────────

    @Override
    public void recordVersion(Models.MemoryVersion version) {
        execute("INSERT INTO memory_versions (" + VERSION_COLUMNS + ")"
                        + " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                version.versionId(),
                version.memoryId(),
                version.userId(),
                version.op(),
                version.content(),
                // 空串归一为 NULL：语义上是"没有前值"（create）。
                version.previousContent().isEmpty() ? null : version.previousContent(),
                version.status(),
                toSqlTs(version.createdAt()));
    }

    @Override
    public List<Models.MemoryVersion> listVersions(String memoryId, int limit) {
        List<Object[]> rows = fetchAll("SELECT " + VERSION_COLUMNS + " FROM memory_versions"
                        + " WHERE memory_id = ? ORDER BY created_at DESC, " + versionsTiebreak()
                        + " LIMIT ?",
                memoryId, Math.max(1, limit));
        List<Models.MemoryVersion> out = new ArrayList<>();
        for (Object[] row : rows) {
            out.add(versionFromRow(row));
        }
        return out;
    }

    @Override
    public Models.MemoryVersion getVersion(String versionId) {
        List<Object[]> rows = fetchAll("SELECT " + VERSION_COLUMNS + " FROM memory_versions"
                + " WHERE version_id = ?", versionId);
        return rows.isEmpty() ? null : versionFromRow(rows.get(0));
    }
}

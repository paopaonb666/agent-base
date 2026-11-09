package io.github.paopaonb666.agentbase.core.storage;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

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
 * sqlite / MySQL 基座存储共享的 DML 与行编解码（对应 Python 侧
 * {@code sql_base.py} 的 {@code _CoreSqlStoreBase}）。
 *
 * <p>所有 SQL 用 JDBC {@code ?} 占位符书写——sqlite 与 MySQL 的 JDBC 驱动
 * 都认 {@code ?}，无需 Python 侧 {@code ?} → {@code %s} 的执行前整体替换
 * （本项目语句不含字面问号的前提同样成立）。整行覆盖式写入沿用 sqlite 的
 * {@code INSERT OR REPLACE} / MySQL 的 {@code REPLACE INTO}——同为"遇冲突
 * 删旧插新"的原子语义，要求提供全列值。
 *
 * <p>子类只需提供两个连接原语（{@link #execute} / {@link #fetchAll}）、
 * 时间戳互转与建表自举；高层仓储方法全部在此共享。
 */
abstract class AbstractSqlCoreStore implements CoreStore {

    /** tasks_json 的编解码（对应 Python json.dumps/loads；Jackson 不转义非 ASCII）。 */
    static final ObjectMapper TASKS_JSON = new ObjectMapper();

    /** 一行 thread_index 的列清单（读取顺序与 {@link #threadFromRow} 对齐）。 */
    private static final String THREAD_COLUMNS =
            "thread_id, user_id, module, title, created_at, updated_at";

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

    /** 微秒精度写入格式：线程列表的同秒并列依赖它稳定排序。 */
    static final DateTimeFormatter TS_WRITE =
            DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.SSSSSS");

    /** 整行覆盖式 upsert 的动词：sqlite 与 MySQL 各有拼写。 */
    private final String replaceVerb;

    /** 缺席才插入的动词（claim 原语用）：sqlite 与 MySQL 各有拼写。 */
    private final String ignoreVerb;

    /**
     * @param replaceVerb {@code "INSERT OR REPLACE"}（sqlite）或
     *                    {@code "REPLACE"}（MySQL），SQL 里后接 {@code INTO}
     * @param ignoreVerb  {@code "INSERT OR IGNORE"}（sqlite）或
     *                    {@code "INSERT IGNORE"}（MySQL），SQL 里后接 {@code INTO}
     */
    protected AbstractSqlCoreStore(String replaceVerb, String ignoreVerb) {
        this.replaceVerb = replaceVerb;
        this.ignoreVerb = ignoreVerb;
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

    /** 缺席才插入的 SQL（claim 原语第一步；属主判定交给后面的条件 UPDATE）。 */
    private String insertIgnoreInto(String table, String... columns) {
        StringJoiner marks = new StringJoiner(", ");
        for (int i = 0; i < columns.length; i++) {
            marks.add("?");
        }
        return "%s INTO %s (%s) VALUES (%s)".formatted(
                ignoreVerb, table, String.join(", ", columns), marks);
    }

    /** 绑定参数（全部用 setObject：本仓储的参数只会是字符串/数值/时间）。 */
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

    /** LocalDateTime → epoch 秒（对应 Python datetime.timestamp()）。 */
    static double toEpochSeconds(LocalDateTime dateTime) {
        return dateTime.atZone(ZoneId.systemDefault()).toInstant().toEpochMilli() / 1000.0;
    }

    /** 当前时刻的 epoch 秒（对应 Python time.time()）。 */
    static double epochNow() {
        return System.currentTimeMillis() / 1000.0;
    }

    private static String str(Object value) {
        return value == null ? "" : value.toString();
    }

    /** 数值宽容转换（对应 Python int(cursor or 0)：null/空/坏值一律 0）。 */
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

    /**
     * tasks_json → 任务清单；损坏/空 JSON 一律返回空清单（对应 Python
     * json.loads 失败 → {@code []}、非 list 也回 {@code []}）。清单里的
     * 非对象元素按 Port 载荷契约（List&lt;Map&gt;）丢弃。
     */
    private static List<Map<String, Object>> parseTasks(Object raw) {
        String json = raw == null ? "[]" : raw.toString();
        if (json.isEmpty()) {
            json = "[]";
        }
        try {
            JsonNode node = TASKS_JSON.readTree(json);
            List<Map<String, Object>> out = new ArrayList<>();
            if (node.isArray()) {
                for (JsonNode item : node) {
                    if (item.isObject()) {
                        out.add(TASKS_JSON.convertValue(item,
                                new TypeReference<Map<String, Object>>() {}));
                    }
                }
            }
            return out;
        } catch (JsonProcessingException e) {
            return new ArrayList<>();
        }
    }

    private static String writeTasksJson(List<Map<String, Object>> tasks) {
        try {
            return TASKS_JSON.writeValueAsString(tasks == null ? List.of() : tasks);
        } catch (JsonProcessingException e) {
            throw new IllegalStateException("计划任务清单无法 JSON 化", e);
        }
    }

    // ── 会话索引（thread_index；S1 线程作用域） ─────────────────────

    private ThreadIndexRecord threadFromRow(Object[] row) {
        return new ThreadIndexRecord(
                str(row[0]),
                str(row[1]),
                str(row[2]),
                str(row[3]),
                fromSqlTs(row[4]),
                fromSqlTs(row[5]));
    }

    @Override
    public void upsertThreadIndex(ThreadIndexRecord entry) {
        execute(replaceInto("thread_index",
                        "thread_id", "user_id", "module", "title", "created_at", "updated_at"),
                entry.threadId(), entry.userId(), entry.module(), entry.title(),
                toSqlTs(entry.createdAt()), toSqlTs(entry.updatedAt()));
    }

    @Override
    public ThreadIndexRecord getThreadIndex(String threadId) {
        List<Object[]> rows = fetchAll(
                "SELECT " + THREAD_COLUMNS + " FROM thread_index WHERE thread_id = ?", threadId);
        return rows.isEmpty() ? null : threadFromRow(rows.get(0));
    }

    @Override
    public ThreadIndexRecord claimThreadIndex(ThreadIndexRecord entry) {
        // 三条单语句各自原子（连接原语每语句独立提交）：并发首发同一
        // 线程时 INSERT [OR] IGNORE 只有一家生效，输家的 UPDATE 匹配
        // 0 行、SELECT 读到赢家的行——属主永远不会被翻转。不用
        // ON DUPLICATE KEY UPDATE ... WHERE：MySQL 5.7 不支持条件写法。
        execute(insertIgnoreInto("thread_index",
                        "thread_id", "user_id", "module", "title", "created_at", "updated_at"),
                entry.threadId(), entry.userId(), entry.module(), entry.title(),
                toSqlTs(entry.createdAt()), toSqlTs(entry.updatedAt()));
        execute("UPDATE thread_index SET updated_at = ? WHERE thread_id = ? AND user_id = ?",
                toSqlTs(entry.updatedAt()), entry.threadId(), entry.userId());
        List<Object[]> rows = fetchAll(
                "SELECT " + THREAD_COLUMNS + " FROM thread_index WHERE thread_id = ?",
                entry.threadId());
        // INSERT [OR] IGNORE 之后行必在席；防御性回退避免 NPE。
        return rows.isEmpty() ? entry : threadFromRow(rows.get(0));
    }

    @Override
    public List<ThreadIndexRecord> listThreadIndexes(String userId, String module, int limit) {
        StringBuilder sql = new StringBuilder("SELECT ").append(THREAD_COLUMNS)
                .append(" FROM thread_index WHERE user_id = ?");
        List<Object> params = new ArrayList<>();
        params.add(userId);
        if (module != null) {
            sql.append(" AND module = ?");
            params.add(module);
        }
        // 微秒精度时间戳的字典序 = 时间序：ORDER BY updated_at DESC 即按最近优先。
        sql.append(" ORDER BY updated_at DESC LIMIT ?");
        params.add(Math.max(1, limit));
        List<Object[]> rows = fetchAll(sql.toString(), params.toArray());
        List<ThreadIndexRecord> out = new ArrayList<>();
        for (Object[] row : rows) {
            out.add(threadFromRow(row));
        }
        return out;
    }

    @Override
    public void deleteThreadIndex(String threadId) {
        execute("DELETE FROM thread_index WHERE thread_id = ?", threadId);
    }

    // ── 计划快照（plan_snapshots；M10 跨轮延续） ────────────────────

    @Override
    public void savePlan(String threadId, String module, List<Map<String, Object>> tasks,
                         int cursor, int replans) {
        execute(replaceInto("plan_snapshots",
                        "thread_id", "module", "tasks_json", "cursor", "replans", "updated_at"),
                threadId, module, writeTasksJson(tasks), cursor, replans, toSqlTs(epochNow()));
    }

    @Override
    public Map<String, Object> getPlan(String threadId) {
        List<Object[]> rows = fetchAll(
                "SELECT module, tasks_json, cursor, replans FROM plan_snapshots WHERE thread_id = ?",
                threadId);
        if (rows.isEmpty()) {
            return null;
        }
        Object[] row = rows.get(0);
        // 载荷与 Python sql_base.get_plan 逐键一致：
        // {thread_id, module, tasks, cursor, replans}（不外露 updated_at）。
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("thread_id", threadId);
        out.put("module", str(row[0]));
        out.put("tasks", parseTasks(row[1]));
        out.put("cursor", toInt(row[2]));
        out.put("replans", toInt(row[3]));
        return out;
    }

    @Override
    public void deletePlan(String threadId) {
        execute("DELETE FROM plan_snapshots WHERE thread_id = ?", threadId);
    }
}

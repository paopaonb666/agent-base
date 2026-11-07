package io.github.paopaonb666.agentbase.memory;

import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.core.llm.LlmFactory;
import io.github.paopaonb666.agentbase.core.llm.ModelRuntime;
import io.github.paopaonb666.agentbase.core.storage.CoreStore;
import io.github.paopaonb666.agentbase.core.storage.CoreStores;
import io.github.paopaonb666.agentbase.core.storage.ThreadIndexRecord;
import io.github.paopaonb666.agentbase.core.graph.Checkpointer;
import io.github.paopaonb666.agentbase.core.graph.CheckpointerFactory;
import io.github.paopaonb666.agentbase.core.graph.GraphState;
import io.github.paopaonb666.agentbase.toolkit.parsing.DocumentParsers;
import io.github.paopaonb666.agentbase.toolkit.parsing.ParseResult;
import io.github.paopaonb666.agentbase.memory.service.MemoryService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * Java 部署形态的运维入口（#278，对应 Python scripts/ 四件套的 thin
 * CLI 版）：{@code java -cp <memory exec jar>} 直接跑，装配走
 * Settings + MemoryService/CoreStores 工厂，语义与 HTTP 端点/服务端
 * 后台路径同源。
 *
 * <p>子命令：
 *
 * <ul>
 *   <li>{@code kb-ingest --user U --path P [--thread T] [--dry-run]}——
 *       批量摄取文档进知识库（pdf/docx/txt/md），fileId = 路径哈希
 *       （同路径重复摄取幂等替换，对齐 watch 语义）；</li>
 *   <li>{@code memory-backfill [--batch N]}——嵌入回填（#164 服务路径）；</li>
 *   <li>{@code memory-capture --user U [--date D] [--dry-run]}——夜间批
 *       记忆形成（force 旁路门控），线程来源 = thread_index 当日活跃；</li>
 *   <li>{@code user-sig --user U --secret S}——记忆身份鉴权请求头生成
 *       （HMAC，防重放时间戳/nonce 与服务端同规）。</li>
 * </ul>
 *
 * <p>退出码：成功（含 0 个待处理项）0；用法错误 2；配置/装配/运行失败 1。
 */
public final class MemoryOpsCli {

    private static final Logger LOG = LoggerFactory.getLogger(MemoryOpsCli.class);

    private static final Set<String> INGEST_EXTENSIONS = Set.of("pdf", "docx", "txt", "md");

    private MemoryOpsCli() {
    }

    public static void main(String[] args) {
        System.exit(run(args));
    }

    /** 可测入口：返回进程退出码。 */
    public static int run(String[] args) {
        if (args.length == 0) {
            usage();
            return 2;
        }
        try {
            return switch (args[0]) {
                case "kb-ingest" -> kbIngest(parseOptions(args));
                case "memory-backfill" -> memoryBackfill(parseOptions(args));
                case "memory-capture" -> memoryCapture(parseOptions(args));
                case "user-sig" -> userSig(parseOptions(args));
                default -> {
                    System.err.println("未知子命令 '" + args[0] + "'");
                    usage();
                    yield 2;
                }
            };
        } catch (UsageError e) {
            System.err.println(e.getMessage());
            usage();
            return 2;
        } catch (Exception e) {
            LOG.error("ops 命令失败", e);
            System.err.println("命令失败：" + e.getMessage());
            return 1;
        }
    }

    private static void usage() {
        System.err.println("""
                用法：ops <子命令> [选项]
                  kb-ingest       --user U --path P [--thread T] [--dry-run]
                  memory-backfill [--batch N]
                  memory-capture  --user U [--date YYYY-MM-DD] [--limit N] [--dry-run]
                  user-sig        --user U --secret S""");
    }

    // ── 选项解析 ──────────────────────────────────────────────────────

    private static final class UsageError extends IllegalArgumentException {
        UsageError(String message) {
            super(message);
        }
    }

    /** 无值旗标（出现即开）。 */
    private static final Set<String> FLAGS = Set.of("dry-run");

    private static Map<String, String> parseOptions(String[] args) {
        Map<String, String> options = new java.util.LinkedHashMap<>();
        for (int i = 1; i < args.length; i++) {
            String arg = args[i];
            if (!arg.startsWith("--")) {
                throw new UsageError("无法识别的参数 '" + arg + "'（选项须以 -- 开头）");
            }
            String key = arg.substring(2);
            if (FLAGS.contains(key)) {
                options.put(key, "1");
                continue;
            }
            if (i + 1 >= args.length) {
                throw new UsageError("选项 --" + key + " 缺少值");
            }
            options.put(key, args[++i]);
        }
        return options;
    }

    private static String require(Map<String, String> options, String key) {
        String value = options.get(key);
        if (value == null || value.isBlank()) {
            throw new UsageError("缺少必需选项 --" + key);
        }
        return value;
    }

    private static int intOption(Map<String, String> options, String key, int fallback) {
        String value = options.get(key);
        if (value == null) {
            return fallback;
        }
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException e) {
            throw new UsageError("选项 --" + key + " 不是整数：" + value);
        }
    }

    // ── kb-ingest ─────────────────────────────────────────────────────

    private static int kbIngest(Map<String, String> options) throws Exception {
        String user = require(options, "user");
        Path root = Path.of(require(options, "path"));
        String thread = options.getOrDefault("thread", "");
        boolean dryRun = options.containsKey("dry-run");
        if (!Files.isDirectory(root)) {
            throw new UsageError("--path 不是目录：" + root);
        }
        Settings settings = Settings.load(Map.of());
        List<Path> files = collectDocuments(root);
        if (dryRun) {
            files.forEach(f -> System.out.println("[dry-run] " + f));
            System.out.println("共 " + files.size() + " 个待摄取文档");
            return 0;
        }
        MemoryService service = MemoryService.build(settings, null, null);
        try {
            if (service == null) {
                System.err.println("MEMORY_ENABLED=false：知识库未启用");
                return 1;
            }
            int ok = 0;
            for (Path file : files) {
                try {
                    byte[] data = Files.readAllBytes(file);
                    ParseResult parsed = DocumentParsers.parseDocument(data,
                            file.getFileName().toString(), "", null,
                            settings.docParse.maxInputBytes, settings.docParse.maxOutputChars);
                    if (parsed.text().strip().isEmpty()) {
                        System.err.println("跳过（无可提取文本）：" + file);
                        continue;
                    }
                    String fileId = pathHashId(root.relativize(file).toString());
                    int chunks = service.ingestDocument(fileId, user, "",
                            parsed.text(), thread);
                    ok++;
                    System.out.println(file + " -> file_id=" + fileId
                            + " chunks=" + chunks
                            + (parsed.truncated() ? "（已截断）" : ""));
                } catch (Exception singleFailure) {
                    // 单文件失败不中断批次（对齐 Python 脚本语义）
                    System.err.println("摄取失败 " + file + "：" + singleFailure.getMessage());
                }
            }
            System.out.println("完成：" + ok + "/" + files.size() + " 个文档已摄取");
            return ok == files.size() ? 0 : 1;
        } finally {
            if (service != null) {
                service.close();
            }
        }
    }

    private static List<Path> collectDocuments(Path root) throws Exception {
        List<Path> files = new ArrayList<>();
        try (var stream = Files.walk(root)) {
            stream.filter(Files::isRegularFile).forEach(file -> {
                String name = file.getFileName().toString().toLowerCase(Locale.ROOT);
                int dot = name.lastIndexOf('.');
                String ext = dot < 0 ? "" : name.substring(dot + 1);
                if (INGEST_EXTENSIONS.contains(ext)) {
                    files.add(file);
                }
            });
        }
        files.sort(Path::compareTo);
        return files;
    }

    /** fileId = 相对路径的 SHA-256 前 20 hex（同路径重复摄取幂等替换）。 */
    static String pathHashId(String relativePath) throws Exception {
        byte[] digest = MessageDigest.getInstance("SHA-256")
                .digest(relativePath.getBytes(java.nio.charset.StandardCharsets.UTF_8));
        StringBuilder hex = new StringBuilder();
        for (byte b : digest) {
            hex.append(String.format("%02x", b));
        }
        return "kb" + hex.substring(0, 20);
    }

    // ── memory-backfill ───────────────────────────────────────────────

    private static int memoryBackfill(Map<String, String> options) throws Exception {
        int batch = intOption(options, "batch", 64);
        Settings settings = Settings.load(Map.of());
        MemoryService service = MemoryService.build(settings, null, null);
        try {
            if (service == null) {
                System.err.println("MEMORY_ENABLED=false：记忆系统未启用");
                return 1;
            }
            if (!service.startBackfill(batch)) {
                System.err.println("回填已在进行中（409 语义）");
                return 1;
            }
            while (true) {
                Map<String, Object> status = service.backfillStatus();
                System.out.print("\rbackfill: " + status);
                String state = String.valueOf(status.get("state"));
                if ("done".equals(state) || "failed".equals(state)
                        || "not_running".equals(state)) {
                    System.out.println();
                    return "done".equals(state) ? 0 : 1;
                }
                Thread.sleep(500);
            }
        } finally {
            if (service != null) {
                service.close();
            }
        }
    }

    // ── memory-capture ────────────────────────────────────────────────

    private static int memoryCapture(Map<String, String> options) throws Exception {
        String user = require(options, "user");
        String date = options.getOrDefault("date", java.time.LocalDate.now().toString());
        int limit = intOption(options, "limit", 200);
        boolean dryRun = options.containsKey("dry-run");
        Settings settings = Settings.load(Map.of());
        MemoryService service = MemoryService.build(settings, null, null);
        try {
            if (service == null) {
                System.err.println("MEMORY_ENABLED=false：记忆系统未启用");
                return 1;
            }
            // 线程来源（与 Python 夜间批同源）：thread_index 中 --date 之后
            // 活跃的线程；消息来源：checkpointer 快照。
            try (CoreStore threadIndex = CoreStores.build(settings);
                 Checkpointer checkpointer = CheckpointerFactory.build(settings)) {
                List<ThreadIndexRecord> threads =
                        threadsActiveSince(threadIndex, user, date, limit);
                System.out.println("活跃线程 " + threads.size() + " 个（自 " + date + "）");
                int failures = 0;
                for (ThreadIndexRecord thread : threads) {
                    GraphState snapshot = checkpointer.get(thread.threadId());
                    if (snapshot == null || snapshot.messages().isEmpty()) {
                        continue;
                    }
                    if (dryRun) {
                        System.out.println("[dry-run] " + thread.threadId()
                                + " 消息 " + snapshot.messages().size() + " 条");
                        continue;
                    }
                    try {
                        Map<String, Object> result = service.captureTurn(user,
                                thread.module(), thread.threadId(),
                                snapshot.messages(), true);
                        System.out.println(thread.threadId() + " -> " + result);
                    } catch (Exception singleFailure) {
                        failures++;
                        System.err.println("形成失败 " + thread.threadId()
                                + "：" + singleFailure.getMessage());
                    }
                }
                return failures == 0 ? 0 : 1;
            }
        } finally {
            if (service != null) {
                service.close();
            }
        }
    }

    /** 线程列表按 updated_at（epoch ms）过滤出 --date 0 点之后活跃的。 */
    private static List<ThreadIndexRecord> threadsActiveSince(CoreStore store, String user,
                                                              String date, int limit) {
        long since = java.time.LocalDate.parse(date)
                .atStartOfDay(java.time.ZoneId.systemDefault()).toInstant().toEpochMilli();
        List<ThreadIndexRecord> out = new ArrayList<>();
        for (ThreadIndexRecord record : store.listThreadIndexes(user, null, limit)) {
            if (record.updatedAt() >= since) {
                out.add(record);
            }
        }
        return out;
    }

    // ── user-sig ──────────────────────────────────────────────────────

    private static int userSig(Map<String, String> options) {
        String user = require(options, "user");
        String secret = require(options, "secret");
        String timestamp = String.valueOf(java.time.Instant.now().getEpochSecond());
        String nonce = java.util.UUID.randomUUID().toString().replace("-", "");
        String signature = io.github.paopaonb666.agentbase.entrypoints.server.AuthBackends
                .HmacHeaderAuth.hmacSha256Hex(secret, user + "\n" + timestamp + "\n" + nonce);
        System.out.println("X-User-Id: " + user);
        System.out.println("X-User-Timestamp: " + timestamp);
        System.out.println("X-User-Nonce: " + nonce);
        System.out.println("X-User-Sig: " + signature);
        return 0;
    }
}

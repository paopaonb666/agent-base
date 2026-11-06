package io.github.paopaonb666.agentbase.entrypoints.server;

import io.github.paopaonb666.agentbase.core.config.Settings;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * 身份作用域（S1 / P0-5）：可插拔的 {@code AuthBackend} + 默认 HMAC 头实现。
 *
 * <p>所有用户作用域端点（记忆 / 文件 / 会话线程）都通过统一入口取得
 * 身份，不再逐个端点手写校验——这是 S1 的结构性修复之一：身份解析只有
 * 一条路径，遗漏只会发生在"忘记声明"，而不是"校验逻辑抄漏了一半"。
 *
 * <p>默认实现 {@code HmacHeaderAuth}：
 * <ul>
 *   <li>{@code X-User-Id} 头缺省为 {@code default}；白名单正则与
 *       X-Request-ID 相同（该值进记忆表与日志，非法值以 400 拒绝而不是
 *       静默替换）。</li>
 *   <li>配置了 {@code MEMORY_AUTH_SECRET} 后必须附带
 *       {@code X-User-Sig = HMAC-SHA256(message, secret)}（hex），否则 401。
 *       防重放（C2）默认开启（300s）：{@code message = user_id + "\n" +
 *       X-User-Timestamp + "\n" + X-User-Nonce}，时间戳必须是 unix 秒且
 *       落在 {@code now ± 窗口} 内；带 nonce 的请求在进程内记录指纹，
 *       窗口期内重复出现即 401（重放拒绝）。窗口设 0 退回仅覆盖 user_id
 *       的旧方案（不推荐）。</li>
 * </ul>
 *
 * <p>已知局限：nonce 重放缓存是<b>单进程</b>口径（与 /metrics 同款），
 * 多实例部署需网关侧防重放或实现自定义 AuthBackend。
 *
 * <p>对应 Python 侧 {@code entrypoints/server/auth.py}，逐行移植。
 */
public final class AuthBackends {

    private AuthBackends() {}

    // 客户端提供的 X-Request-ID / X-User-Id 必须通过此白名单，否则拒绝并
    // 另发新 id：该值会被回显到响应头并写进每条日志，放行任意字符串等于
    // 允许伪造日志行 / 触发非法响应头。
    public static final Pattern REQUEST_ID_RE = Pattern.compile("[A-Za-z0-9._-]{1,64}");

    /** 从请求解析用户作用域身份；认证失败抛 {@link HttpError}（401/400）。 */
    public interface AuthBackend {

        /** @param headers 请求头视图（大小写不敏感由调用方保证） */
        String authenticate(String userIdHeader, String sigHeader, String timestampHeader,
                            String nonceHeader, Settings settings);
    }

    /**
     * 无鉴权部署判定（#196/旧 #224）：MEMORY_AUTH_SECRET 为空时 HMAC
     * 整层跳过，所有请求都落 default 用户、共享同一份记忆/画像/知识库
     * ——本机开发无感，端口一旦暴露即隐私事故。production 已由
     * ensureProductionReady 强制 secret；development 启动时必须以醒目
     * 横幅提示（组合根调用，这里只给判定，方便测试锁定条件）。
     */
    public static boolean unauthenticatedDeployment(Settings settings) {
        return settings.memory.enabled && settings.memory.authSecret.strip().isEmpty();
    }

    /** 无鉴权部署的启动横幅文案（醒目、可执行）。 */
    public static String unauthenticatedBanner() {
        return """
                ----------------------------------------------------------------------
                ⚠ 鉴权未配置（MEMORY_AUTH_SECRET 为空）：所有请求都按 default 用户
                  处理，共享同一份记忆/画像/知识库。本机开发无碍；端口一旦对外
                  暴露即隐私事故。生产部署必须设置 MEMORY_AUTH_SECRET（客户端
                  需带 X-User-Sig = HMAC-SHA256(user_id\\ntimestamp\\nnonce, secret)）。
                ----------------------------------------------------------------------""";
    }

    /** 默认实现：X-User-Id + X-User-Sig HMAC 校验（含时间戳/nonce 防重放）。 */
    public static final class HmacHeaderAuth implements AuthBackend {

        private final int replayCacheLimit;
        // nonce 重放缓存按实例持有（每应用构造一个后端实例，生命周期与
        // 进程一致）；LRU 上限防止长期运行下的无界增长。
        private final Map<String, Double> seenNonces = new LinkedHashMap<>();
        private final Object lock = new Object();

        public HmacHeaderAuth() {
            this(4096);
        }

        public HmacHeaderAuth(int replayCacheLimit) {
            this.replayCacheLimit = replayCacheLimit;
        }

        @Override
        public String authenticate(String userIdHeader, String sigHeader, String timestampHeader,
                                   String nonceHeader, Settings settings) {
            String userId;
            if (userIdHeader == null || userIdHeader.isEmpty()) {
                userId = "default";
            } else if (!REQUEST_ID_RE.matcher(userIdHeader).matches()) {
                throw new HttpError(400,
                        "X-User-Id 非法：仅允许字母/数字/点/下划线/连字符，1-64 字符");
            } else {
                userId = userIdHeader;
            }
            String secret = settings.memory.authSecret.strip();
            if (!secret.isEmpty()) {
                long window = settings.memory.authReplayWindowSeconds;
                String timestamp = "";
                String nonce = "";
                if (window > 0) {
                    timestamp = timestampHeader == null ? "" : timestampHeader;
                    if (timestamp.isEmpty()) {
                        throw new HttpError(401,
                                "缺少 X-User-Timestamp：该部署已启用防重放时间窗");
                    }
                    long ts;
                    try {
                        ts = Long.parseLong(timestamp.trim());
                    } catch (NumberFormatException e) {
                        throw new HttpError(401, "X-User-Timestamp 必须是 unix 秒（整数）");
                    }
                    if (Math.abs(System.currentTimeMillis() / 1000.0 - ts) > window) {
                        throw new HttpError(401,
                                "X-User-Timestamp 超出容忍窗口：请求已过期或时钟漂移过大");
                    }
                    String rawNonce = nonceHeader == null ? "" : nonceHeader;
                    if (!rawNonce.isEmpty()) {
                        if (!REQUEST_ID_RE.matcher(rawNonce).matches()) {
                            throw new HttpError(400,
                                    "X-User-Nonce 非法：仅允许字母/数字/点/下划线/连字符，1-64 字符");
                        }
                        nonce = rawNonce;
                    }
                    String message = userId + "\n" + timestamp + "\n" + nonce;
                    verifySignature(sigHeader, secret, message);
                    if (!nonce.isEmpty()) {
                        // 通过了签名校验才记账/查账：错签名的 nonce 不占用缓存。
                        rejectReplay(nonce, ts + 2.0 * window);
                    }
                } else {
                    verifySignature(sigHeader, secret, userId);
                }
            }
            return userId;
        }

        private static void verifySignature(String sigHeader, String secret, String message) {
            String sig = sigHeader == null ? "" : sigHeader;
            String expected = hmacSha256Hex(secret, message);
            // 常量时间比较（对应 hmac.compare_digest 的防时序语义）。
            if (!java.security.MessageDigest.isEqual(
                    sig.getBytes(StandardCharsets.UTF_8), expected.getBytes(StandardCharsets.UTF_8))) {
                throw new HttpError(401, "X-User-Sig 缺失或不正确：该部署已启用记忆身份鉴权");
            }
        }

        /** 窗口期内的 nonce 只允许出现一次；过期指纹视作新请求。 */
        private void rejectReplay(String nonce, double expiresAt) {
            synchronized (lock) {
                Double seenUntil = seenNonces.get(nonce);
                if (seenUntil != null && seenUntil > System.currentTimeMillis() / 1000.0) {
                    throw new HttpError(401, "X-User-Nonce 已使用：重放被拒绝");
                }
                seenNonces.remove(nonce);
                seenNonces.put(nonce, expiresAt);
                while (seenNonces.size() > replayCacheLimit) {
                    String eldest = seenNonces.keySet().iterator().next();
                    seenNonces.remove(eldest);
                }
            }
        }

        /** HMAC-SHA256 hex 小写（与 Python hmac.new(...).hexdigest() 一致）。 */
        public static String hmacSha256Hex(String secret, String message) {
            try {
                Mac mac = Mac.getInstance("HmacSHA256");
                mac.init(new SecretKeySpec(secret.getBytes(StandardCharsets.UTF_8), "HmacSHA256"));
                byte[] digest = mac.doFinal(message.getBytes(StandardCharsets.UTF_8));
                StringBuilder sb = new StringBuilder(digest.length * 2);
                for (byte b : digest) {
                    sb.append(String.format("%02x", b));
                }
                return sb.toString();
            } catch (Exception e) {
                throw new IllegalStateException("HMAC-SHA256 不可用", e);
            }
        }
    }
}

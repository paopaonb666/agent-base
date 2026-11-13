package io.github.paopaonb666.agentbase.core.storage;

import java.util.concurrent.ConcurrentHashMap;

import com.zaxxer.hikari.HikariDataSource;

/**
 * MySQL 连接池单例注册表（#161/旧 #97）：对话状态 checkpointer、基座
 * core store、记忆 store、工具审计日志、成本账本、上传文件 store 六处
 * MySQL 访问共享同一批池——此前四处各自 DriverManager 短连接，高并发下
 * 连接风暴 + max_connections 打爆。池按 (url, user, password) 三元组键
 * 复用（A-5/审查 P2-5：密码在键内——同库同账号不同密码的调用方不会误拿
 * 先建池；此前 javadoc 声称密码在键内而实现缺该维度）；池体懒启动（首次
 * getConnection 才建连），进程退出钩子统一关闭。
 */
public final class MySqlPools {

    private static final ConcurrentHashMap<String, HikariDataSource> POOLS = new ConcurrentHashMap<>();

    static {
        Runtime.getRuntime().addShutdownHook(new Thread(() ->
                POOLS.values().forEach(HikariDataSource::close), "mysql-pools-close"));
    }

    private MySqlPools() {}

    /**
     * 池键（A-5/审查 P2-5 抽出可测）：url|user|password 三元组。密码在键
     * 内，同库同账号不同密码互不串池（正确密码的调用方永远拿到自己配置
     * 建的池）。
     */
    static String poolKey(String url, String user, String password) {
        return url + "|" + user + "|" + password;
    }

    /**
     * 规范 JDBC URL 单源（A-5/审查 P2-5）：连接参数串（utf8 编码 +
     * allowPublicKeyRetrieval + 关 SSL）与 checkpointer/core store/记忆
     * store 的既有串逐字节一致。此前 toollog/成本账本/上传文件 store 拼
     * 裸 URL（无 query 参数）：同库同账号也因 URL 串不同 split 出第二个
     * Hikari 池；且裸 URL 在 MySQL 8 默认认证（caching_sha2_password）下
     * 缺 allowPublicKeyRetrieval 会直接建连失败。六处调用方一律经本方法
     * 拼装，杜绝再次漂移。
     */
    public static String jdbcUrl(String host, int port, String database) {
        return ("jdbc:mysql://%s:%d/%s?useUnicode=true&characterEncoding=utf8"
                + "&allowPublicKeyRetrieval=true&useSSL=false").formatted(host, port, database);
    }

    /** 取（或建）共享池；连接由调用方用完即还（try-with-resources 语义）。 */
    public static javax.sql.DataSource pool(String url, String user, String password) {
        return POOLS.computeIfAbsent(poolKey(url, user, password), k -> {
            // 无参构造 + setter：池懒启动（构造期不打连接），配置错误在
            // 首次 getConnection 才暴露——与原 DriverManager 语义等价。
            HikariDataSource ds = new HikariDataSource();
            ds.setJdbcUrl(url);
            ds.setUsername(user);
            ds.setPassword(password);
            ds.setMaximumPoolSize(8);
            ds.setMinimumIdle(1);
            ds.setPoolName("agent-base-mysql-" + POOLS.size());
            return ds;
        });
    }
}

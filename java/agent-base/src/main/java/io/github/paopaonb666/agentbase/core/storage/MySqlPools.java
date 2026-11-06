package io.github.paopaonb666.agentbase.core.storage;

import java.util.concurrent.ConcurrentHashMap;

import com.zaxxer.hikari.HikariDataSource;

/**
 * MySQL 连接池单例注册表（#161/旧 #97）：对话状态 checkpointer、基座
 * core store、记忆 store、工具审计日志四处 MySQL 访问共享同一批池——
 * 此前四处各自 DriverManager 短连接，高并发下连接风暴 + max_connections
 * 打爆。池按 (url, user) 键复用（同库不同密码视为不同池）；池体懒启动
 * （首次 getConnection 才建连），进程退出钩子统一关闭。
 */
public final class MySqlPools {

    private static final ConcurrentHashMap<String, HikariDataSource> POOLS = new ConcurrentHashMap<>();

    static {
        Runtime.getRuntime().addShutdownHook(new Thread(() ->
                POOLS.values().forEach(HikariDataSource::close), "mysql-pools-close"));
    }

    private MySqlPools() {}

    /** 取（或建）共享池；连接由调用方用完即还（try-with-resources 语义）。 */
    public static javax.sql.DataSource pool(String url, String user, String password) {
        String key = url + "|" + user;
        return POOLS.computeIfAbsent(key, k -> {
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

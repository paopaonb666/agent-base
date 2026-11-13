package io.github.paopaonb666.agentbase.core.storage;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;

/**
 * MySqlPools 池键与 URL 单源测试（A-5/审查 P2-5）：密码在池键内（同库同
 * 账号不同密码互不串池——此前 javadoc 声称如此而键实际缺该维度）；jdbcUrl
 * 规范串与 checkpointer/core store/记忆 store 的既有内联串逐字节一致（六处
 * 调用方单源，杜绝 URL 漂移 split 池）。池实例惰性启动（不建连），无需
 * MySQL 活体。
 */
class MySqlPoolsTest {

    @Test
    void poolKeyIncludesPasswordDimension() {
        String url = "jdbc:mysql://h:3306/db?useUnicode=true&characterEncoding=utf8"
                + "&allowPublicKeyRetrieval=true&useSSL=false";
        assertEquals(MySqlPools.poolKey(url, "app", "pw1"), MySqlPools.poolKey(url, "app", "pw1"),
                "同三元组同键");
        assertNotEquals(MySqlPools.poolKey(url, "app", "pw1"), MySqlPools.poolKey(url, "app", "pw2"),
                "同 url+user 异密码不同键——修复前键缺密码维度，后建池调用方会误拿先建池");
        assertNotEquals(MySqlPools.poolKey(url, "app", "pw"), MySqlPools.poolKey(url, "other", "pw"),
                "同 url 异 user 不同键");
    }

    @Test
    void poolReusesInstanceForSameTripleAndSplitsOnPassword() {
        String url = MySqlPools.jdbcUrl("h", 3306, "db");
        javax.sql.DataSource first = MySqlPools.pool(url, "app", "pw1");
        assertSame(first, MySqlPools.pool(url, "app", "pw1"), "同三元组复用同一池");
        assertNotSame(first, MySqlPools.pool(url, "app", "pw2"), "异密码是新池");
    }

    @Test
    void jdbcUrlIsTheCanonicalParameterizedForm() {
        // 规范串与 checkpointer/core store/记忆 store 的既有内联串逐字节一致
        // （迁移零行为变化）；四个连接参数齐全——裸 URL 在 MySQL 8 默认认证
        // （caching_sha2_password）下缺 allowPublicKeyRetrieval 会建连失败。
        assertEquals("jdbc:mysql://h:3306/db?useUnicode=true&characterEncoding=utf8"
                        + "&allowPublicKeyRetrieval=true&useSSL=false",
                MySqlPools.jdbcUrl("h", 3306, "db"));
    }
}

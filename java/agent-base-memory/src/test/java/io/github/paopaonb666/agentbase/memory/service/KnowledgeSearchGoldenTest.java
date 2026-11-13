package io.github.paopaonb666.agentbase.memory.service;

import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.memory.embeddings.EmbeddingClient;
import io.github.paopaonb666.agentbase.memory.store.InMemoryMemoryStore;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * #96 场景 eval · RAG 线：ingest→searchKnowledge 打分 golden——文档经真实
 * 摄取管道（TextKit 分块→落库）后，场景查询的 top-1 命中预期文档。离线
 * 确定性（embedding 缺席走降级关键词路径，与生产降级面同规则）；门=
 * top-1 命中率 ≥ {@link #TOP1_GATE}。为 #31/#52 的开启态量化提供地基
 * （web_search 侧语料同型：词面锚点唯一的短文档集）。
 */
class KnowledgeSearchGoldenTest {

    /** top-1 命中率门（6 文档 ≥ 0.85 → 实际要求 6/6）。 */
    private static final double TOP1_GATE = 0.85;

    /** golden 语料：{file_id, 文档正文（<800 字符=单分块）, golden 查询}。 */
    private static final String[][] GOLDEN = {
            {"f-k8s", "集群版本升级流程：先把节点池逐个排水，再替换节点镜像到新版本，最后逐个回注负载。升级窗口必须提前一天在值班群公告。",
                    "集群升级的步骤是什么"},
            {"f-cache", "缓存层用 Redis 集群承载热点数据，命中率跌破九成时先检查大 key 与过期策略，再评估扩容分片。",
                    "缓存命中率下降怎么排查"},
            {"f-auth", "鉴权中心统一签发 JWT，访问令牌有效期两小时，刷新令牌十四天轮换；所有服务通过内网端点校验签名。",
                    "访问令牌多久过期"},
            {"f-log", "日志规范：应用日志输出 JSON 格式，必带 request_id 字段；错误日志额外携带 trace_id 便于跨服务串联排查。",
                    "日志格式有什么要求"},
            {"f-db", "数据库连接池上限按实例规格配置，生产实例 8 核配置上限 60 连接；慢查询超过一秒必须建索引或改写。",
                    "连接池上限是多少"},
            {"f-deploy", "发布采用灰度策略：新版本先接 5% 流量观察两小时，核心指标无异常再逐步放大到全量。",
                    "灰度发布的流量比例"},
    };

    @Test
    void ingestThenSearchKnowledgeHitsExpectedFileTop1() {
        Settings settings = Settings.load(java.util.Map.of("ENV", "development"), null);
        MemoryService service = new MemoryService(new InMemoryMemoryStore(), neverEmbedder(),
                settings, null, null);

        for (String[] entry : GOLDEN) {
            int chunks = service.ingestDocument(entry[0], "eval-user", "chat", entry[1], null);
            assertEquals(1, chunks, "语料契约：每篇文档单分块（<800 字符）：" + entry[0]);
        }

        int hits = 0;
        StringBuilder misses = new StringBuilder();
        for (String[] entry : GOLDEN) {
            List<Map<String, Object>> ranked =
                    service.searchKnowledge("eval-user", "chat", entry[2], 3);
            if (!ranked.isEmpty() && entry[0].equals(ranked.get(0).get("file_id"))) {
                hits++;
            } else {
                misses.append(entry[0]).append(" ← 「").append(entry[2]).append("」 top1=")
                        .append(ranked.isEmpty() ? "空" : ranked.get(0).get("file_id"))
                        .append("; ");
            }
        }
        double top1 = hits / (double) GOLDEN.length;
        assertTrue(top1 >= TOP1_GATE,
                "RAG 场景 top-1 = %.3f < 门限 %.2f；未命中：%s".formatted(top1, TOP1_GATE, misses));
        // 结果形状契约：file_id/text/score 三键在场（消费端锚点）。
        List<Map<String, Object>> first =
                service.searchKnowledge("eval-user", "chat", GOLDEN[0][2], 1);
        assertEquals(1, first.size());
        assertTrue(first.get(0).containsKey("file_id") && first.get(0).containsKey("text")
                && first.get(0).containsKey("score"));
    }

    private static EmbeddingClient neverEmbedder() {
        return new EmbeddingClient() {
            @Override public Integer dims() {
                return null;
            }

            @Override public List<List<Float>> embed(List<String> texts) {
                return null;
            }

            @Override public boolean probe() {
                return false;
            }
        };
    }
}

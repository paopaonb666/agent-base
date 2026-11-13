package io.github.paopaonb666.agentbase.memory.service;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.UserMessage;
import io.github.paopaonb666.agentbase.core.AgentModule;
import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.core.llm.ModelRuntime;
import io.github.paopaonb666.agentbase.memory.embeddings.EmbeddingClient;
import io.github.paopaonb666.agentbase.memory.retrieval.Retrieval;
import io.github.paopaonb666.agentbase.memory.store.InMemoryMemoryStore;
import org.junit.jupiter.api.Test;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * #96 场景 eval · 记忆线：多轮 capture（脚本化抽取→整合落库）→ 全语料
 * 召回的场景级 golden——与 #44 RetrievalGoldenEvalTest 的差别是候选集经
 * <b>真实 capture 管线</b>形成（抽取脚本→整合 ADD→落库），链路含
 * extract/consolidate/profile 三步的编排与归一，不只打分函数。
 *
 * <p>离线确定性：LLM 全脚本化（零网络），embedding 缺席走降级关键词路径
 * （与生产降级面同规则）；门=recall@3 ≥ {@link #RECALL_TOP3_GATE}（misses
 * 随断言消息列出）。为 #46/#43 的开启态量化提供地基。
 */
class MemoryScenarioEvalTest {

    /** 场景级召回门：recall@3 下限（8 场景 ≥ 0.85 → 至少 7 中）。 */
    private static final double RECALL_TOP3_GATE = 0.85;

    /** 脚本化主力（整合裁决）：弹出预置回复。 */
    private static final class ScriptedMain implements ModelRuntime {
        final Deque<String> responses = new ArrayDeque<>();

        @Override
        public dev.langchain4j.model.chat.ChatModel chat() {
            throw new UnsupportedOperationException();
        }

        @Override
        public dev.langchain4j.model.chat.StreamingChatModel streaming() {
            return null;
        }

        @Override
        public AiMessage stream(List<ChatMessage> messages,
                                List<dev.langchain4j.agent.tool.ToolSpecification> tools,
                                java.util.function.Consumer<String> onDelta) {
            throw new UnsupportedOperationException();
        }

        @Override
        public AiMessage invoke(List<ChatMessage> messages) {
            String next = responses.poll();
            if (next == null) {
                throw new IllegalStateException("scripted main exhausted");
            }
            return AiMessage.from(next);
        }
    }

    /** 脚本化快档（抽取/画像）。 */
    private static final class ScriptedFast implements AgentModule.ChatInvokerPort {
        final Deque<String> responses = new ArrayDeque<>();

        @Override
        public String invoke(List<ChatMessage> messages) {
            String next = responses.poll();
            if (next == null) {
                throw new IllegalStateException("scripted fast exhausted");
            }
            return next;
        }
    }

    /** 场景：两轮对话转写（+各轮脚本化抽取输出）与召回 golden。 */
    private record Scenario(String name, String[] transcripts, String[] extractions,
                            String query, String expectedContent) {
    }

    /**
     * 8 个场景（内容/查询带唯一词面锚点——降级关键词路径靠词面与 bigram）。
     * 抽取输出即该轮应落库的记忆（每轮 1 条，整合脚本恒 ADD）。
     */
    private static final Scenario[] SCENARIOS = {
            new Scenario("部署环境",
                    new String[]{"用户：我们把生产库迁到了 PostgreSQL 16，服务器是 Ubuntu 22.04"},
                    new String[]{"[{\"content\": \"生产数据库是 PostgreSQL 16，运行在 Ubuntu 22.04 服务器\", \"kind\": \"semantic\", \"salience\": 0.8}]"},
                    "生产数据库和服务器操作系统是什么",
                    "生产数据库是 PostgreSQL 16，运行在 Ubuntu 22.04 服务器"),
            new Scenario("备份节奏",
                    new String[]{"用户：数据库每天凌晨三点做全量备份，备份传到对象存储"},
                    new String[]{"[{\"content\": \"数据库备份每天凌晨三点全量执行并上传对象存储\", \"kind\": \"semantic\", \"salience\": 0.7}]"},
                    "数据库备份什么时候跑",
                    "数据库备份每天凌晨三点全量执行并上传对象存储"),
            new Scenario("告警去向",
                    new String[]{"用户：线上告警统一发到值班飞书群，严重级别才打电话"},
                    new String[]{"[{\"content\": \"线上告警发送到值班飞书群，严重级别电话通知\", \"kind\": \"semantic\", \"salience\": 0.7}]"},
                    "告警通知发到哪里",
                    "线上告警发送到值班飞书群，严重级别电话通知"),
            new Scenario("评审门槛",
                    new String[]{"用户：我们规定合并代码至少要一个人 approve 才行"},
                    new String[]{"[{\"content\": \"合并代码要求至少一人 approve 才能通过\", \"kind\": \"semantic\", \"salience\": 0.7}]"},
                    "代码合并需要几个人批准",
                    "合并代码要求至少一人 approve 才能通过"),
            new Scenario("编辑器偏好",
                    new String[]{"用户：我写代码一直用 JetBrains 的 IDE，终端配的是 fish"},
                    new String[]{"[{\"content\": \"用户偏好的编辑器是 JetBrains IDE，终端用 fish shell\", \"kind\": \"preference\", \"salience\": 0.6}]"},
                    "用户用什么编辑器和终端",
                    "用户偏好的编辑器是 JetBrains IDE，终端用 fish shell"),
            new Scenario("发布窗口",
                    new String[]{"用户：发版只允许在周二和周四的上午进行"},
                    new String[]{"[{\"content\": \"发布窗口固定在周二与周四上午\", \"kind\": \"semantic\", \"salience\": 0.7}]"},
                    "什么时间可以发版",
                    "发布窗口固定在周二与周四上午"),
            new Scenario("对外开放端口",
                    new String[]{"用户：生产网段现在只有 443 端口是对外开放的"},
                    new String[]{"[{\"content\": \"生产网段仅对外开放 443 端口\", \"kind\": \"semantic\", \"salience\": 0.7}]"},
                    "对外开放了哪些端口",
                    "生产网段仅对外开放 443 端口"),
            new Scenario("令牌有效期",
                    new String[]{"用户：内部系统统一用 JWT 鉴权，令牌有效期两小时"},
                    new String[]{"[{\"content\": \"内部系统用 JWT 鉴权，令牌有效期两小时\", \"kind\": \"semantic\", \"salience\": 0.7}]"},
                    "鉴权令牌的有效期是多久",
                    "内部系统用 JWT 鉴权，令牌有效期两小时"),
    };

    @Test
    void captureChainThenRecallHitsExpectedMemoryInTop3() {
        Settings settings = Settings.load(java.util.Map.of("ENV", "development"), null);
        InMemoryMemoryStore store = new InMemoryMemoryStore();
        ScriptedMain main = new ScriptedMain();
        ScriptedFast fast = new ScriptedFast();
        MemoryService service = new MemoryService(store, neverEmbedder(), settings, main, fast);

        // 走真实 capture 管线把 8 个场景的记忆落进同一语料。
        for (Scenario scenario : SCENARIOS) {
            for (int turn = 0; turn < scenario.transcripts().length; turn++) {
                fast.responses.add(scenario.extractions()[turn]);
                main.responses.add("{\"op\": \"ADD\", \"content\": \""
                        + scenario.expectedContent() + "\"}");
                fast.responses.add("{\"基本信息\": []}");
                service.captureTurn("eval-user", "chat", "thread:" + scenario.name(),
                        List.of(dev.langchain4j.data.message.UserMessage
                                .from(scenario.transcripts()[turn])), false);
            }
        }
        assertEquals(SCENARIOS.length, service.listMemories("eval-user", "chat", false,
                null, null, 100, true).size(), "capture 链路应把 8 条场景记忆全部落库");

        // 全语料（8 条）上逐场景召回：期望记忆进 top-3。
        int hits = 0;
        List<String> misses = new ArrayList<>();
        for (Scenario scenario : SCENARIOS) {
            List<Retrieval.ScoredMemory> ranked =
                    service.searchScored("eval-user", "chat", scenario.query(), 3);
            boolean hit = ranked.stream()
                    .anyMatch(item -> item.record().content().equals(scenario.expectedContent()));
            if (hit) {
                hits++;
            } else {
                misses.add(scenario.name() + " ← 「" + scenario.query() + "」 top1="
                        + (ranked.isEmpty() ? "空" : ranked.get(0).record().content()));
            }
        }
        double recallAt3 = hits / (double) SCENARIOS.length;
        assertTrue(recallAt3 >= RECALL_TOP3_GATE,
                "场景级 recall@3 = %.3f < 门限 %.2f；未命中：%s"
                        .formatted(recallAt3, RECALL_TOP3_GATE, misses));
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

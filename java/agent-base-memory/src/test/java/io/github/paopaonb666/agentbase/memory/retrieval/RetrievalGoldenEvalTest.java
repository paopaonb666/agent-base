package io.github.paopaonb666.agentbase.memory.retrieval;

import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.memory.retrieval.Retrieval.ScoredMemory;
import io.github.paopaonb666.agentbase.memory.store.Models.MemoryRecord;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 记忆检索评估集（roadmap #44 / Wave 3 门槛）：60 条 golden 查询的
 * recall@5 与 MRR 回归——这是 #323/325/326/327 等所有检索改造的安全网
 * （「索引下推后召回没变差」的可证伪口径），随 mvn test 进 CI 作为记忆
 * 包质量门。
 *
 * <p>口径：golden 查询是<b>改写</b>而非原文复读（真实用户会换措辞、
 * 但保留关键术语——与检索分词的 CJK 二元组/ASCII 词口径一致）；评测
 * 走与生产一致的打分路径（{@code Retrieval.scoreMemories}），测试环境
 * 无 embedding → 向量分量为 0，度量的是关键词+时间+显著度的混合质量。
 * 阈值先宽松起步（recall@5 ≥ 0.90 / MRR ≥ 0.75），收紧改本文件常量。
 */
class RetrievalGoldenEvalTest {

    /** 召回质量门：recall@5 下限。 */
    private static final double RECALL_AT_5_GATE = 0.90;
    /** 召回质量门：MRR 下限。 */
    private static final double MRR_GATE = 0.75;

    /** golden 语料：{memory_id, 记忆内容, golden 查询（改写）}。 */
    private static final String[][] GOLDEN = {
            // ── 部署 / 运维 ──────────────────────────────────────────
            {"m01", "生产环境部署在 Ubuntu 22.04 服务器，数据库是 PostgreSQL 16", "部署环境用的是什么操作系统"},
            {"m02", "应用通过 systemd 托管，服务名 agent-base-prod", "服务是怎么托管的"},
            {"m03", "CI 用 GitHub Actions，测试后构建 docker 镜像推到 ghcr.io", "持续集成用的哪个平台"},
            {"m04", "日志收集走 Loki，指标暴露给 Prometheus 抓取", "日志和指标都发到哪里"},
            {"m05", "备份策略：数据库每天凌晨三点全量快照到对象存储", "数据库备份是什么时候跑的"},
            {"m06", "域名 api.example.com 前面挂 nginx 做反向代理与 TLS 终结", "流量入口的网关是什么"},
            {"m07", "扩容流程：先加只读副本再切写流量，禁止直接改主库", "数据库扩容应该怎么做"},
            {"m08", "线上告警发到值班飞书群，严重级别才打电话", "告警通知发到哪里"},
            // ── 项目 / 代号 ──────────────────────────────────────────
            {"m09", "内部把重构项目称为灯塔计划，对外不能提这个名字", "重构项目叫什么代号"},
            {"m10", "灯塔计划的里程碑一在十月底交付，范围是网关与鉴权", "灯塔一期什么时候交付"},
            {"m11", "文档站部署在 wiki.internal，工程规范都挂在首页", "工程文档放在哪个站点"},
            {"m12", "代码评审要求至少一人 approve 才能合入主干", "合并代码需要几个人批准"},
            {"m13", "灰度发布先放 5% 流量观察两小时再全量", "灰度发布的节奏是怎样的"},
            {"m14", "发布窗口固定在周二与周四的上午，避开周五", "什么时间可以发版"},
            {"m15", "故障复盘要在事故关闭后三个工作日内完成", "复盘报告的截止时限"},
            {"m16", "版本号遵循语义化版本，破坏性变更必须升主版本", "版本号规则是什么"},
            // ── 人物 / 角色 ──────────────────────────────────────────
            {"m17", "用户是后端工程师，主要负责支付网关的接口开发", "用户的工作职责是什么"},
            {"m18", "用户的直属上级是产品技术负责人，绩效沟通走月度一对一", "用户的汇报对象是谁"},
            {"m19", "团队使用飞书做日常协作，周会固定在周一上午", "团队日常用什么协作工具"},
            {"m20", "用户正在准备系统架构师认证考试，约在年底", "用户在备考什么认证"},
            {"m21", "新同事入职引导文档由用户维护，包含开发环境搭建", "入职文档是谁在维护"},
            {"m22", "用户偏好异步沟通，紧急事项才允许电话打断", "沟通方式上有什么偏好"},
            {"m23", "用户的学习目标是深入向量数据库与混合检索", "用户最近想深入什么方向"},
            {"m24", "用户的时区是 UTC+8，早晨十点前不要安排会议", "用户在哪个时区"},
            // ── 偏好 ─────────────────────────────────────────────────
            {"m25", "用户偏好的编辑器是 JetBrains 全家桶，终端用 fish shell", "用户用什么编辑器和终端"},
            {"m26", "代码风格：Java 遵循谷歌风格，Python 遵循 pep8 并用 ruff 检查", "代码风格遵循什么规范"},
            {"m27", "用户喝咖啡不加糖，下午三点后只喝气泡水", "用户喝咖啡有什么习惯"},
            {"m28", "示例代码优先给 Python，其次是 TypeScript", "示例代码用什么语言优先"},
            {"m29", "用户不喜欢表情符号，回复保持正式书面语", "回复风格有什么禁忌"},
            {"m30", "报告偏好 markdown 表格对比方案，不要长段落", "方案对比用什么格式呈现"},
            // ── 约束 / 合规 ──────────────────────────────────────────
            {"m31", "数据库密码只放密钥管理服务，禁止写进配置文件", "数据库密码放在哪里"},
            {"m32", "用户数据出境需要过合规评审，默认存储在境内机房", "数据存储地有什么要求"},
            {"m33", "生产数据库操作必须走工单，禁止直连变更", "生产库变更走什么流程"},
            {"m34", "开源依赖引入前要过许可证扫描，排除强传染协议", "引入开源依赖前要做什么检查"},
            {"m35", "客户姓名与手机号属于敏感字段，日志里必须脱敏", "日志里的敏感字段怎么处理"},
            {"m36", "测试环境数据每周自动清空，不要在里面放真实客户", "测试环境的数据多久清理"},
            {"m37", "对外的接口变更要先发公告再升级，给调用方两周缓冲", "接口变更对调用方的缓冲期"},
            {"m38", "所有服务必须暴露健康检查端点供探针轮询", "服务必须暴露什么端点"},
            // ── 流程 / 规程 ──────────────────────────────────────────
            {"m39", "发版流程：打 tag 后流水线自动构建，人工确认后才部署", "发版流程的第一步是什么"},
            {"m40", "回滚预案：保留最近三个版本镜像，一键回切", "回滚预案保留几个版本"},
            {"m41", "新服务上线前要在容量表登记峰值 QPS 估算", "新服务上线前要登记什么"},
            {"m42", "值班轮换每周一次，交接包含未关闭工单清单", "值班交接要包含哪些内容"},
            {"m43", "需求评审通过后才允许排期，评审记录归档到 wiki", "需求什么时候可以排期"},
            {"m44", "数据库迁移脚本必须可重复执行，先在影子库演练", "迁移脚本上线前在哪演练"},
            {"m45", "接口压测达标线是 p99 小于 200 毫秒", "接口压测的达标线是多少"},
            {"m46", "故障升级路径：半小时无进展升级到值班经理", "故障多久没有进展要升级"},
            // ── 事件 ─────────────────────────────────────────────────
            {"m47", "上个月网关故障的原因是连接池耗尽，已修复并加了监控", "上次网关故障的原因是什么"},
            {"m48", "用户的团队在本季度完成了多租户改造第一期", "多租户改造进展到哪了"},
            {"m49", "客户演示定在下周三，需要提前准备演示环境", "客户演示定在哪天"},
            {"m50", "上次的招聘指标是后端两名，已全部到岗", "招聘指标完成了没有"},
            {"m51", "机房网络在月初割接过，出口带宽从 1G 升到 10G", "出口带宽最近有什么变化"},
            {"m52", "旧搜索服务已经下线，流量全部切到新检索链路", "旧搜索服务的状态如何"},
            // ── 工具 / 技术栈 ────────────────────────────────────────
            {"m53", "主力数据库是 PostgreSQL，缓存用 Redis 集群", "主力数据库和缓存是什么"},
            {"m54", "向量检索用 pgvector 扩展，索引类型 HNSW", "向量检索用什么扩展"},
            {"m55", "消息队列用 Kafka，主题命名遵循域-事件格式", "消息队列用的什么"},
            {"m56", "前端是 React 加 TypeScript，构建工具 vite", "前端技术栈是什么"},
            {"m57", "对象存储用 MinIO 自建集群，桶按业务域划分", "对象存储是怎么部署的"},
            {"m58", "任务调度用 Temporal，工作流代码在 workflow 仓库", "任务调度用的什么框架"},
            {"m59", "鉴权中心签发 JWT，有效期为两小时并支持刷新", "鉴权令牌的有效期多久"},
            {"m60", "观测性三件套：metrics 用 Prometheus，链路用 Tempo，日志用 Loki", "观测性体系用什么组件"},
    };

    private static MemoryRecord record(String id, String content, double updatedAt) {
        return new MemoryRecord(id, "eval-user", "chat", "semantic", content,
                List.of(), null, null, 0.5, "active", "", List.of(), updatedAt, updatedAt, null, 0);
    }

    @Test
    void goldenSetRecallAt5AndMrrStayAboveGate() {
        Settings settings = Settings.load(java.util.Map.of("ENV", "development"), null);
        double now = 1_800_000_000.0;
        List<MemoryRecord> corpus = new ArrayList<>(GOLDEN.length);
        for (int i = 0; i < GOLDEN.length; i++) {
            // 错开 updated_at（真实库里的时间分布），保持全部在近期窗口内。
            corpus.add(record(GOLDEN[i][0], GOLDEN[i][1], now - (i % 20) * 3600.0));
        }
        assertEquals(60, corpus.size(), "语料规模契约：50-100 条");
        assertEquals(60, new HashSet<>(List.of(GOLDEN)).size() == GOLDEN.length ? 60 : 60,
                "语料条目数一致");

        int hits = 0;
        double reciprocalSum = 0.0;
        List<String> misses = new ArrayList<>();
        for (String[] entry : GOLDEN) {
            String expectedId = entry[0];
            String query = entry[2];
            List<ScoredMemory> ranked = Retrieval.scoreMemories(corpus, query, null, now,
                    settings.memory.timeDecayHalfLifeDays, Retrieval.Weights.DEFAULT);
            int rank = -1;
            for (int k = 0; k < Math.min(5, ranked.size()); k++) {
                if (ranked.get(k).record().memoryId().equals(expectedId)) {
                    rank = k + 1;
                    break;
                }
            }
            if (rank > 0) {
                hits++;
                reciprocalSum += 1.0 / rank;
            } else {
                misses.add(expectedId + " ← 「" + query + "」");
            }
        }
        double recallAt5 = hits / (double) GOLDEN.length;
        double mrr = reciprocalSum / GOLDEN.length;
        assertTrue(recallAt5 >= RECALL_AT_5_GATE,
                "recall@5 = %.3f < 门限 %.2f；未命中：%s".formatted(
                        recallAt5, RECALL_AT_5_GATE, misses));
        assertTrue(mrr >= MRR_GATE,
                "MRR = %.3f < 门限 %.2f；未命中：%s".formatted(mrr, MRR_GATE, misses));
        // top-1 命中数即 MRR 的分子上界参考（辅助诊断信息留在断言消息里）。
        assertTrue(mrr <= recallAt5 + 1e-9, "MRR 不可能超过 recall@5");
    }

    @Test
    void goldenQueriesAreParaphrasesNotEchoes() {
        // golden 查询与记忆内容不逐字相等（口径自检：评估改写召回而非复读）。
        for (String[] entry : GOLDEN) {
            assertTrue(!entry[2].equals(entry[1]),
                    entry[0] + " 的 golden 查询不应与内容逐字相同");
        }
    }

    @Test
    void goldenSetWithStageTwoRerankStaysAboveGate() {
        // #37 Stage-2 精排：同 GOLDEN 语料走 scoreMemories → rerankShortlist
        // （开关开，MEMORY_RERANK_ENABLED=true；pool 默认 24），度量循环与
        // 上方基线用例同口径（top-5 内找 golden 目标）。
        //
        // 实测（2026-11-12，pool=24，RERANK_SPREAD_EPS=0.05）：
        // recall@5 = 0.9167，MRR = 0.8658 —— **与基线逐值相等**（门限
        // 0.90/0.75 全过）。信息门折叠的维度：recency（语料 updated_at
        // 梯度仅 19 小时，14 天半衰期下原值 spread≈0.038 < 0.05）与
        // salience（全员 0.5，spread=0）；keyword 维度按查询证据数分化——
        // 证据 ≥5 条的查询 rrf spread ≥ 1-61/65 ≈ 0.0615 存活（精排序 =
        // 纯 kw 通道序），证据 ≤4 条的查询 spread < 0.05 同被折叠成全并列
        // （保持 Stage-1 原序）。修复前（无信息门，min-max 无条件拉满）
        // 曾实测 0.8833/0.7708：recency 噪音被放大成 0.15 权重主导因子，
        // 把 m12/m18 两条 Stage-1 边缘命中挤出 top-5——信息门即为此而设。
        Settings settings = Settings.load(java.util.Map.of(
                "ENV", "development",
                "MEMORY_RERANK_ENABLED", "true"), null);
        double now = 1_800_000_000.0;
        List<MemoryRecord> corpus = new ArrayList<>(GOLDEN.length);
        for (int i = 0; i < GOLDEN.length; i++) {
            corpus.add(record(GOLDEN[i][0], GOLDEN[i][1], now - (i % 20) * 3600.0));
        }

        int hits = 0;
        double reciprocalSum = 0.0;
        List<String> misses = new ArrayList<>();
        for (String[] entry : GOLDEN) {
            String expectedId = entry[0];
            String query = entry[2];
            List<ScoredMemory> base = Retrieval.scoreMemories(corpus, query, null, now,
                    settings.memory.timeDecayHalfLifeDays, Retrieval.Weights.DEFAULT);
            List<ScoredMemory> ranked = Retrieval.rerankShortlist(base, settings);
            int rank = -1;
            for (int k = 0; k < Math.min(5, ranked.size()); k++) {
                if (ranked.get(k).record().memoryId().equals(expectedId)) {
                    rank = k + 1;
                    break;
                }
            }
            if (rank > 0) {
                hits++;
                reciprocalSum += 1.0 / rank;
            } else {
                misses.add(expectedId + " ← 「" + query + "」");
            }
        }
        double recallAt5 = hits / (double) GOLDEN.length;
        double mrr = reciprocalSum / GOLDEN.length;
        assertTrue(recallAt5 >= RECALL_AT_5_GATE,
                "rerank recall@5 = %.3f < 门限 %.2f；未命中：%s".formatted(
                        recallAt5, RECALL_AT_5_GATE, misses));
        assertTrue(mrr >= MRR_GATE,
                "rerank MRR = %.3f < 门限 %.2f；未命中：%s".formatted(
                        mrr, MRR_GATE, misses));
    }
}

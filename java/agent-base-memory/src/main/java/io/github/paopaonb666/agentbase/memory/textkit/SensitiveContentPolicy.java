package io.github.paopaonb666.agentbase.memory.textkit;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

/**
 * 敏感内容红线（#48）：记忆入库前的正则规则检查——命中即拒绝入库。
 *
 * <p><b>Java 先行单边增强</b>：Python 侧 {@code agent_base_memory} 的
 * 抽取与手工写入两路均无此层。Java 侧两条入库路径都接本检查——抽取
 * 路径 {@code MemoryPipeline.consolidate}（候选落库前，命中跳过 +
 * rejected 审计）、手工路径 {@code MemoryTools.memory_save}（命中回
 * 脱敏提示文案，不打断对话）。开关为
 * {@code Settings.MemorySettings.sensitiveContentFilter}
 * （env {@code MEMORY_SENSITIVE_CONTENT_FILTER}，默认开）。
 *
 * <p>规则集（误报优先从宽——红线宁可多拦，只拦"高置信形态"）：
 * <ul>
 *   <li>{@code api_key}——{@code sk-} 前缀 + ≥16 位密钥体（OpenAI/
 *       Anthropic/DeepSeek 等家族）；前缀字符前不得是字母数字（排除
 *       risk-/task- 一类英文词的子串）。</li>
 *   <li>{@code aws_access_key}——{@code AKIA} + 16 位大写字母数字
 *       （AWS Access Key ID 标准形态）。</li>
 *   <li>{@code github_token}——{@code ghp_} + ≥20 位（GitHub PAT）。</li>
 *   <li>{@code slack_token}——{@code xoxb-} + ≥10 位（Slack bot token）。</li>
 *   <li>{@code bearer_token}——{@code Bearer}（大小写不敏感）后接
 *       ≥20 位的 token 体。</li>
 *   <li>{@code phone_number}——大陆手机号 {@code 1[3-9] + 9 位数字}，
 *       两侧不得紧邻数字（排除订单号/长数字串的偶然命中）。</li>
 *   <li>{@code id_card}——18 位身份证号的宽松版（17 位数字 + 校验位
 *       数字/X/x；不做 GB 11643 校验位算法复核），两侧不得紧邻数字。</li>
 * </ul>
 *
 * <p>非目标：不做脱敏改写（只判定与报规则名，不动原文）；不覆盖自然
 * 语言形态的敏感信息（如"我家住在……"）；历史数据的一次性扫描清洗
 * 是独立交付物，不在本类职责内。
 */
public final class SensitiveContentPolicy {

    /** 全局共享实例（无状态，规则集固定）。 */
    public static final SensitiveContentPolicy INSTANCE = new SensitiveContentPolicy();

    private SensitiveContentPolicy() {}

    /**
     * 单条红线规则：规则名（审计与拒绝文案里展示）+ 编译好的匹配器。
     * 列表顺序即 {@link #violations} 返回顺序（稳定、可复现）。
     */
    private record Rule(String name, Pattern pattern) {}

    private static final List<Rule> RULES = List.of(
            new Rule("api_key",
                    Pattern.compile("(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{16,}")),
            new Rule("aws_access_key",
                    Pattern.compile("(?<![A-Za-z0-9])AKIA[0-9A-Z]{16}(?![A-Za-z0-9])")),
            new Rule("github_token",
                    Pattern.compile("ghp_[A-Za-z0-9]{20,}")),
            new Rule("slack_token",
                    Pattern.compile("xoxb-[A-Za-z0-9-]{10,}")),
            new Rule("bearer_token",
                    Pattern.compile("(?i)\\bbearer\\s+[A-Za-z0-9._~+/=-]{20,}")),
            new Rule("phone_number",
                    Pattern.compile("(?<!\\d)1[3-9]\\d{9}(?!\\d)")),
            new Rule("id_card",
                    Pattern.compile("(?<!\\d)\\d{17}[0-9Xx](?![0-9Xx])")));

    /**
     * 检查内容命中的规则名列表；<b>空列表 = 干净</b>。null/空白内容
     * 视为干净（上限校验是调用方的职责）。
     */
    public List<String> violations(String content) {
        if (content == null || content.isBlank()) {
            return List.of();
        }
        List<String> hits = new ArrayList<>();
        for (Rule rule : RULES) {
            if (rule.pattern().matcher(content).find()) {
                hits.add(rule.name());
            }
        }
        return List.copyOf(hits);
    }
}

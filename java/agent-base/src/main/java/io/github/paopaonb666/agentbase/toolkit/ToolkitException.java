package io.github.paopaonb666.agentbase.toolkit;

/**
 * 工具库装配失败（启用了不可用的工具、未知工具名等）。
 *
 * <p>对应 Python 侧 {@code tools/spec.py} 的 {@code ToolkitError}（它是
 * {@code ValueError} 的子类，这里同理继承 {@link IllegalArgumentException}）。
 * 治理语义（继承自 Python 侧 registry.py 的注释）：
 *
 * <ul>
 *   <li><b>启用即承诺</b>——显式启用的工具若依赖缺失或配置不全，是启动时
 *       的配置错误（本异常），不是运行时的静默缺席；错误信息带上可执行的
 *       修复指引。</li>
 *   <li><b>名字即契约</b>——{@code TOOLKIT_ENABLED} 的合法取值就是注册表
 *       的键，拼错在启动时快速失败（与 {@code AGENT_MODULES} 的语义一致）。</li>
 * </ul>
 *
 * <p>历史上的 {@code SearchEngineError}（duckduckgo「第一期未迁移」选中
 * 即抛）随 #276 终局移除：Java 侧搜索引擎清单收窄为 tavily，配置边界
 * 由 Settings 启动校验守住。
 */
public class ToolkitException extends IllegalArgumentException {

    public ToolkitException(String message) {
        super(message);
    }
}

package io.github.paopaonb666.agentbase.toolkit;

/**
 * 工具库装配失败（启用了不可用的工具、未知工具名、引擎未迁移等）。
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
 */
public class ToolkitException extends IllegalArgumentException {

    public ToolkitException(String message) {
        super(message);
    }

    /**
     * 搜索引擎装配失败（第一期契约：duckduckgo 引擎未迁移，被选中构建时
     * 直接报错而不是静默跳过——静默降级会让"配了引擎却从未生效"变成
     * 无法察觉的配置漂移）。
     */
    public static final class SearchEngineError extends ToolkitException {

        public SearchEngineError(String message) {
            super(message);
        }
    }
}

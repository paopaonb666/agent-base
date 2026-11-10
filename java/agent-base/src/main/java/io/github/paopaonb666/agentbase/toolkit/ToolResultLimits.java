package io.github.paopaonb666.agentbase.toolkit;

/**
 * 工具结果外置与按需读取（条目 6/23）的共享常量。
 *
 * <p>同一份超阈值工具结果有两条路径，上限矩阵如下：
 * <table summary="外置常量矩阵">
 *   <tr><th>常量</th><th>语义</th><th>消费方</th></tr>
 *   <tr><td>{@link #SPILL_THRESHOLD_CHARS}</td>
 *       <td>外置触发阈值（码点）：工具结果超过它才落库外置；配置键
 *           {@code TOOL_RESULT_SPILL_THRESHOLD} 覆写默认值，0 = 关闭</td>
 *       <td>ContextEngine.spillIfNeeded（Graphs / ExecuteNode 两个组装点
 *           单源）</td></tr>
 *   <tr><td>{@link #FIRST_SCREEN_CHARS}</td>
 *       <td>摘要块携带的首屏长度（码点）——模型不用回读就能判断要不要取全文；
 *           配 renderTranscript 的 300 码点截断线：首屏被转写截住属期望行为</td>
 *       <td>ContextEngine.spillIfNeeded 摘要块</td></tr>
 *   <tr><td>{@link #READ_PAGE_DEFAULT} / {@link #READ_PAGE_MAX}</td>
 *       <td>read_tool_result 单屏默认/上限长度（码点）——上限防一次读回又
 *           撑爆上下文（自伤循环）；越界报错不钳制</td>
 *       <td>ReadToolResultTool</td></tr>
 *   <tr><td>{@link #MAX_STORED_CHARS}</td>
 *       <td>外置行的存储上限（码点）：全文超限时头部截断存入，摘要块末尾
 *           注记「全文超上限已截断存储」</td>
 *       <td>ContextEngine.spillIfNeeded 落库段</td></tr>
 * </table>
 *
 * <p>形态先例：{@code M/tools/BlockLimits.java}（#178 收口）——同族常量
 * 单文件收口，消费方不散落魔法数。摘要块格式与 read_tool_result 的返回
 * 形状都是<b>一次性定稿契约</b>：改动须同步 ContextEngine/ReadToolResultTool
 * 与锁定它们的测试。
 */
public final class ToolResultLimits {

    private ToolResultLimits() {}

    /** 外置触发阈值（码点；配置键 TOOL_RESULT_SPILL_THRESHOLD，0 = 关闭）。 */
    public static final int SPILL_THRESHOLD_CHARS = 4000;

    /** 摘要块首屏长度（码点）。 */
    public static final int FIRST_SCREEN_CHARS = 1200;

    /** read_tool_result 单屏默认长度（码点）。 */
    public static final int READ_PAGE_DEFAULT = 2000;

    /** read_tool_result 单屏上限（码点；越界报错不钳制）。 */
    public static final int READ_PAGE_MAX = 8000;

    /** 外置行存储上限（码点）：全文超限头部截断存入并在摘要块注记。 */
    public static final int MAX_STORED_CHARS = 200_000;
}

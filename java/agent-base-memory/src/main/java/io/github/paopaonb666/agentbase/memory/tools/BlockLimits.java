package io.github.paopaonb666.agentbase.memory.tools;

/**
 * 记忆块（memory block）三路径上限的共享常量（#178 收口）。
 *
 * <p>同一份 content 有三条写入路径，上限矩阵如下：
 * <table summary="三路径上限矩阵">
 *   <tr><th>路径</th><th>content 上限</th><th>char_limit</th><th>截断语义</th></tr>
 *   <tr><td>API PUT /v1/memory/blocks/{label}</td>
 *       <td>1..{@link #CONTENT_MAX_API}</td>
 *       <td>{@link #CHAR_LIMIT_MIN}..{@link #CHAR_LIMIT_MAX}（缺省 {@link #CHAR_LIMIT_DEFAULT}）</td>
 *       <td>越界 422，服务层 {@code putBlock} 原样入库</td></tr>
 *   <tr><td>工具 memory_update_block</td>
 *       <td>1..{@link #CONTENT_MAX_TOOL}</td>
 *       <td>沿用块的声明容量（无则 {@link #CHAR_LIMIT_DEFAULT}）</td>
 *       <td>入参越界抛错；append 合并后超块容量按 char_limit 保尾截断</td></tr>
 *   <tr><td>服务 MemoryService.putBlock</td>
 *       <td>不设上限（由上层把关）</td>
 *       <td>仅作为声明容量存储</td>
 *       <td><b>永不截断</b>——char_limit 只是声明容量，注入组装时才按预算
 *           裁剪；在这里截断会永久丢字（Python api.py put_memory_block 同语义）</td></tr>
 * </table>
 *
 * <p>工具侧 4000 低于 API 侧 8000：模型写入走更保守的预算，管理端
 * （API）保留更大的手写余量。三路径同一块内容的对齐契约由
 * {@code MemoryServiceBlockTest} 的三路径一致性矩阵锁定。
 */
public final class BlockLimits {

    private BlockLimits() {}

    /** API 写入路径的 content 上限（PUT /v1/memory/blocks/{label} 校验 422）。 */
    public static final int CONTENT_MAX_API = 8000;

    /** 工具写入路径（memory_update_block）的 content 入参上限。 */
    public static final int CONTENT_MAX_TOOL = 4000;

    /** 块声明容量缺省值（API 未传 char_limit、工具路径无既有块时）。 */
    public static final int CHAR_LIMIT_DEFAULT = 2000;

    /** 块声明容量下界（API char_limit 校验）。 */
    public static final int CHAR_LIMIT_MIN = 50;

    /** 块声明容量上界（API char_limit 校验，与 API content 上限同值）。 */
    public static final int CHAR_LIMIT_MAX = 8000;
}

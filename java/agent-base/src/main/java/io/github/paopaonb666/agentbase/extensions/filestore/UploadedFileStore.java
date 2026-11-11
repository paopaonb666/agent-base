package io.github.paopaonb666.agentbase.extensions.filestore;

import java.util.List;

/**
 * 附件存储的最小接口；实现必须保证并发安全（对应 Python 侧
 * {@code extensions/filestore.py} 的 {@code UploadedFileStore} Protocol）。
 *
 * <p>语义要点（继承自 Python 侧设计说明）：
 * <ul>
 *   <li><b>thread_id 发送时绑定</b>：上传发生在对话线程生成之前（新对话
 *       的首个附件），行先以空 thread_id 入库，invoke 引用时
 *       {@link #bindThread(List, String)} 回填。</li>
 *   <li><b>孤儿机会式清理</b>：24 小时（{@link #ORPHAN_MAX_AGE_HOURS}）
 *       未绑定的孤儿行由 {@link #purgeOrphans(int)} 删除，上传/撤销等
 *       时机顺带调用。</li>
 *   <li><b>getMany 保序</b>：按请求的 fileIds 顺序返回命中项（与前端
 *       附件列表顺序一致），缺失的 id 静默跳过。</li>
 * </ul>
 *
 * <p>方法签名与 Python 协议逐一对齐；Java 侧为同步阻塞语义（实现各自
 * 消化并发，对应 Python 的全部异步接口在线程模型上的等价物）。
 */
public interface UploadedFileStore extends AutoCloseable {

    /** 未绑定线程的孤儿附件保留时长（小时），超过后在上传时机会式清理。 */
    int ORPHAN_MAX_AGE_HOURS = 24;

    /**
     * 工具结果外置哨兵行的 module 值（#6）：这类行是超大工具结果的全文
     * 寄存，不是用户上传的知识文件——列表侧必须排除（L92）。
     */
    String TOOL_RESULT_MODULE = "tool_result";

    /** 保存（或按 file_id 覆写）一条附件。 */
    void save(UploadedFileInfo info);

    /** 按 id 批量取附件：只返回存在的行，且顺序跟随请求（保序）。 */
    List<UploadedFileInfo> getMany(List<String> fileIds);

    /**
     * 某用户的附件清单（按创建时间倒序，最多 {@code limit} 条）。
     * {@code module} 传 null 表示不过滤模块；{@code limit} 下限为 1。
     */
    List<UploadedFileInfo> listForUser(String userId, String module, int limit);

    /** 某用户的附件清单（不过滤模块，默认 200 条；对应 Python 的默认参数）。 */
    default List<UploadedFileInfo> listForUser(String userId) {
        return listForUser(userId, null, 200);
    }

    /**
     * 某用户的<b>可见</b>附件清单（L92/R4-b 收口）：等价
     * {@code listForUser(userId, null, 200)} 但恒排除
     * {@link #TOOL_RESULT_MODULE} 哨兵行——工具结果外置行不是知识文件，
     * 不得泄入 GET /v1/knowledge/files，也无法经既有 DELETE 路由删除
     * （requireModule 拒非模块名，哨兵行删除走线程级联）。默认形态取回
     * 后内存过滤；SQL 后端应覆写为查询级下推（{@code module != ?}），
     * 让 LIMIT 预算花在真实文件上。
     */
    default List<UploadedFileInfo> listVisibleForUser(String userId) {
        List<UploadedFileInfo> all = listForUser(userId, null, 200);
        List<UploadedFileInfo> visible = new java.util.ArrayList<>(all.size());
        for (UploadedFileInfo info : all) {
            if (!TOOL_RESULT_MODULE.equals(info.module())) {
                visible.add(info);
            }
        }
        return List.copyOf(visible);
    }

    /** 把一批附件回填绑定到对话线程（上传先于线程生成的时序补偿）。 */
    void bindThread(List<String> fileIds, String threadId);

    /** 删除某线程的全部附件（线程删除时的级联清理）。 */
    void deleteForThread(String threadId);

    /** 按 id 删除单份附件；返回是否真的删了（不存在返回 false）。 */
    boolean deleteFile(String fileId);

    /**
     * 清理孤儿：thread_id 为空且 {@code created_at} 早于
     * {@code now - maxAgeHours} 的行删除；返回删除行数。
     */
    int purgeOrphans(int maxAgeHours);

    /** 清理孤儿（默认 24h；对应 Python 的默认参数）。 */
    default int purgeOrphans() {
        return purgeOrphans(ORPHAN_MAX_AGE_HOURS);
    }

    /** 释放后端资源（连接等）；默认无操作（memory 后端无事可做）。 */
    @Override
    default void close() {}
}

package io.github.paopaonb666.agentbase.core.fixtures.boom;

/** 静态初始化失败的 fixture（LinkageError → 按导入失败上报）。 */
public class BoomModule {

    static {
        // 条件形式绕过 javac 的"初始化器必须能够正常完成"检查——
        // 运行期恒真，类初始化必然失败（ExceptionInInitializerError）。
        if (Boolean.parseBoolean("true")) {
            throw new IllegalStateException("fixture boom");
        }
    }
}

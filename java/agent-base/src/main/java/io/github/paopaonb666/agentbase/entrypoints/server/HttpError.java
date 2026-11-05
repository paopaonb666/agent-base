package io.github.paopaonb666.agentbase.entrypoints.server;

/**
 * HTTP 层错误（对应 Python 侧 fastapi.HTTPException 的受控子集）：
 * {@code status} + 人类可读 {@code detail}；由全局异常处理器渲染为
 * FastAPI 同构响应体 {@code {"detail": …}}。
 */
public final class HttpError extends RuntimeException {

    private final int status;

    public HttpError(int status, String detail) {
        super(detail);
        this.status = status;
    }

    public int status() {
        return status;
    }
}

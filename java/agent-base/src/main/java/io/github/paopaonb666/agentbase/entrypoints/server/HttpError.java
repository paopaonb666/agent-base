package io.github.paopaonb666.agentbase.entrypoints.server;

/**
 * HTTP 层错误（对应 Python 侧 fastapi.HTTPException 的受控子集）：
 * {@code status} + 人类可读 {@code detail} + 结构化 {@link ErrorCode}
 * （#91——前端只认 code 不认文案）；由全局异常处理器渲染为
 * {@code {"detail": …, "code": …}}（code 缺省时省键，向后兼容）。
 */
public final class HttpError extends RuntimeException {

    private final int status;
    private final ErrorCode code;

    public HttpError(int status, String detail) {
        this(status, null, detail);
    }

    public HttpError(int status, ErrorCode code, String detail) {
        super(detail);
        this.status = status;
        this.code = code;
    }

    public int status() {
        return status;
    }

    /** 结构化错误码（#91）；两参旧构造为 null（渲染时省键）。 */
    public ErrorCode code() {
        return code;
    }
}

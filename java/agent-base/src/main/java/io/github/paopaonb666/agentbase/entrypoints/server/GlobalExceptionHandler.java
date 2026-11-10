package io.github.paopaonb666.agentbase.entrypoints.server;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.ResponseEntity;
import org.springframework.http.converter.HttpMessageNotReadableException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.method.annotation.MethodArgumentTypeMismatchException;
import org.springframework.web.multipart.support.MissingServletRequestPartException;
import org.springframework.web.bind.MissingServletRequestParameterException;

/**
 * 全局异常处理：{@link HttpError} → status + {"detail": …}（FastAPI 同构
 * 响应体）；请求体不可读/参数缺失 → 422（FastAPI 校验语义）；其余 500
 * 不泄栈（完整堆栈已带 request_id 落服务端日志）。
 */
@RestControllerAdvice
public class GlobalExceptionHandler {

    private static final Logger LOG = LoggerFactory.getLogger(GlobalExceptionHandler.class);

    @ExceptionHandler(HttpError.class)
    public ResponseEntity<Map<String, Object>> httpError(HttpError e) {
        // #91：{detail, code} 旁挂——code 缺省（两参旧构造）省键，旧 wire
        // 逐字节不变；前端只认 code 不认文案（文案可能随语言漂移）。
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("detail", e.getMessage() == null ? "" : e.getMessage());
        if (e.code() != null) {
            body.put("code", e.code().wire());
        }
        return ResponseEntity.status(e.status()).body(body);
    }

    @ExceptionHandler({HttpMessageNotReadableException.class,
            MissingServletRequestParameterException.class,
            MethodArgumentTypeMismatchException.class,
            MissingServletRequestPartException.class})
    public ResponseEntity<Map<String, Object>> validation(Exception e) {
        // FastAPI 校验错误形态：detail 是列表（loc/msg/type）——这里保留
        // 列表形态与 422 状态码，message 放 msg 槽位；code 在 detail 外旁挂
        // （#91——列表内不加键，保持 FastAPI 形状）。
        Map<String, Object> item = new LinkedHashMap<>();
        item.put("msg", e.getMessage() == null ? "请求不合法" : e.getMessage());
        item.put("type", "validation_error");
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("detail", List.of(item));
        body.put("code", ErrorCode.VALIDATION.wire());
        return ResponseEntity.unprocessableEntity().body(body);
    }

    @ExceptionHandler(org.springframework.web.multipart.MaxUploadSizeExceededException.class)
    public ResponseEntity<Map<String, Object>> tooLarge(Exception e) {
        return ResponseEntity.status(413)
                .body(Map.of("detail", "文件超出大小上限", "code", ErrorCode.BAD_REQUEST.wire()));
    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity<Map<String, Object>> internal(Exception e) {
        LOG.error("server: unhandled exception", e);
        return ResponseEntity.internalServerError()
                .body(Map.of("detail", "Internal Server Error", "code", ErrorCode.INTERNAL_ERROR.wire()));
    }
}

package io.github.paopaonb666.agentbase.entrypoints.server;

import org.junit.jupiter.api.Test;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.http.converter.HttpMessageNotReadableException;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 全局异常处理器渲染契约（#91 前零覆盖）：{detail, code} 形状、code 缺省
 * 省键（旧 wire 逐字节兼容）、422 列表形态保持 FastAPI 形状 + code 旁挂、
 * 413/500 兜底带码。直接 new handler 断言 ResponseEntity，无 MockMvc 依赖。
 */
class GlobalExceptionHandlerTest {

    private final GlobalExceptionHandler handler = new GlobalExceptionHandler();

    @SuppressWarnings("unchecked")
    private static Map<String, Object> bodyOf(ResponseEntity<Map<String, Object>> response) {
        return response.getBody();
    }

    @Test
    void httpErrorRendersDetailWithCodeWhenPresent() {
        ResponseEntity<Map<String, Object>> response = handler.httpError(
                new HttpError(409, ErrorCode.THREAD_BUSY, "线程忙"));
        assertEquals(HttpStatus.CONFLICT, response.getStatusCode());
        Map<String, Object> body = bodyOf(response);
        assertEquals("线程忙", body.get("detail"));
        assertEquals("thread_busy", body.get("code"));
    }

    @Test
    void httpErrorWithoutCodeKeepsLegacyWireShape() {
        // 两参旧构造：code 缺省省键——旧消费端读到的 body 逐字节不变。
        ResponseEntity<Map<String, Object>> response = handler.httpError(
                new HttpError(404, "线程不存在"));
        assertEquals("线程不存在", bodyOf(response).get("detail"));
        assertFalse(bodyOf(response).containsKey("code"));
    }

    @Test
    void validationErrorsKeepFastApiListShapeWithCodeAside() {
        // 422 的 detail 仍是 [{msg,type}] 列表（FastAPI 形状）；code 在列表外旁挂。
        ResponseEntity<Map<String, Object>> response = handler.validation(
                new HttpMessageNotReadableException("bad body"));
        assertEquals(HttpStatus.UNPROCESSABLE_ENTITY, response.getStatusCode());
        Map<String, Object> body = bodyOf(response);
        Object detail = body.get("detail");
        assertTrue(detail instanceof List<?> list && list.size() == 1, String.valueOf(detail));
        @SuppressWarnings("unchecked")
        Map<String, Object> item = (Map<String, Object>) ((List<?>) detail).get(0);
        assertEquals("validation_error", item.get("type"));
        assertEquals("validation", body.get("code"));
    }

    @Test
    void payloadTooLargeAndInternalFallBacksCarryCodes() {
        ResponseEntity<Map<String, Object>> tooLarge = handler.tooLarge(
                new org.springframework.web.multipart.MaxUploadSizeExceededException(1024L));
        assertEquals(HttpStatus.PAYLOAD_TOO_LARGE, tooLarge.getStatusCode());
        assertEquals("bad_request", bodyOf(tooLarge).get("code"));

        ResponseEntity<Map<String, Object>> internal = handler.internal(new RuntimeException("boom"));
        assertEquals(HttpStatus.INTERNAL_SERVER_ERROR, internal.getStatusCode());
        assertEquals("Internal Server Error", bodyOf(internal).get("detail"));
        assertEquals("internal_error", bodyOf(internal).get("code"));
    }

    @Test
    void errorCodeWireValuesCoverTheClosedSet() {
        // 封闭枚举的 wire 值锁定（前端按字面量匹配）。
        for (String wire : List.of("rate_limited", "budget_blocked", "run_gate_saturated",
                "thread_busy", "unauthorized", "not_found", "validation", "bad_request",
                "module_unavailable", "storage_unavailable", "tool_timeout",
                "context_overflow", "llm_unavailable", "recursion_limit", "internal_error")) {
            boolean seen = false;
            for (ErrorCode code : ErrorCode.values()) {
                seen |= code.wire().equals(wire);
            }
            assertTrue(seen, "缺少错误码 wire 值：" + wire);
        }
    }
}

package io.github.paopaonb666.agentbase.entrypoints.server;

import io.github.paopaonb666.agentbase.core.config.Settings;

import org.junit.jupiter.api.Test;
import org.springframework.web.servlet.config.annotation.CorsRegistration;
import org.springframework.web.servlet.config.annotation.CorsRegistry;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyBoolean;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/** #117：CORS 方法集收窄——Java 路由面只有 GET/POST/DELETE，HEAD/OPTIONS 供预检与探测。 */
class ServerConfigurationCorsTest {

    @Test
    void corsMethodsAreNarrowedToRoutingSurface() {
        CorsRegistry registry = mock(CorsRegistry.class);
        CorsRegistration registration = mock(CorsRegistration.class);
        when(registry.addMapping("/**")).thenReturn(registration);
        when(registration.allowedOrigins(any(String[].class))).thenReturn(registration);
        when(registration.allowedMethods(any(String[].class))).thenReturn(registration);
        when(registration.exposedHeaders(any(String[].class))).thenReturn(registration);
        when(registration.allowCredentials(anyBoolean())).thenReturn(registration);

        new ServerConfiguration().corsConfigurer(new Settings()).addCorsMappings(registry);

        verify(registration).allowedMethods("GET", "POST", "DELETE", "HEAD", "OPTIONS");
        // #227：暴露给前端消费面的响应头（限流/闸门 429 与忙 409 都带
        // Retry-After；X-Request-ID 用于排障关联）。
        verify(registration).exposedHeaders("X-Request-ID", "Retry-After");
        verify(registration).allowCredentials(false);
    }
}

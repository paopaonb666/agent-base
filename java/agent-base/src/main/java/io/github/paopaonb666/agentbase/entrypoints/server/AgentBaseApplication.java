package io.github.paopaonb666.agentbase.entrypoints.server;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.ConfigurableApplicationContext;

/**
 * SSE 服务入口（对应 {@code uvicorn agent_base.entrypoints.server:app}）。
 * 由 CLI 的 {@code --server} 标志反射加载启动；端口 8000（application.properties）。
 */
@SpringBootApplication
public class AgentBaseApplication {

    public static void main(String[] args) {
        ConfigurableApplicationContext context = SpringApplication.run(AgentBaseApplication.class, args);
        context.registerShutdownHook();
    }
}

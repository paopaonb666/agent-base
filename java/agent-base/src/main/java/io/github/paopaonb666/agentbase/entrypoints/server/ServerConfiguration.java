package io.github.paopaonb666.agentbase.entrypoints.server;

import java.util.Map;

import io.github.paopaonb666.agentbase.core.AgentRuntime;
import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.core.memory.MemoryPlugins;
import io.github.paopaonb666.agentbase.extensions.Observability;
import io.github.paopaonb666.agentbase.extensions.costmeter.CostGovernor;
import io.github.paopaonb666.agentbase.extensions.costmeter.CostLedger;
import io.github.paopaonb666.agentbase.extensions.costmeter.CostLedgers;
import io.github.paopaonb666.agentbase.extensions.costmeter.CostMeters;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.web.servlet.FilterRegistrationBean;
import org.springframework.context.ConfigurableApplicationContext;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.Ordered;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

/**
 * 服务装配（对应 Python 侧 {@code create_app} + lifespan）：
 *
 * <ul>
 *   <li>Settings / AgentRuntime / AuthBackend / CostGovernor 单例 bean；</li>
 *   <li>成本治理：COST_ENABLED=true 时配置价目表 → 按后端选账本 → 重启
 *       恢复（totalsByDay 回放）→ 预算门；关闭时 governor 为 null；</li>
 *   <li>CORS / request-id / metrics 过滤器与全局异常处理；</li>
 *   <li>记忆路由注册：MEMORY_ENABLED=true 时经 SPI 发现控制器类并注册
 *       （未启用不注册——HTTP 层 404 语义）；</li>
 *   <li>关闭时：取消 flush、最终落账、runtime.close()。</li>
 * </ul>
 */
@Configuration
public class ServerConfiguration {

    private static final Logger LOG = LoggerFactory.getLogger(ServerConfiguration.class);

    /** 成本明细 flush 周期（秒）——对应 Python 的 _COST_FLUSH_INTERVAL_SECONDS=5。 */
    private static final long COST_FLUSH_INTERVAL_SECONDS = 5;

    @Bean
    public Settings settings() {
        // 构造期 fail-fast：配置错误拒绝启动。
        return Settings.load(Map.of());
    }

    @Bean(destroyMethod = "close")
    public AgentRuntime agentRuntime(Settings settings) {
        Observability.setupLogging(settings.observability.logJson);
        Observability.logTracingConfig();
        return AgentRuntime.create(settings);
    }

    @Bean
    public AuthBackends.AuthBackend authBackend() {
        return new AuthBackends.HmacHeaderAuth();
    }

    @Bean
    public CostGovernor costGovernor(Settings settings, ConfigurableApplicationContext context) {
        if (!settings.cost.enabled) {
            return null;
        }
        CostMeters.cost().configure(settings.prices());
        CostLedger ledger = CostLedgers.build(settings);
        if (ledger != null) {
            // 重启恢复：账本 totalsByDay 回放进计量器（当日/当月累计延续）。
            ledger.totalsByDay().forEach((day, cost) ->
                    CostMeters.cost().restoreDay(day, 0, 0, cost));
            // 周期 flush：把 pending 明细落账（守护线程，进程退出时中止）。
            Thread flusher = new Thread(() -> {
                while (!Thread.currentThread().isInterrupted()) {
                    try {
                        Thread.sleep(COST_FLUSH_INTERVAL_SECONDS * 1000);
                        CostMeters.cost().takePending().forEach(ledger::append);
                        ledger.flush();
                    } catch (InterruptedException e) {
                        return;
                    }
                }
            }, "cost-flush");
            flusher.setDaemon(true);
            flusher.start();
            Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                flusher.interrupt();
                try {
                    CostMeters.cost().takePending().forEach(ledger::append);
                    ledger.flush();
                    ledger.close();
                } catch (Exception e) {
                    LOG.warn("cost: 最终 flush 失败", e);
                }
            }));
        }
        return new CostGovernor(settings.cost.dailyCny, settings.cost.monthlyCny,
                settings.cost.action, CostMeters.cost());
    }

    /** CORS（对应 Python CORSMiddleware）：origins 来自 CORS_ORIGINS。 */
    @Bean
    public WebMvcConfigurer corsConfigurer(Settings settings) {
        return new WebMvcConfigurer() {
            @Override
            public void addCorsMappings(CorsRegistry registry) {
                registry.addMapping("/**")
                        .allowedOrigins(settings.corsOrigins.toArray(String[]::new))
                        .allowedMethods("*")
                        .exposedHeaders("X-Request-ID")
                        .allowCredentials(false);
            }
        };
    }

    /** request-id + metrics 过滤器（顺序：先 request-id 后 metrics）。 */
    @Bean
    public FilterRegistrationBean<WebFilters.RequestIdFilter> requestIdFilter(Settings settings) {
        FilterRegistrationBean<WebFilters.RequestIdFilter> registration = new FilterRegistrationBean<>();
        registration.setFilter(new WebFilters.RequestIdFilter());
        registration.addUrlPatterns("/*");
        registration.setOrder(Ordered.HIGHEST_PRECEDENCE);
        return registration;
    }

    @Bean
    public FilterRegistrationBean<WebFilters.MetricsFilter> metricsFilter(Settings settings) {
        FilterRegistrationBean<WebFilters.MetricsFilter> registration = new FilterRegistrationBean<>();
        registration.setFilter(new WebFilters.MetricsFilter());
        registration.addUrlPatterns("/*");
        registration.setOrder(Ordered.HIGHEST_PRECEDENCE + 1);
        return registration;
    }

    /**
     * 记忆路由注册（对应 load_memory_routers）：MEMORY_ENABLED=true 时经
     * SPI 发现控制器并在容器<b>最早期</b>注册 bean 定义（控制器构造器
     * 参数由 Spring 自动装配）——晚期注册会错过 RequestMappingHandlerMapping
     * 的扫描。装配失败降级为不挂载（/v1/memory 404，服务照常启动），
     * 对齐 Python 的容错语义。未启用时不注册（404 语义）。
     */
    @Bean
    public static org.springframework.beans.factory.support.BeanDefinitionRegistryPostProcessor
            memoryControllerRegistrar() {
        return new org.springframework.beans.factory.support.BeanDefinitionRegistryPostProcessor() {
            @Override
            public void postProcessBeanDefinitionRegistry(
                    org.springframework.beans.factory.support.BeanDefinitionRegistry registry) {
                Settings settings = Settings.load(Map.of());
                if (!settings.memory.enabled) {
                    return;
                }
                try {
                    for (Class<?> controller : MemoryPlugins.loadRouteControllers()) {
                        registry.registerBeanDefinition(controller.getName(),
                                org.springframework.beans.factory.support.BeanDefinitionBuilder
                                        .genericBeanDefinition(controller).getBeanDefinition());
                    }
                } catch (MemoryPlugins.MemoryPluginError e) {
                    LOG.warn("memory: 管理路由装配失败（/v1/memory 不可用，服务继续）：{}", e.toString());
                }
            }

            @Override
            public void postProcessBeanFactory(
                    org.springframework.beans.factory.config.ConfigurableListableBeanFactory beanFactory) {
                // 不修改 bean 定义。
            }
        };
    }
}

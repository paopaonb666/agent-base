package io.github.paopaonb666.agentbase.core;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 模块注册表测试（roadmap #295）：显式清单加载、命名约定、失败模式全表
 * （重复/非法名/非法包/未知模块/契约不一致/未实现接口/静态初始化失败/
 * 抽象类不可实例化）、prefix 优先级与 .env 兼容垫片。fixture 模块在
 * core.fixtures 包（{包}.{名}.{Name}Module 布局约定的测试落地）。
 */
class ModuleRegistryTest {

    private static final String FIXTURES = "io.github.paopaonb666.agentbase.core.fixtures";

    @Test
    void loadsModulesFromDefaultPackageInListOrder() {
        // 默认包：基座自带模块按 AGENT_MODULES 顺序装配（插入序 == 装配序）。
        var modules = ModuleRegistry.loadModules(List.of("chat", "hello"), null, null);
        assertEquals(List.of("chat", "hello"), List.copyOf(modules.keySet()));
        assertInstanceOf(AgentModule.class, modules.get("chat"));
        assertEquals("chat", modules.get("chat").name());
    }

    @Test
    void loadsFixtureModuleViaPrefix() {
        // prefix 快捷方式：测试把 registry 指向 fixture 包。
        var modules = ModuleRegistry.loadModules(List.of("stubmod"), FIXTURES, null);
        assertEquals("stubmod", modules.get("stubmod").name());
    }

    @Test
    void rejectsDuplicateNames() {
        ModuleRegistry.RegistryError e = assertThrows(ModuleRegistry.RegistryError.class,
                () -> ModuleRegistry.loadModules(List.of("chat", "chat"), null, null));
        assertTrue(e.getMessage().contains("duplicate module 'chat'"));
    }

    @Test
    void rejectsInvalidModuleNames() {
        // 反路径遍历/反加载任意类的严格白名单：大写、连字符、数字开头、
        // 点分隔全部拒绝。
        for (String bad : List.of("Chat", "chat-x", "1chat", "chat.x", "", "..", "chät")) {
            ModuleRegistry.RegistryError e = assertThrows(ModuleRegistry.RegistryError.class,
                    () -> ModuleRegistry.loadModules(List.of(bad), FIXTURES, null),
                    "expected rejection of '" + bad + "'");
            assertTrue(e.getMessage().startsWith("invalid module name"), bad);
        }
    }

    @Test
    void rejectsInvalidPackageFormats() {
        ModuleRegistry.RegistryError e = assertThrows(ModuleRegistry.RegistryError.class,
                () -> ModuleRegistry.loadModules(List.of("stubmod"), null, List.of("Bad-Package")));
        assertTrue(e.getMessage().contains("invalid module package"));
    }

    @Test
    void explicitEmptyPackageListIsAConfigurationError() {
        // 显式空列表 ≠ 走默认：注册成"什么都没有"必须 fail-fast。
        ModuleRegistry.RegistryError e = assertThrows(ModuleRegistry.RegistryError.class,
                () -> ModuleRegistry.loadModules(List.of("chat"), null, List.of()));
        assertTrue(e.getMessage().contains("no module packages configured"));
    }

    @Test
    void unknownModuleListsTriedPackages() {
        ModuleRegistry.RegistryError e = assertThrows(ModuleRegistry.RegistryError.class,
                () -> ModuleRegistry.loadModules(List.of("nope"), FIXTURES, null));
        assertTrue(e.getMessage().contains("unknown module 'nope'"));
        assertTrue(e.getMessage().contains("'" + FIXTURES + "'"));
    }

    @Test
    void prefixTakesPrecedenceOverPackageList() {
        // prefix 非 null 时优先于 packages：默认包里的 chat 不再可见。
        ModuleRegistry.RegistryError e = assertThrows(ModuleRegistry.RegistryError.class,
                () -> ModuleRegistry.loadModules(List.of("chat"), FIXTURES,
                        List.of(ModuleRegistry.DEFAULT_MODULE_PACKAGE)));
        assertTrue(e.getMessage().contains("unknown module 'chat'"));
    }

    @Test
    void nameMismatchFailsTheContractCheck() {
        ModuleRegistry.RegistryError e = assertThrows(ModuleRegistry.RegistryError.class,
                () -> ModuleRegistry.loadModules(List.of("broken"), FIXTURES, null));
        assertTrue(e.getMessage().contains("declares name 'other'"));
    }

    @Test
    void nonAgentModuleClassIsRejected() {
        ModuleRegistry.RegistryError e = assertThrows(ModuleRegistry.RegistryError.class,
                () -> ModuleRegistry.loadModules(List.of("notmodule"), FIXTURES, null));
        assertTrue(e.getMessage().contains("does not implement AgentModule"));
    }

    @Test
    void linkageErrorIsReportedAsLoadFailure() {
        // 静态初始化爆炸（依赖缺失/初始化异常）→ "failed to load"，
        // 不是被吞掉也不是裸 LinkageError 逃逸。
        ModuleRegistry.RegistryError e = assertThrows(ModuleRegistry.RegistryError.class,
                () -> ModuleRegistry.loadModules(List.of("boom"), FIXTURES, null));
        assertTrue(e.getMessage().contains("failed to load"));
        assertTrue(e.getMessage().contains("ExceptionInInitializerError"));
    }

    @Test
    void abstractModuleFailsToInstantiate() {
        ModuleRegistry.RegistryError e = assertThrows(ModuleRegistry.RegistryError.class,
                () -> ModuleRegistry.loadModules(List.of("abstractmod"), FIXTURES, null));
        assertTrue(e.getMessage().contains("failed to instantiate"));
    }

    @Test
    void moduleClassNameFollowsCapitalizationConvention() {
        assertEquals("ChatModule", ModuleRegistry.moduleClassName("chat"));
        assertEquals("HelloModule", ModuleRegistry.moduleClassName("hello"));
        assertEquals("A_moduleModule", ModuleRegistry.moduleClassName("a_module"));
    }

    @Test
    void pythonLegacyPackageNameIsNormalized() {
        // .env 兼容垫片：Python 侧默认包名映射到 Java 包，其他原样。
        assertEquals(ModuleRegistry.DEFAULT_MODULE_PACKAGE,
                ModuleRegistry.normalizePackage("agent_base.modules"));
        assertEquals("com.example.modules", ModuleRegistry.normalizePackage("com.example.modules"));
    }
}

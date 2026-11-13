package io.github.paopaonb666.agentbase.core;

import java.lang.reflect.InvocationTargetException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * 显式清单的模块注册表（ADR-002）。
 *
 * <p>AGENT_MODULES 是"哪些模块处于激活状态、以及以什么顺序"的唯一事实
 * 来源。对于每个名字，registry 会：
 *
 * <ol>
 *   <li>用严格的标识符模式校验名字（反路径遍历 / 反加载任意类）；</li>
 *   <li>依次尝试各个模块包，实例化 {@code <包>.<Name>Module}（命名约定
 *       对应 Python 侧的 {@code importlib.import_module(f"{pkg}.{name}")}
 *       + {@code module} 属性；默认包 {@code io.github.paopaonb666.agentbase.modules}
 *       下按 {@code <Name>Module} 命名，如 {@code chat} → {@code ChatModule}）；</li>
 *   <li>校验它符合契约、且其声明的 {@code name()} 与请求的名字一致。</li>
 * </ol>
 *
 * <p>任何失败都会中止启动——有问题的模块永远不会被静默跳过。
 *
 * <p><b>失败矩阵（#319/#113，行为由 {@code ModuleRegistryTest} 锁定）</b>：
 * <ul>
 *   <li><b>重复名字</b>（AGENT_MODULES 含重复条目）→ RegistryError；</li>
 *   <li><b>非法名字</b>（不过标识符模式：路径遍历/任意类加载面）→
 *       RegistryError；</li>
 *   <li><b>非法包名</b> / <b>空清单</b> / <b>未知模块</b>（包里找不到类）→
 *       RegistryError 带可用清单文案；</li>
 *   <li><b>类名/声明名不一致</b>（实例化成功但 {@code name()} 与请求名
 *       不同）→ RegistryError；</li>
 *   <li><b>未实现 AgentModule 接口</b> / <b>抽象类</b>（无法实例化）→
 *       RegistryError；</li>
 *   <li><b>构造期异常 / LinkageError</b>（类初始化失败、依赖缺失）→
 *       RegistryError 原因链保留。</li>
 * </ul>
 * 兼容垫片：配置出现 Python 默认包名时自动映射（见下）——映射后的包
 * 同样参与上述失败矩阵。
 *
 * <p>对应 Python 侧 {@code core/registry.py}；兼容垫片：配置里出现
 * Python 侧的默认包名 {@code agent_base.modules} 时自动映射到 Java 包
 * （同名配置文件无需迁移）。
 */
public final class ModuleRegistry {

    /** 默认模块包：基座自带的样板模块（chat/writer/hello/planner）。 */
    public static final String DEFAULT_MODULE_PACKAGE = "io.github.paopaonb666.agentbase.modules";

    /** Python 侧默认包名 → Java 包（.env 兼容垫片）。 */
    private static final String PYTHON_LEGACY_PACKAGE = "agent_base.modules";

    // 模块名：小写标识符，无路径分隔符，不能以数字开头。严格白名单让
    // 失败信息更可操作，并排除 `..` 之类的逃逸写法。
    private static final Pattern MODULE_NAME_RE = Pattern.compile("^[a-z][a-z0-9_]*$");

    // 模块包：小写标识符，点分隔（Java 包名与 Python 包名同构）。
    private static final Pattern PACKAGE_RE = Pattern.compile("^[a-z][a-z0-9_]*(\\.[a-z][a-z0-9_]*)*$");

    /** 模块加载失败（未知名字 / 重复 / 契约问题）时抛出。 */
    public static final class RegistryError extends IllegalArgumentException {
        public RegistryError(String message) { super(message); }
    }

    private ModuleRegistry() {}

    /**
     * 按顺序加载并校验指定的模块。返回有序映射 {@code name -> AgentModule}
     * （插入顺序 == 装配顺序）。遇到第一个问题即抛出 {@link RegistryError}。
     *
     * @param prefix   单包快捷方式（测试把 registry 指向 fixture 包时用），
     *                 非 null 时优先于 packages
     * @param packages 模块包的 Java 包名列表，按顺序尝试
     */
    public static Map<String, AgentModule> loadModules(List<String> names,
                                                       String prefix, List<String> packages) {
        List<String> search = searchPackages(prefix, packages);
        Map<String, AgentModule> modules = new LinkedHashMap<>();
        for (String name : names) {
            if (modules.containsKey(name)) {
                throw new RegistryError("duplicate module '%s' in AGENT_MODULES".formatted(name));
            }
            modules.put(name, loadOne(name, search));
        }
        return modules;
    }

    private static List<String> searchPackages(String prefix, List<String> packages) {
        // 对应 Python _search_packages：仅 packages == null（未提供）走默认；
        // 显式空列表是配置错误，必须 fail-fast（注册成"什么都没有"）。
        List<String> candidates;
        if (prefix != null) {
            candidates = List.of(prefix);
        } else if (packages != null) {
            candidates = List.copyOf(packages);
        } else {
            candidates = List.of(DEFAULT_MODULE_PACKAGE);
        }
        if (candidates.isEmpty()) {
            throw new RegistryError("no module packages configured (AGENT_MODULE_PACKAGES is empty)");
        }
        for (String candidate : candidates) {
            if (!PACKAGE_RE.matcher(candidate).matches()) {
                throw new RegistryError("invalid module package '%s'; must match %s"
                        .formatted(candidate, PACKAGE_RE.pattern()));
            }
        }
        return candidates;
    }

    private static AgentModule loadOne(String name, List<String> packages) {
        validateName(name);
        String className = moduleClassName(name);
        for (String packageName : packages) {
            // 布局约定：模块类在 {包}.{模块名} 子包里（modules.chat.ChatModule），
            // 与 Python 侧 importlib.import_module(f"{pkg}.{name}") 的子包布局一致。
            String full = packageName + "." + name + "." + className;
            Class<?> clazz;
            try {
                clazz = Class.forName(full);
            } catch (ClassNotFoundException e) {
                continue; // 换下一个包（对应 Python 的 ModuleNotFoundError 分支）
            } catch (LinkageError e) {
                // 类存在但加载失败（依赖缺失/初始化异常）：按"导入失败"上报。
                throw new RegistryError("module '%s' failed to load from '%s': %s"
                        .formatted(name, packageName, e.toString()));
            }
            Object instance;
            try {
                instance = clazz.getDeclaredConstructor().newInstance();
            } catch (InstantiationException | IllegalAccessException
                     | InvocationTargetException | NoSuchMethodException e) {
                throw new RegistryError("module '%s' failed to instantiate from '%s': %s"
                        .formatted(name, full, e.toString()));
            }
            if (!(instance instanceof AgentModule module)) {
                throw new RegistryError(
                        "module '%s' does not implement AgentModule (%s)".formatted(name, full));
            }
            validateContract(module, name);
            return module;
        }
        String tried = String.join(", ", packages.stream().map(p -> "'" + p + "'").toList());
        throw new RegistryError("unknown module '%s' (not found in packages: %s)".formatted(name, tried));
    }

    /** 命名约定：{@code chat} → 类 {@code chat.ChatModule}（子包 + 大写类名）。 */
    static String moduleClassName(String name) {
        String capitalized = name.substring(0, 1).toUpperCase(Locale.ROOT) + name.substring(1);
        return capitalized + "Module";
    }

    private static void validateName(String name) {
        if (name == null || !MODULE_NAME_RE.matcher(name).matches()) {
            throw new RegistryError(
                    "invalid module name '%s'; must match %s".formatted(name, MODULE_NAME_RE.pattern()));
        }
    }

    private static void validateContract(AgentModule module, String expectedName) {
        String declared = module.name();
        if (!expectedName.equals(declared)) {
            throw new RegistryError(
                    "module loaded as '%s' but declares name '%s'".formatted(expectedName, declared));
        }
        String description = module.description();
        if (description == null || description.isEmpty()) {
            throw new RegistryError(
                    "module '%s' must declare a non-empty 'description'".formatted(expectedName));
        }
        // buildGraph/getTools 是接口方法：能实现本接口即满足（编译期保证）。
    }

    /** .env 兼容垫片：Python 侧默认包名映射到 Java 包。 */
    public static String normalizePackage(String pkg) {
        return PYTHON_LEGACY_PACKAGE.equals(pkg) ? DEFAULT_MODULE_PACKAGE : pkg;
    }
}

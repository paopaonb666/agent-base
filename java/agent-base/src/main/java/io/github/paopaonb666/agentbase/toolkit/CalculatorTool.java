package io.github.paopaonb666.agentbase.toolkit;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;

import io.github.paopaonb666.agentbase.core.tools.AgentTool;

/**
 * {@code calculator} 工具：计算算术表达式（零依赖基础工具之一，对应
 * Python 侧 {@code tools/basic.py} 的 {@code calculator}）。
 *
 * <p>Python 侧用 {@code ast.parse} + AST 白名单求值（绝不 eval——表达式
 * 来自模型输出）；Java 无 AST 入口，这里实现一个等价的递归下降求值器，
 * 支持同一运算符集合：{@code + - * / // % **}、一元 {@code +/-} 与括号，
 * {@code **} 右结合且优先级高于一元负号（{@code -2**2 == -4}）。
 *
 * <p>数值语义（与 Python 对齐的部分）：
 * <ul>
 *   <li>Python {@code int} 任意精度 → {@link BigInteger}；含 float 操作数
 *       时整值自动提升为 {@code double}（对应 Python int→float 混合运算）。</li>
 *   <li>{@code /} 恒为双精度除、{@code //} 向下取整、{@code %} 结果符号
 *       随除数（Python 语义，与 Java 原生 {@code %} 不同）、整数 {@code **}
 *       非负指数保持整数、负指数得 float。</li>
 *   <li>输出格式对齐 {@code _format_number}：浮点值为整值时按整数打印
 *       （{@code 0.5 + 0.25 → 0.75}、{@code 2 ** 10 → 1024}）。</li>
 * </ul>
 *
 * <p>与 Python 的已声明差异：Python float 即 IEEE 双精度，Java double
 * 等价，舍入行为一致；但 Python 的巨大整数转 float 会抛
 * {@code OverflowError}（如 10**400 / 3），Java 会得到 {@code Infinity}
 * 打印为 {@code inf}——极端量级下的差异如实保留。
 */
public final class CalculatorTool implements AgentTool {

    public static final String NAME = "calculator";

    private static final ObjectMapper MAPPER = new ObjectMapper();

    /** 给模型看的说明书（逐字复制 Python 版 docstring）。 */
    private static final String DESCRIPTION = """
            计算算术表达式的值。仅支持四则运算（+ - * / // %）、幂运算（**）
            与括号，例如 "(1 + 2) * 3 ** 2"。不支持变量、函数调用与单位换算。

            Args:
                expression: 算术表达式，如 "144 / 12" 或 "2 ** 10"。
            """.stripTrailing();

    @Override
    public String name() {
        return NAME;
    }

    @Override
    public String description() {
        return DESCRIPTION;
    }

    @Override
    public ToolSpecification specification() {
        return ToolSpecification.builder()
                .name(NAME)
                .description(DESCRIPTION)
                .parameters(JsonObjectSchema.builder()
                        .addStringProperty("expression",
                                "算术表达式，如 \"144 / 12\" 或 \"2 ** 10\"。")
                        .required("expression")
                        .build())
                .build();
    }

    @Override
    public String call(String argsJson, ToolContext ctx) {
        JsonNode node = parseArgs(argsJson).get("expression");
        if (node == null || node.isNull()) {
            throw new IllegalArgumentException("缺少必需参数 expression：算术表达式字符串");
        }
        String expression = node.asText();
        // Python: ast.parse(expression.strip(), mode="eval")；语法错误信息
        // 里带的是原始参数的 repr。
        try {
            return format(evaluate(expression.strip()));
        } catch (Syntax e) {
            throw new IllegalArgumentException(
                    "表达式语法错误：" + PyText.repr(expression) + "（" + e.msg + "）");
        }
    }

    // ── 求值 ────────────────────────────────────────────────────────
    // 值模型：BigInteger（Python int）| Double（Python float）。

    /** 分词期的局部语法错误；最终包装成与 Python 一致的"表达式语法错误"文案。 */
    private static final class Syntax extends RuntimeException {
        final String msg;

        Syntax(String msg) {
            this.msg = msg;
        }
    }

    private static Object evaluate(String expression) {
        List<Token> tokens = new Tokenizer(expression).tokenize();
        Parser parser = new Parser(tokens);
        Object value = parser.parseExpr();
        parser.expectEnd();
        return value;
    }

    private enum TokenType { NUMBER, STRING, CONST, NAME, OP, LPAREN, RPAREN, EOF }

    private record Token(TokenType type, Object value, String text) {
        static final Token EOF = new Token(TokenType.EOF, null, "");
    }

    private static final class Tokenizer {
        private final String s;
        private int i;

        Tokenizer(String s) {
            this.s = s;
        }

        List<Token> tokenize() {
            List<Token> out = new ArrayList<>();
            while (true) {
                skipSpace();
                if (i >= s.length()) {
                    out.add(Token.EOF);
                    return out;
                }
                char c = s.charAt(i);
                if (Character.isDigit(c) || (c == '.' && i + 1 < s.length() && Character.isDigit(s.charAt(i + 1)))) {
                    out.add(number());
                } else if (c == '\'' || c == '"') {
                    out.add(stringLiteral());
                } else if (Character.isJavaIdentifierStart(c)) {
                    out.add(identifier());
                } else if (c == '(') {
                    i++;
                    out.add(new Token(TokenType.LPAREN, null, "("));
                } else if (c == ')') {
                    i++;
                    out.add(new Token(TokenType.RPAREN, null, ")"));
                } else if (c == '*' && peek(1) == '*') {
                    i += 2;
                    out.add(op("**"));
                } else if (c == '/' && peek(1) == '/') {
                    i += 2;
                    out.add(op("//"));
                } else if (c == '+' || c == '-' || c == '*' || c == '/' || c == '%') {
                    i++;
                    out.add(op(String.valueOf(c)));
                } else if (c == '.') {
                    // 属性访问（a.b）：Python ast.parse 通过（合法语法），但
                    // ast.Attribute 落在白名单之外——按"不支持的语法"拒绝，
                    // 而不是报语法错误。
                    i++;
                    out.add(op("."));
                } else {
                    throw new Syntax("invalid syntax");
                }
            }
        }

        private Token op(String op) {
            return new Token(TokenType.OP, null, op);
        }

        private char peek(int ahead) {
            int at = i + ahead;
            return at < s.length() ? s.charAt(at) : '\0';
        }

        private void skipSpace() {
            while (i < s.length() && Character.isWhitespace(s.charAt(i))) {
                i++;
            }
        }

        private void expectNotIdentifierPart() {
            if (i < s.length() && (Character.isJavaIdentifierPart(s.charAt(i)) || Character.isDigit(s.charAt(i)))) {
                throw new Syntax("invalid decimal literal");
            }
        }

        /** 数字字面量：十进制整数/浮点/科学计数法，支持 0x/0o/0b 与下划线分组。 */
        private Token number() {
            int start = i;
            char c = s.charAt(i);
            // 带进制前缀的整数字面量（Python 同样支持）。
            if (c == '0' && i + 1 < s.length()) {
                char p = Character.toLowerCase(s.charAt(i + 1));
                if (p == 'x' || p == 'o' || p == 'b') {
                    int radix = p == 'x' ? 16 : p == 'o' ? 8 : 2;
                    i += 2;
                    int digitsStart = i;
                    while (i < s.length() && (Character.digit(s.charAt(i), radix) >= 0 || s.charAt(i) == '_')) {
                        i++;
                    }
                    if (i == digitsStart) {
                        throw new Syntax("invalid syntax");
                    }
                    String digits = s.substring(digitsStart, i).replace("_", "");
                    expectNotIdentifierPart();
                    return new Token(TokenType.NUMBER, new BigInteger(digits, radix), s.substring(start, i));
                }
            }
            boolean isFloat = false;
            while (i < s.length() && (Character.isDigit(s.charAt(i)) || s.charAt(i) == '_')) {
                i++;
            }
            if (i < s.length() && s.charAt(i) == '.') {
                isFloat = true;
                i++;
                while (i < s.length() && (Character.isDigit(s.charAt(i)) || s.charAt(i) == '_')) {
                    i++;
                }
            }
            if (i < s.length() && (s.charAt(i) == 'e' || s.charAt(i) == 'E')) {
                int save = i;
                i++;
                if (i < s.length() && (s.charAt(i) == '+' || s.charAt(i) == '-')) {
                    i++;
                }
                if (i < s.length() && Character.isDigit(s.charAt(i))) {
                    isFloat = true;
                    while (i < s.length() && (Character.isDigit(s.charAt(i)) || s.charAt(i) == '_')) {
                        i++;
                    }
                } else {
                    i = save; // 不是合法指数，把 'e' 留给后面的字母检查
                }
            }
            expectNotIdentifierPart();
            String literal = s.substring(start, i).replace("_", "");
            if (isFloat) {
                return new Token(TokenType.NUMBER, Double.parseDouble(literal), s.substring(start, i));
            }
            return new Token(TokenType.NUMBER, new BigInteger(literal), s.substring(start, i));
        }

        /** 字符串字面量（Python 里它们是 Constant 节点，求值期报"只支持数字字面量"）。 */
        private Token stringLiteral() {
            char q = s.charAt(i);
            i++;
            StringBuilder sb = new StringBuilder();
            while (i < s.length()) {
                char c = s.charAt(i);
                if (c == '\\') {
                    i++;
                    if (i >= s.length()) {
                        break;
                    }
                    char e = s.charAt(i);
                    sb.append(switch (e) {
                        case 'n' -> '\n';
                        case 't' -> '\t';
                        case 'r' -> '\r';
                        default -> e;
                    });
                    i++;
                } else if (c == q) {
                    i++;
                    return new Token(TokenType.STRING, sb.toString(), sb.toString());
                } else {
                    sb.append(c);
                    i++;
                }
            }
            throw new Syntax("unterminated string literal (detected at line 1)");
        }

        private Token identifier() {
            int start = i;
            while (i < s.length() && Character.isJavaIdentifierPart(s.charAt(i))) {
                i++;
            }
            String name = s.substring(start, i);
            if (name.equals("True") || name.equals("False") || name.equals("None")) {
                return new Token(TokenType.CONST, name, name);
            }
            return new Token(TokenType.NAME, name, name);
        }
    }

    /**
     * 递归下降求值器。文法（优先级从低到高）：
     * <pre>
     * expr  := term (('+'|'-') term)*
     * term  := unary (('*'|'/'|'//'|'%') unary)*
     * unary := ('+'|'-') unary | power
     * power := atom ('**' unary)?        # 右结合；右目允许一元负号
     * atom  := NUMBER | '(' expr ')'
     * </pre>
     * 边解析边求值（左到右），与 Python AST 求值的报错顺序一致。
     */
    private static final class Parser {
        private final List<Token> tokens;
        private int pos;
        private static final BigInteger BI_ZERO = BigInteger.ZERO;

        Parser(List<Token> tokens) {
            this.tokens = tokens;
        }

        Object parseExpr() {
            Object value = parseTerm();
            while (matchOp("+") || matchOp("-")) {
                String op = previous().text();
                Object right = parseTerm();
                value = op.equals("+") ? add(value, right) : sub(value, right);
            }
            return value;
        }

        private Object parseTerm() {
            Object value = parseUnary();
            while (matchOp("*") || matchOp("/") || matchOp("//") || matchOp("%")) {
                String op = previous().text();
                Object right = parseUnary();
                value = switch (op) {
                    case "*" -> mul(value, right);
                    case "/" -> truediv(value, right);
                    case "//" -> floordiv(value, right);
                    default -> mod(value, right);
                };
            }
            return value;
        }

        private Object parseUnary() {
            if (matchOp("+")) {
                // operator.pos：数值不变（Python 仍校验操作数类型，这里必然是数）。
                return parseUnary();
            }
            if (matchOp("-")) {
                return negate(parseUnary());
            }
            return parsePower();
        }

        private Object parsePower() {
            Object base = parseAtom();
            if (matchOp("**")) {
                return pow(base, parseUnary());
            }
            return base;
        }

        private Object parseAtom() {
            Token t = next();
            if (t.type() == TokenType.OP && ".".equals(t.text())) {
                throw new IllegalArgumentException("表达式包含不支持的语法；仅支持 + - * / // % ** 与括号");
            }
            switch (t.type()) {
                case NUMBER -> {
                    return t.value();
                }
                case CONST -> {
                    // bool/None 字面量：bool 是 int 的子类，但 True/False 不是算术，一律拒绝。
                    throw new IllegalArgumentException("只支持数字字面量，收到了 " + t.text());
                }
                case STRING -> {
                    throw new IllegalArgumentException("只支持数字字面量，收到了 " + PyText.repr(t.text()));
                }
                case NAME -> {
                    // 名字/调用（如 __import__(...)）在 Python 侧落在 AST 白名单之外。
                    throw new IllegalArgumentException("表达式包含不支持的语法；仅支持 + - * / // % ** 与括号");
                }
                case LPAREN -> {
                    Object value = parseExpr();
                    if (!match(TokenType.RPAREN)) {
                        throw new Syntax("invalid syntax");
                    }
                    return value;
                }
                case EOF -> throw new Syntax("invalid syntax");
                default -> throw new Syntax("invalid syntax");
            }
        }

        void expectEnd() {
            if (peek().type() == TokenType.EOF) {
                return;
            }
            // Python 里 "1, 2" 解析成 Tuple、"a.b" 解析成 Attribute——都在
            // AST 白名单之外，文案与名字/调用一致；其余残余 token 是真正的
            // 语法错误。
            if (peek().type() == TokenType.OP && (peek().text().equals(",") || peek().text().equals("."))) {
                throw new IllegalArgumentException("表达式包含不支持的语法；仅支持 + - * / // % ** 与括号");
            }
            throw new Syntax("invalid syntax");
        }

        private Token peek() {
            return tokens.get(pos);
        }

        private Token next() {
            return tokens.get(pos++);
        }

        private Token previous() {
            return tokens.get(pos - 1);
        }

        private boolean match(TokenType type) {
            if (peek().type() == type) {
                pos++;
                return true;
            }
            return false;
        }

        private boolean matchOp(String op) {
            if (peek().type() == TokenType.OP && peek().text().equals(op)) {
                pos++;
                return true;
            }
            return false;
        }
    }

    // ── Python 语义的运算 ────────────────────────────────────────────

    private static Object add(Object a, Object b) {
        if (a instanceof BigInteger x && b instanceof BigInteger y) {
            return x.add(y);
        }
        return dbl(a) + dbl(b);
    }

    private static Object sub(Object a, Object b) {
        if (a instanceof BigInteger x && b instanceof BigInteger y) {
            return x.subtract(y);
        }
        return dbl(a) - dbl(b);
    }

    private static Object mul(Object a, Object b) {
        if (a instanceof BigInteger x && b instanceof BigInteger y) {
            return x.multiply(y);
        }
        return dbl(a) * dbl(b);
    }

    private static Object truediv(Object a, Object b) {
        // Python 先查 right == 0（覆盖 0 与 0.0），文案与 Python 一致。
        if (isZero(b)) {
            throw new IllegalArgumentException("除数为零");
        }
        return dbl(a) / dbl(b);
    }

    private static Object floordiv(Object a, Object b) {
        if (a instanceof BigInteger x && b instanceof BigInteger y) {
            if (y.signum() == 0) {
                throw new ArithmeticException("integer division or modulo by zero");
            }
            // BigInteger 没有向下取整除法，用余数符号校正实现 floor 语义。
            BigInteger rem = x.remainder(y);
            if (rem.signum() != 0 && rem.signum() != y.signum()) {
                rem = rem.add(y);
            }
            return x.subtract(rem).divide(y);
        }
        double d1 = dbl(b);
        if (d1 == 0.0) {
            throw new ArithmeticException("float floor division by zero");
        }
        return Math.floor(dbl(a) / d1);
    }

    private static Object mod(Object a, Object b) {
        if (a instanceof BigInteger x && b instanceof BigInteger y) {
            if (y.signum() == 0) {
                throw new ArithmeticException("integer division or modulo by zero");
            }
            // Python 的 % 结果符号随除数（与 Java 原生 % 不同）。
            BigInteger r = x.remainder(y);
            if (r.signum() != 0 && r.signum() != y.signum()) {
                r = r.add(y);
            }
            return r;
        }
        double d1 = dbl(b);
        if (d1 == 0.0) {
            throw new ArithmeticException("float modulo");
        }
        double d0 = dbl(a);
        return d0 - Math.floor(d0 / d1) * d1;
    }

    private static Object pow(Object a, Object b) {
        if (a instanceof BigInteger x && b instanceof BigInteger y) {
            if (y.signum() >= 0) {
                if (y.bitLength() > 31) {
                    // Python 会尝试构造任意大的整数直至 MemoryError；这里
                    // 直接拒绝（超出 int 指数范围的表达式没有实际用途）。
                    throw new ArithmeticException("幂运算指数过大，已拒绝求值");
                }
                return x.pow(y.intValueExact());
            }
            // 负指数：Python 返回 float（2 ** -1 == 0.5）。
            if (x.signum() == 0) {
                throw new ArithmeticException("0.0 cannot be raised to a negative power");
            }
            return Math.pow(x.doubleValue(), y.doubleValue());
        }
        double d0 = dbl(a);
        double d1 = dbl(b);
        if (d0 == 0.0 && d1 < 0) {
            // 文案与 Python 的 ZeroDivisionError 一致。
            throw new ArithmeticException("0.0 cannot be raised to a negative power");
        }
        return Math.pow(d0, d1);
    }

    private static Object negate(Object v) {
        if (v instanceof BigInteger x) {
            return x.negate();
        }
        return -dbl(v);
    }

    private static boolean isZero(Object v) {
        if (v instanceof BigInteger x) {
            return x.signum() == 0;
        }
        return dbl(v) == 0.0;
    }

    private static double dbl(Object v) {
        return v instanceof BigInteger x ? x.doubleValue() : (Double) v;
    }

    /** 对齐 Python {@code _format_number}：浮点整值打印为整数。 */
    static String format(Object value) {
        if (value instanceof BigInteger x) {
            return x.toString();
        }
        double d = (Double) value;
        if (d == Math.rint(d)) {
            // int(2.0e20) 与 Python 的 str(int(float)) 一致（按最短十进制
            // 表示还原整值，BigDecimal.valueOf 走 Double.toString）。
            return BigDecimal.valueOf(d).toBigInteger().toString();
        }
        return PyText.floatRepr(d);
    }

    private static JsonNode parseArgs(String argsJson) {
        try {
            return new ObjectMapper().readTree(argsJson);
        } catch (Exception e) {
            throw new IllegalArgumentException("工具参数不是合法 JSON：" + e.getMessage());
        }
    }
}

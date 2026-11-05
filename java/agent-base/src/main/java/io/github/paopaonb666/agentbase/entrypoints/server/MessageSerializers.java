package io.github.paopaonb666.agentbase.entrypoints.server;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.Content;
import dev.langchain4j.data.message.ImageContent;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * HTTP 层的序列化与内容嗅探：消息 ↔ 前端结构、图片类型。
 *
 * <p>对应 Python 侧 {@code entrypoints/server/serializers.py}，逐字段对齐。
 */
public final class MessageSerializers {

    private MessageSerializers() {}

    // 图片附件（多模态对话）：按 magic bytes 嗅探真实类型——扩展名可伪装，
    // 内容头不会。webp 的 RIFF/WEBP 头单独检查。
    private static final byte[][] SIGNATURES = {
            {0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A}, // \x89PNG\r\n\x1a\n → 见下
    };
    private static final String[] SIGNATURE_MIMES = {"image/png"};

    /** 图片扩展名集合。 */
    public static final Set<String> IMAGE_EXTENSIONS = Set.of("png", "jpg", "jpeg", "webp", "gif");
    /** filestore 中图片附件的归一化 format 值（mime 去前缀）。 */
    public static final Set<String> IMAGE_STORED_FORMATS = Set.of("png", "jpeg", "webp", "gif");

    /** 按 magic bytes 识别图片真实类型；无法识别返回 null。 */
    public static String sniffImageMime(byte[] data) {
        if (data == null) {
            return null;
        }
        if (startsWith(data, new byte[]{(byte) 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A})) {
            return "image/png";
        }
        if (startsWith(data, new byte[]{(byte) 0xFF, (byte) 0xD8, (byte) 0xFF})) {
            return "image/jpeg";
        }
        if (startsWith(data, "GIF87a".getBytes(java.nio.charset.StandardCharsets.US_ASCII))) {
            return "image/gif";
        }
        if (startsWith(data, "GIF89a".getBytes(java.nio.charset.StandardCharsets.US_ASCII))) {
            return "image/gif";
        }
        if (data.length >= 12
                && startsWith(data, "RIFF".getBytes(java.nio.charset.StandardCharsets.US_ASCII))
                && data[8] == 'W' && data[9] == 'E' && data[10] == 'B' && data[11] == 'P') {
            return "image/webp";
        }
        return null;
    }

    private static boolean startsWith(byte[] data, byte[] prefix) {
        if (data.length < prefix.length) {
            return false;
        }
        for (int i = 0; i < prefix.length; i++) {
            if (data[i] != prefix[i]) {
                return false;
            }
        }
        return true;
    }

    /** 提取消息 content 里的纯文本（多段 content 拼接 text 段）。 */
    public static String extractText(ChatMessage message) {
        if (message instanceof UserMessage user) {
            StringBuilder sb = new StringBuilder();
            for (Content content : user.contents()) {
                if (content instanceof TextContent text) {
                    sb.append(text.text());
                }
            }
            return sb.toString();
        }
        if (message instanceof SystemMessage system) {
            return system.text();
        }
        if (message instanceof AiMessage ai) {
            return ai.text() == null ? "" : ai.text();
        }
        if (message instanceof ToolExecutionResultMessage tool) {
            return tool.text();
        }
        return String.valueOf(message);
    }

    /**
     * 把一条消息序列化为前端可渲染的简单结构；返回 {@code null} 表示这条
     * 消息不该显示（如 system 注入块）。工具调用可观测（M5）：AI 消息携带
     * {@code tool_calls}（id/name/args），tool 消息携带
     * {@code tool_call_id}/{@code status}——前端按 id 配对出完整调用面板。
     * 仅含 tool_calls 的 AI 消息也保留（它是调用参数的唯一持久化位置）。
     *
     * <p>{@code attachmentMeta}：human 消息的附件元数据（Python 侧存在
     * message.additional_kwargs；Java 侧存于图状态 extras 的
     * {@code attachments} 键，按消息索引对齐——序列化行为一致）。
     */
    public static Map<String, Object> serializeMessage(ChatMessage message, Object attachmentMeta) {
        String text = extractText(message);
        if (message instanceof SystemMessage) {
            // 附件注入的系统消息：内容只在当轮给模型，历史回放由 human 消息
            // 上的附件元数据承载——blob 永不上屏。
            return null;
        }
        if (message instanceof UserMessage) {
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("role", "human");
            out.put("content", text);
            if (attachmentMeta instanceof List<?> list && !list.isEmpty()) {
                out.put("attachments", attachmentMeta);
            }
            return out;
        }
        if (message instanceof AiMessage ai) {
            List<Map<String, Object>> toolCalls = new ArrayList<>();
            if (ai.hasToolExecutionRequests()) {
                for (ToolExecutionRequest tc : ai.toolExecutionRequests()) {
                    Map<String, Object> call = new LinkedHashMap<>();
                    call.put("id", tc.id() == null ? "" : tc.id());
                    call.put("name", tc.name() == null ? "" : tc.name());
                    call.put("args", parseArgs(tc.arguments()));
                    toolCalls.add(call);
                }
            }
            if ((text == null || text.isEmpty()) && toolCalls.isEmpty()) {
                return null; // 既无文本也无调用的空 AI 消息，没有可展示内容
            }
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("role", "assistant");
            out.put("content", text);
            if (!toolCalls.isEmpty()) {
                out.put("tool_calls", toolCalls);
            }
            return out;
        }
        if (message instanceof ToolExecutionResultMessage tool) {
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("role", "tool");
            out.put("name", tool.toolName() == null ? "" : tool.toolName());
            out.put("content", text);
            out.put("tool_call_id", tool.id() == null ? "" : tool.id());
            out.put("status", "ok");
            return out;
        }
        return null;
    }

    /** 工具调用参数 JSON → Map（畸形参数降级为空 Map，对应 Python 语义）。 */
    private static Map<String, Object> parseArgs(String arguments) {
        try {
            Object parsed = new com.fasterxml.jackson.databind.ObjectMapper()
                    .readValue(arguments == null ? "{}" : arguments, Object.class);
            if (parsed instanceof Map<?, ?> map) {
                @SuppressWarnings("unchecked")
                Map<String, Object> typed = (Map<String, Object>) map;
                return typed;
            }
            return new LinkedHashMap<>();
        } catch (Exception e) {
            return new LinkedHashMap<>();
        }
    }
}

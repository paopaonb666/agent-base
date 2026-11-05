package io.github.paopaonb666.agentbase.entrypoints.server;

import dev.langchain4j.data.message.Content;
import dev.langchain4j.data.message.ImageContent;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.UserMessage;
import io.github.paopaonb666.agentbase.core.context.ContextEngine;

import java.util.ArrayList;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 附件（M4b/M4c）的 invoke 侧装配：文本注入块、图片多模态 content
 * blocks、附件元数据。对应 Python 侧 routes/agents.py invoke 里的附件段。
 */
public final class Attachments {

    private Attachments() {}

    /**
     * 文本附件的注入块（system 消息，以约定的 ATTACHMENT_CONTEXT_PREFIX
     * 开头——上下文引擎靠它做注入去重）。
     */
    public static String systemBlock(List<Map<String, Object>> docInfos) {
        List<String> blocks = new ArrayList<>();
        for (Map<String, Object> info : docInfos) {
            String filename = String.valueOf(info.getOrDefault("filename", ""));
            String format = String.valueOf(info.getOrDefault("format", ""));
            Object pagesObj = info.get("pages");
            String pages = pagesObj != null && !String.valueOf(pagesObj).isEmpty()
                    ? pagesObj + " 页，" : "";
            Object textLenObj = info.get("text_len");
            String textLen = String.valueOf(textLenObj == null ? 0 : textLenObj);
            String extracted = String.valueOf(info.getOrDefault("extracted_text", ""));
            if (!extracted.strip().isEmpty()) {
                blocks.add("[附件文件：%s（%s，%s%s 字符）]\n%s".formatted(filename, format, pages, textLen, extracted));
            } else {
                Object warning = info.get("warning");
                String note = warning == null || String.valueOf(warning).isEmpty()
                        ? "无法提取文本" : String.valueOf(warning);
                blocks.add("[附件文件：%s]（%s）".formatted(filename, note));
            }
        }
        if (blocks.isEmpty()) {
            return null;
        }
        return ContextEngine.ATTACHMENT_CONTEXT_PREFIX + "，供回答时参考：\n\n"
                + String.join("\n\n", blocks);
    }

    /**
     * 带图片附件的 human 消息（多模态 content blocks：文本 + base64 图，
     * vision 模型直接看图）。
     */
    public static UserMessage humanMessage(String text, List<Map<String, Object>> imageInfos) {
        if (imageInfos == null || imageInfos.isEmpty()) {
            return UserMessage.from(text);
        }
        List<Content> contents = new ArrayList<>();
        contents.add(TextContent.from(text));
        for (Map<String, Object> img : imageInfos) {
            byte[] data = (byte[]) img.get("content");
            String format = String.valueOf(img.getOrDefault("format", "png"));
            String encoded = Base64.getEncoder().encodeToString(data == null ? new byte[0] : data);
            contents.add(ImageContent.from(encoded, "image/" + format));
        }
        return UserMessage.from(contents);
    }

    /** 附件元数据列表（info.meta() 视图，落到图状态 extras 的 attachments 键）。 */
    public static List<Map<String, Object>> metaList(List<Map<String, Object>> infos) {
        return new ArrayList<>(infos);
    }

    /** 元数据 Map 构造（filestore meta 的子集承载：无字节/全文）。 */
    public static Map<String, Object> metaOf(String fileId, String filename, String format,
                                             Integer pages, boolean truncated, int textLen,
                                             String warning) {
        Map<String, Object> meta = new LinkedHashMap<>();
        meta.put("file_id", fileId);
        meta.put("filename", filename);
        meta.put("format", format);
        meta.put("pages", pages);
        meta.put("truncated", truncated);
        meta.put("text_len", textLen);
        if (warning != null && !warning.isEmpty()) {
            meta.put("warning", warning);
        }
        return meta;
    }
}

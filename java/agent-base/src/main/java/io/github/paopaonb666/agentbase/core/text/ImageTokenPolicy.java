package io.github.paopaonb666.agentbase.core.text;

/**
 * 图片块 token 估价（#130 ← 旧 #20）：vision 请求的真实图片成本数千至上万
 * token，此前预算路径按 "[image]" 占位走文本 tokenizer（≈2 token/图；Python
 * 侧对图片 dict 更计 0）——多模态轮次的修剪触发点严重失真。现状持久化只存
 * base64+mime、无分辨率信息，按每图固定值起步；provider 侧的图片 token 已
 * 折进 prompt_tokens 如实计费（usage 无拆分字段），本类只服务预算/修剪面，
 * 不是计量源。
 */
public final class ImageTokenPolicy {

    /**
     * 每张图片的固定 token 估价：OpenAI 视觉低清 ≈85、高清 ≈1105（自动档
     * 介于两者），取中档偏上覆盖常见截图尺寸。
     */
    public static final int TOKENS_PER_IMAGE = 1500;

    private ImageTokenPolicy() {
    }

    /** 单图估价（mime 保留分层入口：低清/高清分档待有分辨率信息后再启用）。 */
    public static int estimate(String mime) {
        return TOKENS_PER_IMAGE;
    }
}

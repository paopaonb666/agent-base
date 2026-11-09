package io.github.paopaonb666.agentbase.core.text;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** #130：图片块固定 token 估价——预算/修剪面的计量口径锁定。 */
class ImageTokenPolicyTest {

    @Test
    void estimateIsFixedRegardlessOfMime() {
        assertEquals(ImageTokenPolicy.TOKENS_PER_IMAGE,
                ImageTokenPolicy.estimate("image/png"));
        assertEquals(ImageTokenPolicy.TOKENS_PER_IMAGE,
                ImageTokenPolicy.estimate("image/jpeg"));
        assertEquals(ImageTokenPolicy.TOKENS_PER_IMAGE,
                ImageTokenPolicy.estimate(null));
    }

    @Test
    void estimateIsInRealVisionCostMagnitude() {
        // 真实 vision 单图数千 token 量级（OpenAI 低清≈85/高清≈1105，自动档
        // 介于两者）；估价必须显著高于文本占位的 ≈2 token，且不高于高清
        // 极端值的一个数量级。
        assertTrue(ImageTokenPolicy.TOKENS_PER_IMAGE >= 1000,
                "估价过低则多模态修剪触发点失真（#130 动机）");
        assertTrue(ImageTokenPolicy.TOKENS_PER_IMAGE <= 10_000,
                "估价过高会过度修剪多模态历史");
    }
}

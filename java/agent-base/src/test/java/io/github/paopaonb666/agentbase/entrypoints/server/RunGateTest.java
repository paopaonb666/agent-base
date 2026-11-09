package io.github.paopaonb666.agentbase.entrypoints.server;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** #79：全局 run 并发闸门——容量、饱和拒绝、配对释放、不限形态。 */
class RunGateTest {

    @Test
    void admitsUpToCapacityThenRejectsAndReleases() {
        RunGate gate = new RunGate(2);
        assertTrue(gate.tryEnter());
        assertTrue(gate.tryEnter());
        assertEquals(2, gate.inFlight());
        // 饱和：快速失败并计数（不排队）。
        assertFalse(gate.tryEnter());
        assertEquals(1, gate.rejectionCount());
        // 配对释放后额度恢复。
        gate.exit();
        assertEquals(1, gate.inFlight());
        assertTrue(gate.tryEnter());
        gate.exit();
        gate.exit();
        assertEquals(0, gate.inFlight());
        assertTrue(gate.tryEnter());
    }

    @Test
    void zeroMeansUnlimited() {
        RunGate gate = new RunGate(0);
        for (int i = 0; i < 100; i++) {
            assertTrue(gate.tryEnter());
        }
        assertEquals(0, gate.rejectionCount());
        // 不限形态 exit 是 no-op（不能误放别人的许可）。
        gate.exit();
        assertTrue(gate.tryEnter());
    }
}

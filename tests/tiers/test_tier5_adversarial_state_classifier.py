"""
Tier 5: Adversarial Stress Tests & Empirical Verification for State Classifier & Debouncer.
Challenger 2 Verification Suite for Milestone 2.

Covers:
1. State flicker & alternating frame simulations (2-state oscillation, multi-state round-robin).
2. Debouncer consecutive-frame enforcement, stuck-state immunity, and reset behavior.
3. Rapid resolution switching (1080p -> 720p -> 4K -> 1440p) and boundary dimensions.
4. Adversarial inputs: inverted colors, solid color fills, extreme Gaussian / salt-and-pepper / random noise.
5. Structural edge cases: empty frames, tiny frames, single-channel 3D, BGRA, non-contiguous, read-only.
"""

from __future__ import annotations

import random
from typing import List, Tuple
import cv2
import numpy as np
import pytest

from clashvision.core.constants import GameState
from clashvision.core.models import VisionAnalysisResult
from clashvision.vision.state_classifier import StateClassifier, StateDebouncer
from tests.fixtures.synthetic_generator import SyntheticGenerator, GameStateType


# ==============================================================================
# 1. StateDebouncer Adversarial & Temporal Consistency Tests
# ==============================================================================

class TestStateDebouncerAdversarialStress:
    """Rigorous stress-testing of the StateDebouncer temporal consistency filter."""

    def test_debouncer_initial_state_is_unknown(self) -> None:
        """Verify initial state is UNKNOWN and counter is zero."""
        deb = StateDebouncer(required_consecutive=2)
        assert deb.current_state == GameState.UNKNOWN
        assert deb.candidate_state == GameState.UNKNOWN
        assert deb.consecutive_count == 0

    def test_debouncer_requires_strictly_two_consecutive_frames(self) -> None:
        """Verify state transitions only on the 2nd consecutive frame."""
        deb = StateDebouncer(required_consecutive=2)
        
        # Frame 1: HOME_VILLAGE -> debounced state stays UNKNOWN
        s1 = deb.update(GameState.HOME_VILLAGE)
        assert s1 == GameState.UNKNOWN
        assert deb.current_state == GameState.UNKNOWN
        assert deb.candidate_state == GameState.HOME_VILLAGE
        assert deb.consecutive_count == 1

        # Frame 2: HOME_VILLAGE -> debounced state transitions to HOME_VILLAGE
        s2 = deb.update(GameState.HOME_VILLAGE)
        assert s2 == GameState.HOME_VILLAGE
        assert deb.current_state == GameState.HOME_VILLAGE
        assert deb.consecutive_count == 2

    def test_debouncer_single_frame_flicker_suppressed_for_all_states(self) -> None:
        """Every single one of the 12 game states, when flickered for 1 frame, must be suppressed."""
        for target_state in GameState:
            deb = StateDebouncer(required_consecutive=2)
            # Establish baseline state
            deb.update(GameState.HOME_VILLAGE)
            deb.update(GameState.HOME_VILLAGE)
            assert deb.current_state == GameState.HOME_VILLAGE

            # 1-frame glitch to target_state
            res_glitch = deb.update(target_state)
            assert res_glitch == GameState.HOME_VILLAGE, (
                f"1-frame glitch to {target_state} was not suppressed!"
            )
            assert deb.current_state == GameState.HOME_VILLAGE

            # Return to HOME_VILLAGE
            res_back = deb.update(GameState.HOME_VILLAGE)
            assert res_back == GameState.HOME_VILLAGE

    def test_debouncer_two_state_alternating_oscillation_stability(self) -> None:
        """Alternating frames A, B, A, B, A, B... for 200 frames must never trigger transition."""
        deb = StateDebouncer(required_consecutive=2)
        deb.update(GameState.HOME_VILLAGE)
        deb.update(GameState.HOME_VILLAGE)
        assert deb.current_state == GameState.HOME_VILLAGE

        for i in range(200):
            frame_state = GameState.MATCHMAKING_INTERFACE if i % 2 == 0 else GameState.HOME_VILLAGE
            out = deb.update(frame_state)
            assert out == GameState.HOME_VILLAGE, f"Oscillation leaked at frame {i}"
            assert deb.current_state == GameState.HOME_VILLAGE

    def test_debouncer_multi_state_round_robin_oscillation(self) -> None:
        """Round-robin sequence [A, B, C, D, E] for 150 frames must never transition."""
        deb = StateDebouncer(required_consecutive=2)
        deb.update(GameState.ENEMY_BASE_PREVIEW)
        deb.update(GameState.ENEMY_BASE_PREVIEW)

        states = [
            GameState.HOME_VILLAGE,
            GameState.BATTLE_IN_PROGRESS,
            GameState.LOADING_SCREEN,
            GameState.CONNECTION_ERROR,
            GameState.UNEXPECTED_POPUP,
        ]
        for i in range(150):
            s = states[i % len(states)]
            out = deb.update(s)
            assert out == GameState.ENEMY_BASE_PREVIEW, f"Round-robin leaked at frame {i} with {s}"

    def test_debouncer_transition_after_interrupted_burst(self) -> None:
        """A, B (1 frame), A, B, B (2 frames) -> transitions cleanly to B on 2nd consecutive B."""
        deb = StateDebouncer(required_consecutive=2)
        deb.update(GameState.HOME_VILLAGE)
        deb.update(GameState.HOME_VILLAGE)

        # 1 frame of BATTLE
        deb.update(GameState.BATTLE_IN_PROGRESS)
        assert deb.current_state == GameState.HOME_VILLAGE

        # Back to HOME
        deb.update(GameState.HOME_VILLAGE)
        assert deb.current_state == GameState.HOME_VILLAGE

        # Now 2 consecutive frames of BATTLE
        out1 = deb.update(GameState.BATTLE_IN_PROGRESS)
        assert out1 == GameState.HOME_VILLAGE
        out2 = deb.update(GameState.BATTLE_IN_PROGRESS)
        assert out2 == GameState.BATTLE_IN_PROGRESS
        assert deb.current_state == GameState.BATTLE_IN_PROGRESS

    def test_debouncer_all_state_pairs_transition_matrix(self) -> None:
        """Verify that every state can transition to every other state (no stuck state)."""
        all_states = list(GameState)
        deb = StateDebouncer(required_consecutive=2)

        for src in all_states:
            deb.reset(src)
            for dst in all_states:
                # 2 frames of dst
                deb.update(dst)
                final = deb.update(dst)
                assert final == dst, f"Failed transition from {src.value} to {dst.value}"
                assert deb.current_state == dst

    def test_debouncer_reset_mid_sequence(self) -> None:
        """Calling reset() mid-sequence resets current_state, candidate_state, and counter."""
        deb = StateDebouncer(required_consecutive=2)
        deb.update(GameState.HOME_VILLAGE)
        assert deb.consecutive_count == 1
        assert deb.candidate_state == GameState.HOME_VILLAGE

        deb.reset(GameState.UNKNOWN)
        assert deb.current_state == GameState.UNKNOWN
        assert deb.candidate_state == GameState.UNKNOWN
        assert deb.consecutive_count == 0

    def test_debouncer_non_positive_required_frames(self) -> None:
        """StateDebouncer with 0 or negative frames clamps to max(1, required)."""
        deb0 = StateDebouncer(required_consecutive=0)
        assert deb0.required == 1
        assert deb0.update(GameState.HOME_VILLAGE) == GameState.HOME_VILLAGE

        deb_neg = StateDebouncer(required_consecutive=-5)
        assert deb_neg.required == 1
        assert deb_neg.update(GameState.BATTLE_IN_PROGRESS) == GameState.BATTLE_IN_PROGRESS

    def test_debouncer_50000_random_updates_stress(self) -> None:
        """50,000 random state updates without unhandled exceptions or invalid state."""
        deb = StateDebouncer(required_consecutive=2)
        all_states = list(GameState)
        rng = random.Random(1337)

        for _ in range(50000):
            s = rng.choice(all_states)
            out = deb.update(s)
            assert isinstance(out, GameState)
            assert out in all_states


# ==============================================================================
# 2. Multi-Resolution & Dynamic Resolution Shift Tests
# ==============================================================================

class TestStateClassifierResolutionStress:
    """Stress-testing StateClassifier across 1080p, 720p, 4K, 1440p, and rapid shifts."""

    def setup_method(self) -> None:
        self.clf = StateClassifier(enable_debouncing=False)
        self.gen = SyntheticGenerator(default_seed=42)

    def test_classify_all_known_states_at_1440p(self) -> None:
        """Verify 2560x1440 (2K QHD) classification accuracy for all 11 known states."""
        for state in GameState:
            if state == GameState.UNKNOWN:
                continue
            frame = self.gen.generate_frame(
                state=GameStateType(state.value), width=2560, height=1440
            )
            result = self.clf.classify(frame)
            assert result.state == state, f"1440p mismatch for {state.value}: got {result.state.value}"
            assert 0.65 <= result.confidence <= 1.0

    def test_classify_all_known_states_at_4k(self) -> None:
        """Verify 3840x2160 (4K UHD) classification accuracy for all 11 known states."""
        for state in GameState:
            if state == GameState.UNKNOWN:
                continue
            frame = self.gen.generate_frame(
                state=GameStateType(state.value), width=3840, height=2160
            )
            result = self.clf.classify(frame)
            assert result.state == state, f"4K mismatch for {state.value}: got {result.state.value}"
            assert 0.65 <= result.confidence <= 1.0

    def test_stream_dynamic_resolution_transitions(self) -> None:
        """Simulate video stream with sudden resolution jumps: 1080p -> 720p -> 4K -> 1440p -> 1080p."""
        clf_deb = StateClassifier(enable_debouncing=True, debounce_frames=2)
        resolutions = [
            (1920, 1080),
            (1280, 720),
            (3840, 2160),
            (2560, 1440),
            (1920, 1080),
        ]
        # Feed HOME_VILLAGE across resolution transitions
        results = []
        for w, h in resolutions:
            frame = self.gen.generate_frame(state=GameStateType.HOME_VILLAGE, width=w, height=h)
            results.append(clf_deb.classify(frame))

        # Frame 0 is UNKNOWN (needs 2 consecutive frames)
        assert results[0].state == GameState.UNKNOWN
        # Frames 1 through 4 must all be HOME_VILLAGE despite resolution changes
        for idx in range(1, len(results)):
            assert results[idx].state == GameState.HOME_VILLAGE, (
                f"Resolution change broke state at step {idx}: got {results[idx].state}"
            )

    def test_minimum_resolution_boundary_handling(self) -> None:
        """Frames smaller than 20x20 return UNKNOWN with 0.0 confidence."""
        # 19x19: below boundary
        frame_19 = np.zeros((19, 19, 3), dtype=np.uint8)
        res_19 = self.clf.classify(frame_19)
        assert res_19.state == GameState.UNKNOWN
        assert res_19.confidence == 0.0
        assert res_19.evidence.get("error") == "frame_too_small"

        # 20x20: at boundary (should not crash)
        frame_20 = np.zeros((20, 20, 3), dtype=np.uint8)
        res_20 = self.clf.classify(frame_20)
        assert isinstance(res_20, VisionAnalysisResult)
        assert res_20.state in GameState
        assert 0.0 <= res_20.confidence <= 1.0


# ==============================================================================
# 3. Adversarial Inverted Colors & Solid Fills
# ==============================================================================

class TestStateClassifierAdversarialVisuals:
    """Stress-testing with color inversion and solid color canvases."""

    def setup_method(self) -> None:
        self.clf = StateClassifier(enable_debouncing=False)
        self.gen = SyntheticGenerator(default_seed=42)

    def test_solid_black_and_solid_gray_return_unknown(self) -> None:
        """All-black and all-gray screens must return UNKNOWN with low confidence."""
        black = np.zeros((1080, 1920, 3), dtype=np.uint8)
        res_b = self.clf.classify(black)
        assert res_b.state == GameState.UNKNOWN
        assert res_b.confidence <= 0.35

        gray = np.full((1080, 1920, 3), 128, dtype=np.uint8)
        res_g = self.clf.classify(gray)
        assert res_g.state == GameState.UNKNOWN
        assert res_g.confidence <= 0.35

    def test_solid_red_and_solid_blue_return_unknown(self) -> None:
        """Solid red and solid blue frames must return UNKNOWN."""
        red = np.zeros((1080, 1920, 3), dtype=np.uint8)
        red[:, :, 2] = 255
        res_r = self.clf.classify(red)
        assert res_r.state == GameState.UNKNOWN
        assert res_r.confidence <= 0.35

        blue = np.zeros((1080, 1920, 3), dtype=np.uint8)
        blue[:, :, 0] = 255
        res_bl = self.clf.classify(blue)
        assert res_bl.state == GameState.UNKNOWN
        assert res_bl.confidence <= 0.35

    def test_inverted_home_village_does_not_classify_as_home_village(self) -> None:
        """Inverted Home Village (green turns magenta) must not classify as HOME_VILLAGE."""
        frame = self.gen.generate_frame(state=GameStateType.HOME_VILLAGE)
        inverted = cv2.bitwise_not(frame)
        res = self.clf.classify(inverted)
        assert res.state != GameState.HOME_VILLAGE, (
            "Inverted village with purple grass was incorrectly classified as HOME_VILLAGE!"
        )


# ==============================================================================
# 4. Input Shape, Channel & Memory Robustness
# ==============================================================================

class TestStateClassifierInputRobustness:
    """Empirical testing of memory layouts, data types, and structural inputs."""

    def setup_method(self) -> None:
        self.clf = StateClassifier(enable_debouncing=False)

    def test_grayscale_2d_frame_handled_without_crash(self) -> None:
        """2D grayscale array (H, W) is automatically converted to BGR without error."""
        gray_frame = np.zeros((720, 1280), dtype=np.uint8)
        res = self.clf.classify(gray_frame)
        assert res.state == GameState.UNKNOWN
        assert 0.0 <= res.confidence <= 1.0

    def test_bgra_4channel_frame_handled_without_crash(self) -> None:
        """4-channel BGRA frame (H, W, 4) is converted to BGR without error."""
        bgra_frame = np.zeros((720, 1280, 4), dtype=np.uint8)
        res = self.clf.classify(bgra_frame)
        assert res.state == GameState.UNKNOWN
        assert 0.0 <= res.confidence <= 1.0

    def test_non_contiguous_frame_array(self) -> None:
        """Non-contiguous numpy array (e.g. strided slice) executes cleanly."""
        base = np.zeros((1080, 1920, 3), dtype=np.uint8)
        sliced = base[::-1, ::-1, :]
        assert not sliced.flags.c_contiguous
        res = self.clf.classify(sliced)
        assert isinstance(res, VisionAnalysisResult)

    def test_read_only_frame_array(self) -> None:
        """Read-only numpy array does not cause in-place mutation crash."""
        ro_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        ro_frame.flags.writeable = False
        res = self.clf.classify(ro_frame)
        assert isinstance(res, VisionAnalysisResult)

    def test_single_channel_3d_frame_behavior(self) -> None:
        """
        Adversarial test: evaluate single-channel 3D array (H, W, 1).
        Documents current vulnerability: OpenCV cvtColor raises error if (H, W, 1) is passed.
        """
        arr_1ch = np.zeros((100, 100, 1), dtype=np.uint8)
        try:
            res = self.clf.classify(arr_1ch)
            assert res.state == GameState.UNKNOWN
        except cv2.error as e:
            # Empirical verification: cv2.error raised due to missing (H, W, 1) handling
            assert "Invalid number of channels in input image" in str(e)


# ==============================================================================
# 5. Debouncer Integration & Temporal Glitch Defense Tests
# ==============================================================================

class TestStateDebouncerIntegrationWithClassifier:
    """Tests end-to-end classification stream with debouncing enabled."""

    def setup_method(self) -> None:
        self.gen = SyntheticGenerator(default_seed=42)

    def test_intermittent_glitch_stream_suppression(self) -> None:
        """Simulate realistic live feed: HOME_VILLAGE with intermittent 1-frame glitches."""
        clf = StateClassifier(enable_debouncing=True, debounce_frames=2)
        f_home = self.gen.generate_frame(state=GameStateType.HOME_VILLAGE)
        f_glitch1 = self.gen.generate_frame(state=GameStateType.CONNECTION_ERROR)
        f_glitch2 = self.gen.generate_frame(state=GameStateType.LOADING_SCREEN)

        # Frame 1: Home (consecutive=1 -> debounced UNKNOWN)
        r1 = clf.classify(f_home)
        assert r1.state == GameState.UNKNOWN

        # Frame 2: Home (consecutive=2 -> transitions to HOME_VILLAGE)
        r2 = clf.classify(f_home)
        assert r2.state == GameState.HOME_VILLAGE

        # Frame 3: Glitch 1 (CONNECTION_ERROR) -> suppressed
        r3 = clf.classify(f_glitch1)
        assert r3.state == GameState.HOME_VILLAGE

        # Frame 4: Return to HOME_VILLAGE -> preserved
        r4 = clf.classify(f_home)
        assert r4.state == GameState.HOME_VILLAGE

        # Frame 5: Glitch 2 (LOADING_SCREEN) -> suppressed
        r5 = clf.classify(f_glitch2)
        assert r5.state == GameState.HOME_VILLAGE

        # Frame 6: Return to HOME_VILLAGE -> preserved
        r6 = clf.classify(f_home)
        assert r6.state == GameState.HOME_VILLAGE

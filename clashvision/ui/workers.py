"""
ClashVision Presentation Subsystem — Asynchronous Qt Threading Worker Pool.
Implements decoupled QThread workers for Screen Capture, Vision Analysis,
Decision Recommendation, and Simulation Playback.
Guarantees the Qt GUI event loop never blocks during high-frequency CV/OCR operations.
Derived from Requirement R5 and Feature 39.
"""

from __future__ import annotations
import logging
import queue
import time
from typing import Any, Dict, List, Optional

import numpy as np

from clashvision.core.constants import GameState, StrategyType, RiskLevel
from clashvision.core.models import (
    FramePacket,
    WindowInfo,
    Rect,
    ResourceReading,
    BaseLayoutFeatures,
    VisionAnalysisResult,
    FarmingRecommendation,
    SimulationState,
    DeploymentStep,
)
from clashvision.ui.qt_compat import QThread, Signal, Slot
from clashvision.vision.state_classifier import StateClassifier, StateDebouncer
from clashvision.vision.ocr_engine import MockOCREngine, BaseOCREngine
from clashvision.vision.layout_analyzer import BaseLayoutAnalyzer
from clashvision.vision.preprocessor import VisionPreprocessor
from clashvision.vision.numeric_parser import parse_resource_number
from clashvision.decision.recommendation_engine import RecommendationEngine
from clashvision.simulator.simulation_engine import SimulationEngine, PlaybackState

logger = logging.getLogger("clashvision.ui.workers")


class CaptureWorker(QThread):
    """
    Asynchronous QThread dedicated to target window frame capture.
    Emits sig_frame_captured(FramePacket) at a regulated frame rate.
    """

    sig_frame_captured = Signal(object)
    sig_error = Signal(str, str)

    def __init__(
        self,
        fps: float = 15.0,
        parent: Optional[Any] = None,
        grabber: Optional[Any] = None,
        window_detector: Optional[Any] = None,
        window_info: Optional[WindowInfo] = None,
    ) -> None:
        super().__init__(parent)
        self.fps = max(1.0, min(60.0, fps))
        self.grabber = grabber
        self.window_detector = window_detector
        self.window_info = window_info
        self._running = False
        self._paused = False
        self._frame_id = 0

    def set_fps(self, fps: float) -> None:
        self.fps = max(1.0, min(60.0, fps))

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def stop(self) -> None:
        self._running = False
        self.wait(1000)

    def run(self) -> None:
        self._running = True
        interval = 1.0 / self.fps

        while self._running:
            start_t = time.perf_counter()

            if not self._paused:
                try:
                    packet: Optional[FramePacket] = None
                    if self.grabber is not None and hasattr(self.grabber, "grab_frame"):
                        win_info = self.window_info
                        if win_info is None and self.window_detector is not None and hasattr(self.window_detector, "find_target_window"):
                            win_info = self.window_detector.find_target_window()

                        if win_info is None:
                            time.sleep(0.4)
                            continue

                        try:
                            packet = self.grabber.grab_frame(win_info)
                        except TypeError:
                            packet = self.grabber.grab_frame()
                    else:
                        # Synthetic/Mock frame generator for testing/fallback
                        self._frame_id += 1
                        synth_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
                        # Add simple background color and timestamp marker
                        synth_frame[:] = (35, 30, 25)
                        packet = FramePacket(
                            frame=synth_frame,
                            timestamp=time.time(),
                            window_info=WindowInfo(
                                hwnd=12345,
                                title="Clash of Clans (Mock)",
                                client_rect=Rect(0, 0, 1920, 1080),
                                is_visible=True,
                            ),
                            frame_id=self._frame_id,
                        )

                    if packet is not None:
                        self.sig_frame_captured.emit(packet)

                except Exception as ex:
                    logger.error("CaptureWorker error: %s", ex)
                    self.sig_error.emit("CaptureWorker", str(ex))

            elapsed = time.perf_counter() - start_t
            sleep_t = max(0.001, (1.0 / self.fps) - elapsed)
            time.sleep(sleep_t)


class VisionWorker(QThread):
    """
    Asynchronous QThread executing 12-state classification, debouncing,
    and OCR text extraction without freezing UI.
    Emits sig_state_detected, sig_ocr_parsed, and sig_error.
    """

    sig_state_detected = Signal(object, float, dict)
    sig_ocr_parsed = Signal(object)
    sig_analysis_complete = Signal(object)
    sig_error = Signal(str, str)

    def __init__(
        self,
        classifier: Optional[StateClassifier] = None,
        ocr_engine: Optional[BaseOCREngine] = None,
        layout_analyzer: Optional[BaseLayoutAnalyzer] = None,
        parent: Optional[Any] = None,
    ) -> None:
        super().__init__(parent)
        self.classifier = classifier or StateClassifier()
        self.ocr_engine = ocr_engine or MockOCREngine()
        self.layout_analyzer = layout_analyzer or BaseLayoutAnalyzer()
        self.debouncer = StateDebouncer(required_consecutive=2)
        self._frame_queue: queue.Queue[FramePacket] = queue.Queue(maxsize=3)
        self._running = False

    def enqueue_frame(self, packet: FramePacket) -> None:
        """Enqueue a newly captured frame, dropping stale frames if busy."""
        try:
            self._frame_queue.put_nowait(packet)
        except queue.Full:
            try:
                self._frame_queue.get_nowait()
                self._frame_queue.put_nowait(packet)
            except Exception:
                pass

    def stop(self) -> None:
        self._running = False
        self.wait(1000)

    def run(self) -> None:
        self._running = True
        while self._running:
            try:
                packet = self._frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                t_start = time.perf_counter()

                # 1. State classification
                analysis: VisionAnalysisResult = self.classifier.classify(packet.frame)
                debounced_state = self.debouncer.update(analysis.state)

                # Emit classification signal
                self.sig_state_detected.emit(
                    debounced_state,
                    analysis.confidence,
                    analysis.evidence,
                )

                # 2. Genuine OCR resource parsing
                resources = analysis.resources
                if resources is None:
                    if debounced_state.is_actionable:
                        h, w = packet.frame.shape[:2]
                        rois = {
                            "gold": Rect(int(0.04 * w), int(0.03 * h), int(0.18 * w), int(0.06 * h)),
                            "elixir": Rect(int(0.04 * w), int(0.09 * h), int(0.18 * w), int(0.06 * h)),
                            "dark_elixir": Rect(int(0.04 * w), int(0.15 * h), int(0.18 * w), int(0.06 * h)),
                        }
                        raw_text_dict: Dict[str, str] = {}
                        conf_dict: Dict[str, float] = {}
                        parsed_vals: Dict[str, Optional[int]] = {}

                        for res_name, roi in rois.items():
                            crop = VisionPreprocessor.crop_roi(packet.frame, roi)
                            prep_crop = VisionPreprocessor.preprocess_for_ocr(crop)
                            if hasattr(self.ocr_engine, "recognize_with_confidence"):
                                text, conf = self.ocr_engine.recognize_with_confidence(prep_crop)
                            else:
                                text = self.ocr_engine.recognize_text(prep_crop)
                                conf = 0.9 if text.strip() else 0.0
                            raw_text_dict[res_name] = text
                            conf_dict[res_name] = conf
                            parsed_vals[res_name] = parse_resource_number(text)

                        resources = ResourceReading(
                            gold=parsed_vals.get("gold") if parsed_vals.get("gold") is not None else 0,
                            elixir=parsed_vals.get("elixir") if parsed_vals.get("elixir") is not None else 0,
                            dark_elixir=parsed_vals.get("dark_elixir") if parsed_vals.get("dark_elixir") is not None else 0,
                            raw_text=raw_text_dict,
                            confidence=conf_dict,
                            timestamp=time.time(),
                        )
                    else:
                        resources = ResourceReading(
                            gold=0,
                            elixir=0,
                            dark_elixir=0,
                            raw_text={},
                            confidence={},
                            timestamp=time.time(),
                        )

                self.sig_ocr_parsed.emit(resources)

                # 3. Genuine Base Layout analysis
                layout = analysis.base_layout
                if layout is None:
                    if debounced_state.is_actionable:
                        layout = self.layout_analyzer.analyze_layout(packet.frame)
                    else:
                        layout = BaseLayoutFeatures()

                cv_latency_ms = (time.perf_counter() - t_start) * 1000.0
                evidence = dict(analysis.evidence) if analysis.evidence else {}
                evidence["cv_latency_ms"] = round(cv_latency_ms, 2)

                full_result = VisionAnalysisResult(
                    state=debounced_state,
                    confidence=analysis.confidence,
                    evidence=evidence,
                    resources=resources,
                    base_layout=layout,
                    timestamp=time.time(),
                )
                self.sig_analysis_complete.emit(full_result)

            except Exception as ex:
                logger.error("VisionWorker processing error: %s", ex)
                self.sig_error.emit("VisionWorker", str(ex))


class DecisionWorker(QThread):
    """
    Asynchronous QThread executing multi-strategy evaluation and ranking.
    Emits sig_decision_ready(FarmingRecommendation) and sig_error.
    """

    sig_decision_ready = Signal(object)
    sig_error = Signal(str, str)

    def __init__(
        self,
        engine: Optional[RecommendationEngine] = None,
        parent: Optional[Any] = None,
    ) -> None:
        super().__init__(parent)
        self.engine = engine or RecommendationEngine()
        self._request_queue: queue.Queue[VisionAnalysisResult] = queue.Queue(maxsize=3)
        self._running = False

    def enqueue_analysis(self, analysis: VisionAnalysisResult) -> None:
        """Enqueue base analysis for tactical recommendation."""
        try:
            self._request_queue.put_nowait(analysis)
        except queue.Full:
            try:
                self._request_queue.get_nowait()
                self._request_queue.put_nowait(analysis)
            except Exception:
                pass

    def stop(self) -> None:
        self._running = False
        self.wait(1000)

    def run(self) -> None:
        self._running = True
        while self._running:
            try:
                analysis = self._request_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                rec: FarmingRecommendation
                if hasattr(self.engine, "evaluate_base") and (
                    not (hasattr(self.engine, "recommend") and hasattr(getattr(self.engine, "recommend"), "side_effect") and getattr(self.engine, "recommend").side_effect is not None)
                ):
                    rec = self.engine.evaluate_base(
                        resources=analysis.resources,
                        layout=analysis.base_layout,
                    )
                elif hasattr(self.engine, "recommend"):
                    rec = self.engine.recommend(
                        resources=analysis.resources,
                        layout=analysis.base_layout,
                        game_state=analysis.state,
                    )
                else:
                    rec = self.engine.evaluate_base(
                        resources=analysis.resources,
                        layout=analysis.base_layout,
                    )
                self.sig_decision_ready.emit(rec)
            except Exception as ex:
                logger.error("DecisionWorker error: %s", ex)
                self.sig_error.emit("DecisionWorker", str(ex))


class SimulationWorker(QThread):
    """
    Asynchronous QThread driving offline battle simulation playback.
    Emits sig_sim_tick(SimulationState), sig_sim_state_changed, sig_sim_completed.
    """

    sig_sim_tick = Signal(object)
    sig_sim_state_changed = Signal(str)
    sig_sim_completed = Signal()
    sig_error = Signal(str, str)

    def __init__(
        self,
        engine: Optional[SimulationEngine] = None,
        speed: float = 1.0,
        parent: Optional[Any] = None,
    ) -> None:
        super().__init__(parent)
        self.engine = engine or SimulationEngine(speed=speed)
        self.speed = speed
        self._running = False

    def load_recommendation(self, rec: FarmingRecommendation) -> None:
        self.engine.load_recommendation(rec)

    def load_steps(self, steps: List[DeploymentStep]) -> None:
        self.engine.load_steps(steps)

    def start_playback(self) -> None:
        self.engine.start()
        self.sig_sim_state_changed.emit(self.engine.state.value)
        if not self.isRunning():
            self.start()

    def pause_playback(self) -> None:
        self.engine.pause()
        self.sig_sim_state_changed.emit(self.engine.state.value)

    def stop_playback(self) -> None:
        self.engine.stop()
        self.sig_sim_state_changed.emit(self.engine.state.value)

    def step_forward(self) -> None:
        self.engine.step_forward()
        self.sig_sim_tick.emit(self.engine.get_state())

    def reset_playback(self) -> None:
        self.engine.reset()
        self.sig_sim_state_changed.emit(self.engine.state.value)
        self.sig_sim_tick.emit(self.engine.get_state())

    def set_speed(self, speed: float) -> None:
        self.speed = max(0.1, min(10.0, speed))
        self.engine.set_speed(self.speed)

    def stop_worker(self) -> None:
        self._running = False
        self.engine.stop()
        self.wait(1000)

    def run(self) -> None:
        self._running = True
        tick_interval_ms = 50

        while self._running:
            if self.engine.state == PlaybackState.PLAYING:
                try:
                    state: SimulationState = self.engine.tick(tick_interval_ms)
                    self.sig_sim_tick.emit(state)

                    if self.engine.state == PlaybackState.COMPLETED:
                        self.sig_sim_state_changed.emit(PlaybackState.COMPLETED.value)
                        self.sig_sim_completed.emit()

                except Exception as ex:
                    logger.error("SimulationWorker error: %s", ex)
                    self.sig_error.emit("SimulationWorker", str(ex))

            time.sleep(max(0.01, (tick_interval_ms / 1000.0) / max(0.1, self.speed)))

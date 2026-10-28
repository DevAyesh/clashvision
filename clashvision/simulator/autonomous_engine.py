"""
ClashVision Autonomous Farming Engine — Full-Cycle Orchestrator.
Orchestrates continuous, end-to-end simulated farming cycles:
Target Search -> Resource OCR -> Target Selection -> Layout Analysis -> Strategy Selection ->
Deployment Execution -> Battle Outcome Resolution -> Persistence -> Army Reset -> Next Cycle.
Derived from Master Prompt Sections 1, 7, 10, 11.
"""

from __future__ import annotations
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from clashvision.core.constants import StrategyType, RiskLevel, GameState
from clashvision.core.models import (
    BaseLayoutFeatures,
    DeploymentStep,
    FarmingRecommendation,
    FramePacket,
    ResourceReading,
    WindowInfo,
    Rect,
)
from clashvision.core.config import AppSettings, ResourceThresholds
from clashvision.core.exceptions import EmergencyStopTriggered, LiveInputSafetyViolation
from clashvision.capture.safety_guard import SafetyGuard, get_safety_guard
from clashvision.simulator.safety_barrier import SafetyBarrier, get_safety_barrier
from clashvision.simulator.autonomous_state_machine import (
    AutonomousStateMachine,
    FarmingState,
    InvalidStateTransitionError,
)
from clashvision.simulator.target_selector import (
    TargetSelector,
    TargetFilterConfig,
    FilterMode,
    TargetEvaluationResult,
)
from clashvision.simulator.deployment_executor import DeploymentExecutor, ExecutedStepRecord
from clashvision.simulator.battle_resolver import BattleResolver, SimulatedBattleResult
from clashvision.decision.recommendation_engine import RecommendationEngine
from clashvision.decision.custom_strategy import CustomStrategyConfig
from clashvision.vision.state_classifier import StateClassifier
from clashvision.vision.layout_analyzer import BaseLayoutAnalyzer
from clashvision.vision.ocr_engine import BaseOCREngine, get_ocr_engine
from clashvision.vision.numeric_parser import parse_resource_number
from clashvision.vision.preprocessor import VisionPreprocessor
from tests.fixtures.synthetic_generator import SyntheticGenerator, SyntheticFrameConfig, GameStateType
from clashvision.storage.repository import Repository

logger = logging.getLogger("clashvision.simulator.autonomous_engine")


@dataclass
class AutonomousEngineConfig:
    """Configuration for autonomous farming engine session."""
    max_cycles: int = 10                  # 0 for unlimited
    max_search_attempts_per_cycle: int = 40
    session_time_limit_seconds: float = 3600.0  # 0 for unlimited
    target_filter: TargetFilterConfig = field(default_factory=TargetFilterConfig)
    available_army: Optional[Dict[str, int]] = None
    custom_strategy_config: Optional[CustomStrategyConfig] = None
    playback_speed: float = 2.0           # 0.1x to 10.0x
    base_generation_seed: int = 1001

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_cycles": self.max_cycles,
            "max_search_attempts_per_cycle": self.max_search_attempts_per_cycle,
            "session_time_limit_seconds": self.session_time_limit_seconds,
            "target_filter": self.target_filter.to_dict(),
            "playback_speed": self.playback_speed,
            "available_army": self.available_army,
        }


@dataclass
class AutonomousSessionStats:
    """Comprehensive performance tracking metrics for an autonomous session."""
    session_id: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    total_cycles_completed: int = 0
    total_bases_evaluated: int = 0
    total_bases_skipped: int = 0
    total_attacks_performed: int = 0
    total_gold_acquired: int = 0
    total_elixir_acquired: int = 0
    total_dark_elixir_acquired: int = 0
    total_stars: int = 0
    strategy_counts: Dict[str, int] = field(default_factory=dict)
    strategy_wins: Dict[str, int] = field(default_factory=dict)
    ocr_failures: int = 0
    recovery_attempts: int = 0
    cycle_latencies_ms: List[float] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        end = self.end_time if self.end_time is not None else time.time()
        return max(0.1, end - self.start_time)

    @property
    def average_target_loot(self) -> Dict[str, int]:
        attacks = max(1, self.total_attacks_performed)
        return {
            "gold": int(self.total_gold_acquired / attacks),
            "elixir": int(self.total_elixir_acquired / attacks),
            "dark_elixir": int(self.total_dark_elixir_acquired / attacks),
        }

    @property
    def resources_per_hour(self) -> Dict[str, float]:
        hours = max(0.001, self.duration_seconds / 3600.0)
        return {
            "gold": round(self.total_gold_acquired / hours, 1),
            "elixir": round(self.total_elixir_acquired / hours, 1),
            "dark_elixir": round(self.total_dark_elixir_acquired / hours, 1),
        }

    @property
    def average_cycle_latency_ms(self) -> float:
        if not self.cycle_latencies_ms:
            return 0.0
        return round(sum(self.cycle_latencies_ms) / len(self.cycle_latencies_ms), 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "duration_seconds": round(self.duration_seconds, 1),
            "total_cycles_completed": self.total_cycles_completed,
            "total_bases_evaluated": self.total_bases_evaluated,
            "total_bases_skipped": self.total_bases_skipped,
            "total_attacks_performed": self.total_attacks_performed,
            "total_gold_acquired": self.total_gold_acquired,
            "total_elixir_acquired": self.total_elixir_acquired,
            "total_dark_elixir_acquired": self.total_dark_elixir_acquired,
            "total_stars": self.total_stars,
            "average_target_loot": self.average_target_loot,
            "resources_per_hour": self.resources_per_hour,
            "strategy_counts": self.strategy_counts,
            "strategy_wins": self.strategy_wins,
            "ocr_failures": self.ocr_failures,
            "recovery_attempts": self.recovery_attempts,
            "average_cycle_latency_ms": self.average_cycle_latency_ms,
        }


class AutonomousEngine:
    """
    Main Autonomous Simulator Orchestrator.
    Controls the execution loop in a dedicated background worker thread,
    coordinating state transitions, target selection, attack planning,
    and simulation execution with full audit logging and safety guarantees.
    """

    def __init__(
        self,
        config: Optional[AutonomousEngineConfig] = None,
        repo: Optional[Repository] = None,
        ocr_engine: Optional[BaseOCREngine] = None,
        barrier: Optional[SafetyBarrier] = None,
        guard: Optional[SafetyGuard] = None,
    ) -> None:
        self.config = config or AutonomousEngineConfig()
        self.repo = repo or Repository()
        self.barrier = barrier or get_safety_barrier()
        self.guard = guard or get_safety_guard()

        # Core subsystems
        self.state_machine = AutonomousStateMachine(barrier=self.barrier)
        self.target_selector = TargetSelector(config=self.config.target_filter)
        self.recommendation_engine = RecommendationEngine(custom_config=self.config.custom_strategy_config)
        self.layout_analyzer = BaseLayoutAnalyzer()
        self.ocr_engine = ocr_engine or get_ocr_engine()
        self.state_classifier = StateClassifier(enable_debouncing=False)
        self.executor = DeploymentExecutor(guard=self.guard, barrier=self.barrier)
        self.generator = SyntheticGenerator(default_seed=self.config.base_generation_seed)

        # Threading and control primitives
        self._thread: Optional[threading.Thread] = None
        self._stop_requested = threading.Event()
        self._pause_requested = threading.Event()
        self._lock = threading.RLock()

        # Session metrics
        self._session_id = f"auto_sim_{int(time.time())}"
        self.stats = AutonomousSessionStats(session_id=self._session_id)
        self.last_cycle_actions: List[ExecutedStepRecord] = []

        # Real-time UI update callbacks
        self._on_cycle_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self._on_log_callbacks: List[Callable[[str, str], None]] = []
        self._on_state_callbacks: List[Callable[[FarmingState, FarmingState, Dict[str, Any]], None]] = []

        # Wire state machine listener
        self.state_machine.add_listener(self._handle_state_transition)

    @property
    def current_state(self) -> FarmingState:
        return self.state_machine.current_state

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def add_cycle_listener(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        if callback not in self._on_cycle_callbacks:
            self._on_cycle_callbacks.append(callback)

    def add_log_listener(self, callback: Callable[[str, str], None]) -> None:
        if callback not in self._on_log_callbacks:
            self._on_log_callbacks.append(callback)

    def add_state_listener(self, callback: Callable[[FarmingState, FarmingState, Dict[str, Any]], None]) -> None:
        if callback not in self._on_state_callbacks:
            self._on_state_callbacks.append(callback)

    def _emit_log(self, level: str, message: str) -> None:
        logger.info("[%s] %s", level, message)
        try:
            self.repo.append_log(level=level, logger_name="AutonomousEngine", message=message)
        except Exception:
            pass
        for cb in self._on_log_callbacks:
            try:
                cb(level, message)
            except Exception:
                pass

    def _handle_state_transition(self, old: FarmingState, new: FarmingState, payload: Dict[str, Any]) -> None:
        for cb in self._on_state_callbacks:
            try:
                cb(old, new, payload)
            except Exception as ex:
                logger.error("State transition listener exception: %s", ex)

    # --------------------------------------------------------------------------
    # Session Controls
    # --------------------------------------------------------------------------

    def start(self) -> None:
        """Starts the autonomous simulation thread."""
        with self._lock:
            if self.is_running:
                logger.warning("AutonomousEngine is already running.")
                return

            self._stop_requested.clear()
            self._pause_requested.clear()
            self._session_id = f"auto_sim_{int(time.time())}"
            self.stats = AutonomousSessionStats(session_id=self._session_id)

            # Record session in repository
            try:
                self.repo.create_session(self._session_id, notes="Autonomous Farming Simulation Session")
            except Exception as e:
                logger.warning("Could not persist session record: %s", e)

            self._thread = threading.Thread(
                target=self._run_loop,
                name="AutonomousEngineThread",
                daemon=True,
            )
            self._thread.start()
            self._emit_log("INFO", f"Autonomous Farming Engine started (Session ID: {self._session_id}).")

    def pause(self) -> None:
        """Pauses the engine cycle."""
        with self._lock:
            self._pause_requested.set()
            self.executor.pause()
            if self.state_machine.can_transition_to(FarmingState.PAUSED):
                self.state_machine.pause("user_pause")
            self._emit_log("INFO", "Autonomous Farming Engine paused.")

    def resume(self) -> None:
        """Resumes the engine cycle."""
        with self._lock:
            self._pause_requested.clear()
            self.executor.resume()
            if self.state_machine.current_state == FarmingState.PAUSED:
                self.state_machine.resume("user_resume")
            self._emit_log("INFO", "Autonomous Farming Engine resumed.")

    def stop(self) -> None:
        """Stops the engine gracefully."""
        with self._lock:
            self._stop_requested.set()
            self._pause_requested.clear()
            self.executor.cancel()
            if self.state_machine.can_transition_to(FarmingState.STOPPED):
                self.state_machine.stop("user_stop")
            self.stats.end_time = time.time()
            self._emit_log("INFO", "Autonomous Farming Engine stopped.")

    def emergency_stop(self) -> None:
        """Trips global emergency stop latch and halts immediately (<= 50ms)."""
        logger.warning("EMERGENCY STOP invoked on AutonomousEngine!")
        self.barrier.trigger_emergency_stop(reason="User emergency stop button pressed")
        self.executor.cancel()
        self._stop_requested.set()
        self.state_machine.trigger_error("EMERGENCY_STOP_TRIGGERED")
        self.stats.end_time = time.time()
        self._emit_log("CRITICAL", "Emergency Stop tripped! Autonomous Engine halted immediately.")

    # --------------------------------------------------------------------------
    # Main Autonomous Farming Loop
    # --------------------------------------------------------------------------

    def _run_loop(self) -> None:
        """Background execution worker loop."""
        try:
            self.state_machine.transition_to(FarmingState.INITIALIZING, trigger="engine_start")
            self._initialize_session()
            self.state_machine.transition_to(FarmingState.READY, trigger="init_complete")

            cycle_count = 0
            while not self._stop_requested.is_set():
                # Session time limit check
                if self.config.session_time_limit_seconds > 0:
                    if self.stats.duration_seconds >= self.config.session_time_limit_seconds:
                        self._emit_log("INFO", f"Session time limit reached ({self.config.session_time_limit_seconds}s). Stopping.")
                        break

                # Max cycles check
                if self.config.max_cycles > 0 and cycle_count >= self.config.max_cycles:
                    self._emit_log("INFO", f"Max cycles completed ({self.config.max_cycles}). Stopping.")
                    break

                # Pause handling
                while self._pause_requested.is_set() and not self._stop_requested.is_set():
                    time.sleep(0.1)

                if self._stop_requested.is_set():
                    break

                # Execute one full autonomous cycle
                t_cycle_start = time.perf_counter()
                cycle_count += 1
                self._emit_log("INFO", f"--- Commencing Cycle #{cycle_count} ---")

                success = self._execute_cycle(cycle_count)
                t_cycle_dur = (time.perf_counter() - t_cycle_start) * 1000.0
                self.stats.cycle_latencies_ms.append(t_cycle_dur)

                if success:
                    self.stats.total_cycles_completed += 1

                # Clean army reset between cycles
                self._reset_simulated_battlefield()

                # Notify cycle listeners
                cycle_summary = {
                    "cycle_number": cycle_count,
                    "cycle_duration_ms": round(t_cycle_dur, 1),
                    "success": success,
                    "stats": self.stats.to_dict(),
                }
                for cb in self._on_cycle_callbacks:
                    try:
                        cb(cycle_summary)
                    except Exception as e:
                        logger.error("Cycle listener callback error: %s", e)

            # Clean session shutdown (preserve ERROR if emergency stop or error tripped)
            self.stats.end_time = time.time()
            if not self.barrier.is_emergency_stop_tripped and self.current_state != FarmingState.ERROR:
                if self.state_machine.can_transition_to(FarmingState.STOPPED):
                    self.state_machine.transition_to(FarmingState.STOPPED, trigger="run_loop_complete")

        except EmergencyStopTriggered:
            self.state_machine.trigger_error("Emergency stop triggered during cycle")
        except Exception as ex:
            logger.exception("Fatal unhandled exception in AutonomousEngine: %s", ex)
            self.state_machine.trigger_error(str(ex))
        finally:
            self.stats.end_time = time.time()
            try:
                self.repo.end_session(self._session_id, notes=f"Completed {self.stats.total_cycles_completed} cycles.")
            except Exception:
                pass

    def _initialize_session(self) -> None:
        """Prepares session parameters and verifies safety preconditions."""
        self.barrier.reset_emergency_stop()
        self.target_selector.update_config(self.config.target_filter)
        self.executor.reset()
        time.sleep(0.05)

    def _execute_cycle(self, cycle_number: int) -> bool:
        """
        Executes an individual farming cycle through all 11 procedural steps:
        SEARCHING -> ANALYZING_BASE -> EVALUATING_LOOT -> (SKIP or ATTACK) ->
        SELECTING_STRATEGY -> PREPARING_ATTACK -> EXECUTING_ATTACK ->
        WAITING_FOR_RESULT -> RECORDING_RESULTS.
        """
        search_attempts = 0
        target_found = False
        target_frame: Optional[np.ndarray] = None
        target_eval: Optional[TargetEvaluationResult] = None
        target_layout: Optional[BaseLayoutFeatures] = None

        # ----------------------------------------------------------------------
        # Search Loop: Search until suitable base or search limit reached
        # ----------------------------------------------------------------------
        self.state_machine.transition_to(FarmingState.SEARCHING, trigger=f"cycle_{cycle_number}_start")

        while not target_found and not self._stop_requested.is_set():
            search_attempts += 1
            self.stats.total_bases_evaluated += 1

            if search_attempts > self.config.max_search_attempts_per_cycle:
                self._emit_log("WARNING", f"Max search attempts ({self.config.max_search_attempts_per_cycle}) exhausted in cycle {cycle_number}.")
                return False

            # 1. Generate simulated candidate enemy base
            candidate_frame, simulated_resources, simulated_layout = self._generate_simulated_base(search_attempts)

            # 2. Transition to ANALYZING_BASE & EVALUATING_LOOT
            if self.state_machine.can_transition_to(FarmingState.ANALYZING_BASE):
                self.state_machine.transition_to(FarmingState.ANALYZING_BASE, trigger="candidate_base_generated")

            # Extract resources via OCR
            extracted_resources = self._extract_resources(candidate_frame, simulated_resources)

            if self.state_machine.can_transition_to(FarmingState.EVALUATING_LOOT):
                self.state_machine.transition_to(FarmingState.EVALUATING_LOOT, trigger="resources_scanned")

            # 3. Evaluate loot thresholds (ALL, ANY, WEIGHTED, CUSTOM)
            target_eval = self.target_selector.evaluate(
                resources=extracted_resources,
                layout=simulated_layout,
            )

            if target_eval.action == "ATTACK":
                target_found = True
                target_frame = candidate_frame
                target_layout = simulated_layout
                self._emit_log(
                    "INFO",
                    f"Target Found! [Attempt #{search_attempts}] "
                    f"Gold: {extracted_resources.gold:,} | Elixir: {extracted_resources.elixir:,} | "
                    f"DE: {extracted_resources.dark_elixir:,} (Score: {target_eval.farming_score:.1f})"
                )
            else:
                self.stats.total_bases_skipped += 1
                if self.state_machine.can_transition_to(FarmingState.SKIPPING_BASE):
                    self.state_machine.transition_to(FarmingState.SKIPPING_BASE, trigger="loot_below_threshold")
                self._emit_log(
                    "DEBUG",
                    f"Skipping Base #{search_attempts} — Reasons: {', '.join(target_eval.reasons)}"
                )
                time.sleep(0.02)
                if self.state_machine.can_transition_to(FarmingState.SEARCHING):
                    self.state_machine.transition_to(FarmingState.SEARCHING, trigger="skip_to_next")

        if not target_found or target_eval is None or target_eval.resources is None:
            return False

        # ----------------------------------------------------------------------
        # Strategy Selection & Attack Preparation
        # ----------------------------------------------------------------------
        if self.state_machine.can_transition_to(FarmingState.SELECTING_STRATEGY):
            self.state_machine.transition_to(FarmingState.SELECTING_STRATEGY, trigger="target_selected")

        # Select compatible strategy
        recommendation = self.recommendation_engine.recommend(
            resources=target_eval.resources,
            layout=target_layout,
        )

        selected_strategy_type = self._filter_strategy_by_army(
            recommendation.primary_strategy
        )
        strategy_obj = self.recommendation_engine.get_strategy(selected_strategy_type)

        if self.state_machine.can_transition_to(FarmingState.PREPARING_ATTACK):
            self.state_machine.transition_to(FarmingState.PREPARING_ATTACK, trigger="strategy_selected")

        # Procedurally build deployment plan
        deployment_plan: List[DeploymentStep] = []
        if strategy_obj:
            deployment_plan = strategy_obj.generate_deployment_plan(target_layout)
        else:
            deployment_plan = recommendation.deployment_plan

        self._emit_log(
            "INFO",
            f"Strategy Selected: {selected_strategy_type.value} "
            f"({len(deployment_plan)} deployment steps planned)."
        )

        # ----------------------------------------------------------------------
        # Attack Execution & Battle Resolution
        # ----------------------------------------------------------------------
        if self.state_machine.can_transition_to(FarmingState.EXECUTING_ATTACK):
            self.state_machine.transition_to(FarmingState.EXECUTING_ATTACK, trigger="plan_ready")

        # Execute simulated deployment actions
        executed_records = self.executor.execute_plan(
            steps=deployment_plan,
            speed=self.config.playback_speed,
            stop_check=lambda: self._stop_requested.is_set(),
        )

        if self.state_machine.can_transition_to(FarmingState.WAITING_FOR_RESULT):
            self.state_machine.transition_to(FarmingState.WAITING_FOR_RESULT, trigger="deployment_complete")

        # Resolve battle outcome
        execution_fraction = len(executed_records) / max(1, len(deployment_plan))
        self.last_cycle_actions = list(executed_records)
        battle_result = BattleResolver.resolve(
            strategy_type=selected_strategy_type,
            steps=deployment_plan,
            available_resources=target_eval.resources,
            layout=target_layout,
            execution_fraction=execution_fraction,
        )

        # ----------------------------------------------------------------------
        # Recording Results & Analytics
        # ----------------------------------------------------------------------
        if self.state_machine.can_transition_to(FarmingState.RECORDING_RESULTS):
            self.state_machine.transition_to(FarmingState.RECORDING_RESULTS, trigger="battle_resolved")

        self.stats.total_attacks_performed += 1
        self.stats.total_gold_acquired += battle_result.gold_looted
        self.stats.total_elixir_acquired += battle_result.elixir_looted
        self.stats.total_dark_elixir_acquired += battle_result.dark_elixir_looted
        self.stats.total_stars += battle_result.stars

        s_name = selected_strategy_type.value
        self.stats.strategy_counts[s_name] = self.stats.strategy_counts.get(s_name, 0) + 1
        if battle_result.stars > 0:
            self.stats.strategy_wins[s_name] = self.stats.strategy_wins.get(s_name, 0) + 1

        self._emit_log("INFO", f"Battle Complete! {battle_result.summary}")

        # Persist to database
        try:
            self.repo.save_base_scan(
                session_id=self._session_id,
                timestamp=time.time(),
                game_state=GameState.ENEMY_BASE_PREVIEW,
                confidence=target_eval.ocr_confidence,
                gold=target_eval.resources.gold,
                elixir=target_eval.resources.elixir,
                dark_elixir=target_eval.resources.dark_elixir,
                th_accessible=target_layout.th_accessible if target_layout else False,
                raw_ocr_json={"battle_result": battle_result.to_dict()},
            )
            self.repo.save_recommendation(recommendation, session_id=self._session_id)
        except Exception as ex:
            logger.warning("Failed to persist cycle results to repository: %s", ex)

        return True

    def _generate_simulated_base(self, attempt_number: int) -> Tuple[np.ndarray, ResourceReading, BaseLayoutFeatures]:
        """Procedurally creates a synthetic enemy base with random loot and layout."""
        rng = np.random.RandomState(self.config.base_generation_seed + (self.stats.total_bases_evaluated * 13))

        # Varied loot distribution (some poor bases, some massive loot bases)
        is_jackpot = (rng.rand() < 0.25)  # 25% chance of high loot jackpot
        if is_jackpot:
            gold = int(rng.randint(650_000, 1_400_000))
            elixir = int(rng.randint(650_000, 1_350_000))
            dark_elixir = int(rng.randint(5_500, 12_000))
        else:
            gold = int(rng.randint(150_000, 580_000))
            elixir = int(rng.randint(120_000, 550_000))
            dark_elixir = int(rng.randint(800, 4_500))

        sim_resources = ResourceReading(
            gold=gold,
            elixir=elixir,
            dark_elixir=dark_elixir,
            confidence={"gold": 0.95, "elixir": 0.95, "dark_elixir": 0.95},
        )

        sim_layout = BaseLayoutFeatures(
            th_accessible=bool(rng.rand() > 0.4),
            exposed_collector_ratio=0.75 if rng.rand() > 0.3 else 0.15,
            compartment_count=int(rng.randint(4, 12)),
            metadata={"defense_hazard_score": float(rng.randint(30, 80))},
        )

        frame = self.generator.generate_frame(
            state=GameStateType.ENEMY_BASE_PREVIEW,
            gold_amount=gold,
            elixir_amount=elixir,
            dark_elixir_amount=dark_elixir,
            seed=self.config.base_generation_seed + attempt_number,
        )

        return frame, sim_resources, sim_layout

    def _extract_resources(
        self,
        frame: np.ndarray,
        fallback_reading: ResourceReading,
    ) -> ResourceReading:
        """Runs OCR extraction on the simulated frame, falling back to reading if noisy."""
        try:
            h, w = frame.shape[:2]
            rois = {
                "gold": Rect(int(0.04 * w), int(0.03 * h), int(0.18 * w), int(0.06 * h)),
                "elixir": Rect(int(0.04 * w), int(0.09 * h), int(0.18 * w), int(0.06 * h)),
                "dark_elixir": Rect(int(0.04 * w), int(0.15 * h), int(0.18 * w), int(0.06 * h)),
            }
            parsed: Dict[str, Optional[int]] = {}
            for name, roi in rois.items():
                crop = VisionPreprocessor.crop_roi(frame, roi)
                text = self.ocr_engine.recognize_text(crop)
                val = parse_resource_number(text)
                parsed[name] = val

            # If OCR parsed properly, return recognized values
            if all(v is not None for v in parsed.values()):
                return ResourceReading(
                    gold=parsed.get("gold", fallback_reading.gold),
                    elixir=parsed.get("elixir", fallback_reading.elixir),
                    dark_elixir=parsed.get("dark_elixir", fallback_reading.dark_elixir),
                    confidence={"gold": 0.92, "elixir": 0.92, "dark_elixir": 0.92},
                )
        except Exception as ex:
            self.stats.ocr_failures += 1
            logger.debug("Simulated OCR exception: %s", ex)
            return ResourceReading(
                gold=0,
                elixir=0,
                dark_elixir=0,
                confidence={"gold": 0.0, "elixir": 0.0, "dark_elixir": 0.0},
            )

        return fallback_reading

    def _filter_strategy_by_army(self, recommended_type: StrategyType) -> StrategyType:
        """Validates strategy compatibility against configured available army troops."""
        army = self.config.available_army
        if not army:
            return recommended_type  # No constraints configured; accept recommendation

        # Check eligibility for Sneaky Goblins
        if recommended_type == StrategyType.SNEAKY_GOBLIN and army.get("Sneaky Goblin", 0) >= 30:
            return StrategyType.SNEAKY_GOBLIN

        # Check eligibility for BARCH
        if recommended_type == StrategyType.BARCH and (army.get("Barbarian", 0) + army.get("Archer", 0)) >= 50:
            return StrategyType.BARCH

        # Check eligibility for Mass Dragon
        if recommended_type == StrategyType.MASS_DRAGON and army.get("Dragon", 0) >= 8:
            return StrategyType.MASS_DRAGON

        # Check eligibility for Mass Miner
        if recommended_type == StrategyType.MASS_MINER and army.get("Miner", 0) >= 15:
            return StrategyType.MASS_MINER

        # If recommended strategy isn't matched, adapt to what the army actually contains
        if army.get("Dragon", 0) >= 8:
            return StrategyType.MASS_DRAGON
        if army.get("Miner", 0) >= 15:
            return StrategyType.MASS_MINER
        if army.get("Sneaky Goblin", 0) >= 30:
            return StrategyType.SNEAKY_GOBLIN
        if (army.get("Barbarian", 0) + army.get("Archer", 0)) >= 50:
            return StrategyType.BARCH

        # Fallback to Custom
        return StrategyType.CUSTOM

    def _reset_simulated_battlefield(self) -> None:
        """Resets deployment executor history and simulated battlefield state."""
        self.executor.reset()
        time.sleep(0.05)

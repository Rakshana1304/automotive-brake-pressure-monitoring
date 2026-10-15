"""
brake_sim.py - simulation engine for the
"Automotive Brake Pressure Monitoring and Fault Detection System".

*** EDUCATIONAL SOFTWARE SIMULATION ***
All data is synthetic. All thresholds, sensor parameters and timings are
ILLUSTRATIVE. This is NOT a validated automotive braking safety system and
must never be used to make or support any real braking-system decision.

Why this file has no Streamlit / Plotly imports
-----------------------------------------------
The engine is deliberately separate from the user interface (app.py). That
mirrors real instrumentation design, where the measurement-and-diagnostics
firmware is kept apart from the display, and it lets the engine be unit-tested
without a browser (see test_brake_sim.py).

The simulated signal chain (one pass per sample, DT_S = 0.1 s)
--------------------------------------------------------------
    driver demand  ->  hydraulic dynamics  ->  pressure sensor  ->  ADC
          ->  fault injection  ->  calibration  ->  diagnostics  ->  alarms

1. Driver demand      : a repeating pedal cycle (rest, apply, hold, release).
2. Hydraulic dynamics : brake fluid + hoses + caliper compliance behave like a
                        FIRST-ORDER LAG, so pressure cannot change instantly.
3. Pressure sensor    : a ratiometric transducer - 0.5 V at 0 bar and 4.5 V at
                        full scale - with gain error, offset error,
                        non-linearity and electrical noise.
4. ADC                : a 10-bit converter quantises the voltage.
5. Fault injection    : in "Sensor failure" mode the ADC code is corrupted
                        (open circuit, short, frozen output, dropouts).
6. Calibration        : an optional two-point (zero + span) correction.
7. Diagnostics        : plausibility checks decide whether the reading can be
                        trusted (range check, rate-of-change check, flatline).
8. Alarms             : low / high pressure alarms with delay (debounce) and
                        hysteresis; they are INHIBITED when the reading is
                        invalid, because an alarm on garbage data is itself
                        misleading.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, replace
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants (all illustrative)
# ---------------------------------------------------------------------------
DT_S = 0.1                  # sample period [s] -> 10 Hz sampling
FULL_SCALE_BAR = 160.0      # transducer span: 0 .. 160 bar

# Ratiometric output: 10 % .. 90 % of a 5 V supply. The 0.5 V "live zero"
# means a broken wire (0 V) can be told apart from a genuine 0 bar reading -
# the same idea as the 4 mA live zero of a 4-20 mA loop.
V_SUPPLY = 5.0
V_OUT_MIN = 0.5
V_OUT_MAX = 4.5

# Diagnostic band: a healthy sensor never leaves 0.25 .. 4.75 V. Anything
# outside it means open circuit, short to ground or short to supply.
V_DIAG_LOW = 0.25
V_DIAG_HIGH = 4.75

ADC_BITS = 10
ADC_LEVELS = 2 ** ADC_BITS - 1   # 1023 -> 4.89 mV per count with a 5 V reference

# Driver pedal cycle, expressed in samples so the timing is exact.
CYCLE_STEPS = 80     # 8.0 s per cycle
APPLY_START = 20     # pedal pressed at 2.0 s
RAMP_STEPS = 6       # 0.6 s apply / release ramp
HOLD_END = 50        # pedal released at 5.0 s
DEMAND_RANGE_BAR = (58.0, 72.0)   # per-cycle driver demand ("nominal braking")

# Hydraulic lag: p[n+1] = p[n] + alpha * (target - p[n]), alpha = 1 - exp(-dt/tau)
HYDRAULIC_TAU_S = 0.12
LAG_ALPHA = 1.0 - math.exp(-DT_S / HYDRAULIC_TAU_S)

# Mode multipliers applied to the driver demand.
LOW_FACTOR = 0.35     # e.g. leak or air in the line -> only ~20-25 bar
HIGH_FACTOR = 1.85    # e.g. over-boost -> ~107-133 bar

DROPOUT_PROB = 0.25   # intermittent-fault model: 25 % of samples read 0 V

MAX_HISTORY = 12000   # keep the last 20 minutes of samples
MAX_EVENTS = 1000

# Simulation modes
MODE_NORMAL = "Normal pressure"
MODE_LOW = "Low pressure"
MODE_HIGH = "Excessive pressure"
MODE_SENSOR = "Sensor failure"
MODES = [MODE_NORMAL, MODE_LOW, MODE_HIGH, MODE_SENSOR]

# Sensor-failure sub-types
FAULT_OPEN = "Open circuit / short to ground (0 V)"
FAULT_SHORT = "Short to supply (5 V)"
FAULT_STUCK = "Stuck / frozen output"
FAULT_INTERMITTENT = "Intermittent dropouts"
FAULT_TYPES = [FAULT_OPEN, FAULT_SHORT, FAULT_STUCK, FAULT_INTERMITTENT]

# Alarm identifiers
ALARM_LOW = "LOW_PRESSURE"
ALARM_HIGH = "HIGH_PRESSURE"
ALARM_SENSOR = "SENSOR_FAULT"
ALARM_CODES = [ALARM_LOW, ALARM_HIGH, ALARM_SENSOR]

HISTORY_COLUMNS = [
    "time_s", "true_bar", "adc_code", "voltage_v", "measured_bar", "valid",
    "brake_on", "mode", "low_alarm", "high_alarm", "sensor_fault",
    "fault_reason",
]
EVENT_COLUMNS = ["time_s", "severity", "event", "detail", "code", "action"]


# ---------------------------------------------------------------------------
# Configuration objects
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Thresholds:
    """Alarm settings (ILLUSTRATIVE demonstration values).

    low_bar        : minimum acceptable pressure WHILE THE PEDAL IS PRESSED.
                     (0 bar with the pedal released is perfectly normal, so the
                     low alarm is a pressure-versus-demand plausibility check.)
    high_bar       : maximum acceptable pressure at any time.
    hysteresis_bar : dead band that stops an alarm chattering on/off when the
                     reading hovers near a threshold. An alarm clears only after
                     the value has moved this far back inside the limit.
    alarm_delay_s  : the condition must persist this long before the alarm
                     raises (debounce) - rejects the harmless transient while
                     pressure ramps up after the pedal is pressed.
    """
    low_bar: float = 40.0
    high_bar: float = 100.0
    hysteresis_bar: float = 5.0
    alarm_delay_s: float = 1.0


@dataclass(frozen=True)
class SensorModel:
    """Imperfect pressure transducer.

    gain_error_pct     : span error. +3 % means it reads 3 % too high at every
                         pressure, so the error grows with pressure.
    offset_bar         : zero error. +1.5 means it reads 1.5 bar even at 0 bar.
    nonlinearity_pct_fs: bow in the transfer curve (zero at both ends, maximum
                         at mid-range). A two-point calibration cannot remove it.
    noise_bar_rms      : electrical noise, expressed in bar (1 sigma).
    """
    gain_error_pct: float = 3.0
    offset_bar: float = 1.5
    nonlinearity_pct_fs: float = 0.3
    noise_bar_rms: float = 0.4

    def output_voltage(self, p_true_bar: float, noise_bar: float = 0.0) -> float:
        """True pressure -> sensor output voltage (clamped to the 0..5 V rail)."""
        x = min(max(p_true_bar / FULL_SCALE_BAR, 0.0), 1.0)
        bow_bar = self.nonlinearity_pct_fs / 100.0 * FULL_SCALE_BAR * 4.0 * x * (1.0 - x)
        p_eff = p_true_bar * (1.0 + self.gain_error_pct / 100.0) + self.offset_bar + bow_bar + noise_bar
        v = V_OUT_MIN + (V_OUT_MAX - V_OUT_MIN) * p_eff / FULL_SCALE_BAR
        return min(max(v, 0.0), V_SUPPLY)


@dataclass(frozen=True)
class DiagnosticsConfig:
    """Sensor plausibility-check settings (ILLUSTRATIVE)."""
    max_rate_bar_s: float = 400.0   # faster than the hydraulics can physically move
    bad_window: int = 10            # look at the last 10 samples ...
    bad_limit: int = 3              # ... fault if 3 or more were bad ("3-of-10" filter)
    flatline_window: int = 20       # 2 s of identical ADC codes = frozen sensor
    clear_good_steps: int = 20      # 2 s of clean data before a fault may clear


@dataclass(frozen=True)
class Calibration:
    """Two-point correction applied in software: p_corrected = gain * p + offset."""
    gain: float = 1.0
    offset_bar: float = 0.0

    def apply(self, p_bar: float) -> float:
        return self.gain * p_bar + self.offset_bar


# ---------------------------------------------------------------------------
# Measurement-chain helpers
# ---------------------------------------------------------------------------
def adc_code(voltage: float) -> int:
    """Quantise a voltage into a 10-bit ADC code (0..1023)."""
    code = int(round(voltage / V_SUPPLY * ADC_LEVELS))
    return min(max(code, 0), ADC_LEVELS)


def code_to_voltage(code: int) -> float:
    """ADC code back to volts (quantisation step = 5 V / 1023 ~ 4.89 mV)."""
    return code * V_SUPPLY / ADC_LEVELS


def voltage_to_bar_nominal(voltage: float) -> float:
    """The ECU's *nominal* transfer function: assumes a perfect sensor.

    Any gain / offset / non-linearity error in the real sensor therefore shows
    up as a measurement error unless calibration corrects it.
    """
    return (voltage - V_OUT_MIN) / (V_OUT_MAX - V_OUT_MIN) * FULL_SCALE_BAR


def read_through_chain(sensor: SensorModel, p_ref_bar: float,
                       rng: Optional[np.random.Generator] = None,
                       n_avg: int = 1) -> float:
    """Apply a known pressure and read it through sensor + ADC.

    Averaging n_avg samples reduces random noise by about sqrt(n_avg); this is
    why technicians let the reading settle and average it during calibration.
    """
    if rng is not None:
        noise = rng.normal(0.0, sensor.noise_bar_rms, n_avg)
    else:
        noise = np.zeros(n_avg)
    values = [
        voltage_to_bar_nominal(code_to_voltage(adc_code(sensor.output_voltage(p_ref_bar, float(n)))))
        for n in noise
    ]
    return float(np.mean(values))


def two_point_calibration(read_lo: float, read_hi: float,
                          ref_lo: float = 0.0, ref_hi: float = FULL_SCALE_BAR) -> Calibration:
    """Zero-and-span calibration from two reference points.

    The straight line through (read_lo -> ref_lo) and (read_hi -> ref_hi):
        gain   = (ref_hi - ref_lo) / (read_hi - read_lo)
        offset = ref_lo - gain * read_lo
    """
    gain = (ref_hi - ref_lo) / (read_hi - read_lo)
    return Calibration(gain=gain, offset_bar=ref_lo - gain * read_lo)


def calibrate_sensor(sensor: SensorModel) -> Calibration:
    """Run a noise-free two-point calibration (0 bar and full scale) on a sensor."""
    lo = read_through_chain(sensor, 0.0)
    hi = read_through_chain(sensor, FULL_SCALE_BAR)
    return two_point_calibration(lo, hi)


def run_calibration_demo(gain_error_pct: float = 3.0, offset_bar: float = 1.5,
                         nonlinearity_pct_fs: float = 0.3, noise_bar_rms: float = 0.4,
                         n_avg: int = 20, ref_points: Optional[List[float]] = None,
                         seed: int = 7) -> Tuple[pd.DataFrame, Calibration]:
    """Calibration demonstration against a (simulated) deadweight tester.

    Known reference pressures are applied; each reading is the average of
    n_avg samples. The 'as-found' readings expose the sensor errors. A
    two-point calibration is computed from the lowest and highest points and
    applied, giving the 'as-left' readings. Residual as-left error is what a
    two-point calibration cannot remove: non-linearity, noise and quantisation.
    """
    if ref_points is None:
        ref_points = [float(p) for p in np.linspace(0.0, FULL_SCALE_BAR, 9)]
    sensor = SensorModel(gain_error_pct=gain_error_pct, offset_bar=offset_bar,
                         nonlinearity_pct_fs=nonlinearity_pct_fs, noise_bar_rms=noise_bar_rms)
    rng = np.random.default_rng(seed)
    readings = [read_through_chain(sensor, p, rng, n_avg) for p in ref_points]
    cal = two_point_calibration(readings[0], readings[-1], ref_points[0], ref_points[-1])

    rows = []
    for ref, found in zip(ref_points, readings):
        left = cal.apply(found)
        rows.append({
            "reference_bar": ref,
            "as_found_bar": found,
            "as_found_error_bar": found - ref,
            "as_found_error_pct_fs": (found - ref) / FULL_SCALE_BAR * 100.0,
            "as_left_bar": left,
            "as_left_error_bar": left - ref,
            "as_left_error_pct_fs": (left - ref) / FULL_SCALE_BAR * 100.0,
        })
    return pd.DataFrame(rows), cal


# ---------------------------------------------------------------------------
# The simulator
# ---------------------------------------------------------------------------
class BrakePressureSimulator:
    """Steps the whole measurement chain one sample at a time."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        # User-adjustable settings survive reset().
        self.mode = MODE_NORMAL
        self.fault_type = FAULT_OPEN
        self.thresholds = Thresholds()
        self.sensor = SensorModel()
        self.diag = DiagnosticsConfig()
        self.calibration_enabled = True
        self.calibration = calibrate_sensor(self.sensor)
        self.reset()

    # -- lifecycle -------------------------------------------------------
    def reset(self) -> None:
        """Clear all simulation state (settings are kept)."""
        self.rng = np.random.default_rng(self.seed)
        self.step_count = 0
        self.t = 0.0
        self.p_true = 0.0
        self._cycle_idx = -1
        self._demand_bar = 65.0
        self._frozen_code: Optional[int] = None
        self._prev_p_cal: Optional[float] = None
        self._codes: Deque[int] = deque(maxlen=self.diag.flatline_window)
        self._bad_flags: Deque[bool] = deque(maxlen=self.diag.bad_window)
        self._bad_reasons: Deque[Optional[str]] = deque(maxlen=self.diag.bad_window)
        self._good_run = 0
        self._sensor_latched = False
        self.fault_reason = ""
        self._alarms: Dict[str, Dict] = {
            ALARM_LOW: {"active": False, "count": 0},
            ALARM_HIGH: {"active": False, "count": 0},
        }
        self._history: Deque[Dict] = deque(maxlen=MAX_HISTORY)
        self._events: List[Dict] = []
        self.last_record: Optional[Dict] = None

    # -- settings --------------------------------------------------------
    def set_mode(self, mode: str, fault_type: Optional[str] = None) -> None:
        if mode not in MODES:
            raise ValueError(f"Unknown mode: {mode}")
        if fault_type is not None and fault_type not in FAULT_TYPES:
            raise ValueError(f"Unknown fault type: {fault_type}")
        if fault_type is not None and fault_type != self.fault_type:
            self.fault_type = fault_type
            self._frozen_code = None       # a new fault type starts fresh
        if mode != self.mode:
            suffix = f" ({self.fault_type})" if mode == MODE_SENSOR else ""
            self._log("INFO", "Mode changed", f"{self.mode} -> {mode}{suffix}")
            self.mode = mode
            self._frozen_code = None

    def set_thresholds(self, thresholds: Thresholds) -> None:
        if thresholds != self.thresholds:
            self.thresholds = thresholds
            self._log("INFO", "Thresholds updated",
                      f"low {thresholds.low_bar:.0f} bar, high {thresholds.high_bar:.0f} bar, "
                      f"hysteresis {thresholds.hysteresis_bar:.0f} bar, delay {thresholds.alarm_delay_s:.1f} s")

    def set_sensor(self, gain_error_pct: float, offset_bar: float, nonlinearity_pct_fs: float) -> None:
        new = replace(self.sensor, gain_error_pct=gain_error_pct, offset_bar=offset_bar,
                      nonlinearity_pct_fs=nonlinearity_pct_fs)
        if new != self.sensor:
            self.sensor = new
            self._log("INFO", "Sensor error model updated",
                      f"gain {gain_error_pct:+.1f} %, offset {offset_bar:+.1f} bar, "
                      f"non-linearity {nonlinearity_pct_fs:.2f} %FS")
            if self.calibration_enabled:
                # Model a technician re-running the calibration on the new sensor.
                self.calibration = calibrate_sensor(self.sensor)

    def set_calibration_enabled(self, enabled: bool) -> None:
        if enabled != self.calibration_enabled:
            self.calibration_enabled = enabled
            self.calibration = calibrate_sensor(self.sensor) if enabled else Calibration()
            self._log("INFO", "Calibration correction " + ("enabled" if enabled else "disabled"),
                      f"gain {self.calibration.gain:.4f}, offset {self.calibration.offset_bar:+.2f} bar")

    def log(self, severity: str, event: str, detail: str = "") -> None:
        """Public hook so the UI can record Start / Stop / Reset."""
        self._log(severity, event, detail)

    # -- stepping --------------------------------------------------------
    def advance(self, n_steps: int) -> None:
        for _ in range(n_steps):
            self.step()

    def step(self) -> Dict:
        """Simulate one sample and return its record."""
        k = self.step_count
        self.t = round(k * DT_S, 6)

        # 1. Driver demand and brake-light/pedal switch state.
        frac, brake_on = self._pedal_profile(k)
        factor = {MODE_NORMAL: 1.0, MODE_LOW: LOW_FACTOR, MODE_HIGH: HIGH_FACTOR,
                  MODE_SENSOR: 1.0}[self.mode]
        target = frac * self._demand_bar * factor

        # 2. Hydraulic dynamics (first-order lag).
        self.p_true += LAG_ALPHA * (target - self.p_true)

        # 3-4. Sensor (with noise) and ADC.
        noise = float(self.rng.normal(0.0, self.sensor.noise_bar_rms))
        code = adc_code(self.sensor.output_voltage(self.p_true, noise))

        # 5. Fault injection (only in Sensor-failure mode).
        code = self._apply_fault(code)
        v = code_to_voltage(code)

        # 6. Engineering-unit conversion + optional calibration.
        p_nom = voltage_to_bar_nominal(v)
        p_cal = self.calibration.apply(p_nom) if self.calibration_enabled else p_nom

        # 7. Diagnostics.
        out_of_range = v < V_DIAG_LOW or v > V_DIAG_HIGH
        rate_bad = (self._prev_p_cal is not None and
                    abs(p_cal - self._prev_p_cal) / DT_S > self.diag.max_rate_bar_s)
        self._prev_p_cal = p_cal
        bad = out_of_range or rate_bad
        reason_now = None
        if out_of_range:
            reason_now = "Signal outside 0.25-4.75 V diagnostic band (open/short circuit)"
        elif rate_bad:
            reason_now = "Implausible rate of change"
        self._bad_flags.append(bad)
        self._bad_reasons.append(reason_now)
        self._codes.append(code)

        circuit_fault = sum(self._bad_flags) >= self.diag.bad_limit
        flatline = (len(self._codes) == self.diag.flatline_window
                    and max(self._codes) - min(self._codes) == 0)
        self._update_sensor_latch(circuit_fault, flatline)

        # A reading is trusted only if this sample is in range AND no fault is latched.
        valid = (not out_of_range) and (not self._sensor_latched)
        measured = p_cal if valid else float("nan")

        # 8. Alarms (inhibited when the reading is not valid).
        th = self.thresholds
        need = max(1, math.ceil(th.alarm_delay_s / DT_S - 1e-9))
        self._update_pressure_alarm(
            ALARM_LOW, "WARNING", "Low pressure alarm", valid,
            cond_raise=brake_on and p_cal < th.low_bar,
            cond_hold=brake_on and p_cal < th.low_bar + th.hysteresis_bar,
            need=need,
            detail_raise=f"{p_cal:.1f} bar below {th.low_bar:.0f} bar with pedal applied",
            detail_clear=f"{p_cal:.1f} bar (pedal released or pressure recovered)")
        self._update_pressure_alarm(
            ALARM_HIGH, "CRITICAL", "High pressure alarm", valid,
            cond_raise=p_cal > th.high_bar,
            cond_hold=p_cal > th.high_bar - th.hysteresis_bar,
            need=need,
            detail_raise=f"{p_cal:.1f} bar above {th.high_bar:.0f} bar",
            detail_clear=f"{p_cal:.1f} bar (back below {th.high_bar - th.hysteresis_bar:.0f} bar)")

        record = {
            "time_s": self.t,
            "true_bar": self.p_true,
            "adc_code": code,
            "voltage_v": v,
            "measured_bar": measured,
            "valid": valid,
            "brake_on": brake_on,
            "mode": self.mode,
            "low_alarm": self._alarms[ALARM_LOW]["active"],
            "high_alarm": self._alarms[ALARM_HIGH]["active"],
            "sensor_fault": self._sensor_latched,
            "fault_reason": self.fault_reason if self._sensor_latched else "",
        }
        self._history.append(record)
        self.last_record = record
        self.step_count += 1
        return record

    # -- internals -------------------------------------------------------
    def _pedal_profile(self, k: int) -> Tuple[float, bool]:
        """Return (demand fraction 0..1, pedal-switch state) for sample k."""
        cycle_idx, ks = divmod(k, CYCLE_STEPS)
        if cycle_idx != self._cycle_idx:   # a new stop: the driver asks for a new pressure
            self._cycle_idx = cycle_idx
            self._demand_bar = float(self.rng.uniform(*DEMAND_RANGE_BAR))
        if ks < APPLY_START:
            return 0.0, False
        if ks < APPLY_START + RAMP_STEPS:
            return (ks - APPLY_START) / RAMP_STEPS, True
        if ks < HOLD_END:
            ripple = 1.0 + 0.01 * math.sin(2.0 * math.pi * ks * DT_S / 1.3)
            return ripple, True
        if ks < HOLD_END + RAMP_STEPS:
            return 1.0 - (ks - HOLD_END) / RAMP_STEPS, False
        return 0.0, False

    def _apply_fault(self, code: int) -> int:
        if self.mode != MODE_SENSOR:
            self._frozen_code = None
            return code
        if self.fault_type == FAULT_OPEN:
            return 0
        if self.fault_type == FAULT_SHORT:
            return ADC_LEVELS
        if self.fault_type == FAULT_STUCK:
            if self._frozen_code is None:
                self._frozen_code = code     # output freezes at its last value
            return self._frozen_code
        if self.fault_type == FAULT_INTERMITTENT:
            return 0 if self.rng.random() < DROPOUT_PROB else code
        return code

    def _update_sensor_latch(self, circuit_fault: bool, flatline: bool) -> None:
        """Latch the sensor fault on detection; release only after a clean spell."""
        condition = circuit_fault or flatline
        if condition:
            self._good_run = 0
            if circuit_fault:
                self.fault_reason = next((r for r in reversed(self._bad_reasons) if r),
                                         "Repeated bad samples")
            else:
                self.fault_reason = "Flatline: ADC output frozen (no noise or variation)"
            if not self._sensor_latched:
                self._sensor_latched = True
                self._log("CRITICAL", "Sensor fault detected", self.fault_reason,
                          ALARM_SENSOR, "RAISED")
        else:
            self._good_run += 1
            if self._sensor_latched and self._good_run >= self.diag.clear_good_steps:
                self._sensor_latched = False
                self.fault_reason = ""
                self._log("INFO", "Sensor fault cleared", "Signal plausible again",
                          ALARM_SENSOR, "CLEARED")

    def _update_pressure_alarm(self, code: str, severity: str, label: str, valid: bool,
                               cond_raise: bool, cond_hold: bool, need: int,
                               detail_raise: str, detail_clear: str) -> None:
        state = self._alarms[code]
        if not valid:
            state["count"] = 0
            if state["active"]:
                state["active"] = False
                self._log("INFO", f"{label} cleared",
                          "Inhibited: measurement invalid (sensor fault)", code, "CLEARED")
            return
        if state["active"]:
            if not cond_hold:
                state["active"] = False
                self._log("INFO", f"{label} cleared", detail_clear, code, "CLEARED")
            return
        if cond_raise:
            state["count"] += 1
            if state["count"] >= need:
                state["active"] = True
                state["count"] = 0
                self._log(severity, f"{label} raised", detail_raise, code, "RAISED")
        else:
            state["count"] = 0

    def _log(self, severity: str, event: str, detail: str = "",
             code: str = "", action: str = "") -> None:
        self._events.append({"time_s": self.t, "severity": severity, "event": event,
                             "detail": detail, "code": code, "action": action})
        if len(self._events) > MAX_EVENTS:
            del self._events[: len(self._events) - MAX_EVENTS]

    # -- read-outs -------------------------------------------------------
    @property
    def active_alarms(self) -> List[str]:
        out = [c for c in (ALARM_LOW, ALARM_HIGH) if self._alarms[c]["active"]]
        if self._sensor_latched:
            out.append(ALARM_SENSOR)
        return out

    @property
    def status(self) -> str:
        """Single headline condition, most serious first."""
        if self._sensor_latched:
            return "SENSOR FAULT"
        if self._alarms[ALARM_HIGH]["active"]:
            return "HIGH PRESSURE"
        if self._alarms[ALARM_LOW]["active"]:
            return "LOW PRESSURE"
        return "NORMAL"

    def history_df(self, window_s: Optional[float] = None) -> pd.DataFrame:
        rows = list(self._history)
        if window_s is not None:
            rows = rows[-(int(round(window_s / DT_S)) + 1):]
        if not rows:
            return pd.DataFrame(columns=HISTORY_COLUMNS)
        return pd.DataFrame(rows, columns=HISTORY_COLUMNS)

    def events_df(self) -> pd.DataFrame:
        if not self._events:
            return pd.DataFrame(columns=EVENT_COLUMNS)
        return pd.DataFrame(self._events, columns=EVENT_COLUMNS)

    def raised_times(self, code: str) -> List[float]:
        return [e["time_s"] for e in self._events if e["code"] == code and e["action"] == "RAISED"]

    def cleared_times(self, code: str) -> List[float]:
        return [e["time_s"] for e in self._events if e["code"] == code and e["action"] == "CLEARED"]


# ---------------------------------------------------------------------------
# Headless scenario runner (drives the "Test results" tab and the unit tests)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Segment:
    """Hold one mode for duration_s seconds of simulation time."""
    duration_s: float
    mode: str
    fault_type: str = FAULT_OPEN


def run_plan(plan: List[Segment], thresholds: Optional[Thresholds] = None,
             seed: int = 42) -> BrakePressureSimulator:
    sim = BrakePressureSimulator(seed=seed)
    if thresholds is not None:
        sim.set_thresholds(thresholds)
    for seg in plan:
        sim.set_mode(seg.mode, seg.fault_type)
        sim.advance(int(round(seg.duration_s / DT_S)))
    return sim


# The pedal is first applied at APPLY_START * DT_S seconds.
FIRST_BRAKE_S = APPLY_START * DT_S
FAULT_INJECT_S = 15.0   # injected while the pedal is released (rest period)


def _pressure_alarm_raises(sim: BrakePressureSimulator, after_s: float = 0.0) -> List[Tuple[str, float]]:
    out = []
    for code in (ALARM_LOW, ALARM_HIGH):
        out += [(code, t) for t in sim.raised_times(code) if t >= after_s]
    return out


def run_scenario_suite() -> Tuple[pd.DataFrame, Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]]:
    """Run every demonstration scenario headless (fixed seed) and grade it.

    Pass/fail criteria are defined by this software simulation's illustrative
    thresholds. They are NOT automotive validation criteria.
    """
    rows: List[Dict] = []
    traces: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}

    def add(name, condition, expected, observed, latency, passed, sim=None):
        rows.append({
            "Scenario": name,
            "Condition applied": condition,
            "Expected response": expected,
            "Observed response": observed,
            "Detection latency (s)": None if latency is None else round(latency, 2),
            "Result": "PASS" if passed else "FAIL",
        })
        if sim is not None:
            traces[name] = (sim.history_df(), sim.events_df())

    # 1. Normal ----------------------------------------------------------
    name = "1. Normal pressure"
    sim = run_plan([Segment(60, MODE_NORMAL)])
    df = sim.history_df()
    n_alarms = len(sim.raised_times(ALARM_LOW) + sim.raised_times(ALARM_HIGH) + sim.raised_times(ALARM_SENSOR))
    peak = float(df["measured_bar"].max())
    add(name, "Normal braking cycles for 60 s",
        "No alarms; peak stays below the 100 bar high threshold",
        f"{n_alarms} alarms; peak {peak:.1f} bar", None, n_alarms == 0 and peak < 100.0, sim)

    # 2. Low pressure ----------------------------------------------------
    name = "2. Low pressure"
    sim = run_plan([Segment(60, MODE_LOW)])
    low = sim.raised_times(ALARM_LOW)
    others = sim.raised_times(ALARM_HIGH) + sim.raised_times(ALARM_SENSOR)
    latency = (low[0] - FIRST_BRAKE_S) if low else None
    peak = float(sim.history_df()["measured_bar"].max())
    add(name, "Pressure limited to ~35 % of demand (simulated leak / air in line)",
        "Low-pressure alarm within 1.5 s of first brake application; no other alarms",
        f"Low alarm at {low[0]:.1f} s; peak {peak:.1f} bar; other alarms: {len(others)}" if low
        else "No low-pressure alarm",
        latency, bool(low) and not others and latency <= 1.5, sim)

    # 3. Excessive pressure ---------------------------------------------
    name = "3. Excessive pressure"
    sim = run_plan([Segment(60, MODE_HIGH)])
    high = sim.raised_times(ALARM_HIGH)
    others = sim.raised_times(ALARM_LOW) + sim.raised_times(ALARM_SENSOR)
    latency = (high[0] - FIRST_BRAKE_S) if high else None
    peak = float(sim.history_df()["measured_bar"].max())
    add(name, "Pressure raised to ~185 % of demand (simulated over-boost)",
        "High-pressure alarm within 2.5 s of first brake application; no other alarms",
        f"High alarm at {high[0]:.1f} s; peak {peak:.1f} bar; other alarms: {len(others)}" if high
        else "No high-pressure alarm",
        latency, bool(high) and not others and latency <= 2.5, sim)

    # 4-7. Sensor failures (injected during a rest period) ---------------
    fault_cases = [
        ("4. Sensor failure: open circuit", FAULT_OPEN, 0.5,
         "Output reads 0 V (below the 0.25 V diagnostic band)"),
        ("5. Sensor failure: short to supply", FAULT_SHORT, 0.5,
         "Output reads 5 V (above the 4.75 V diagnostic band)"),
        ("6. Sensor failure: stuck output", FAULT_STUCK, 2.5,
         "Output freezes at its last value (undetectable by range check)"),
        ("7. Sensor failure: intermittent", FAULT_INTERMITTENT, 5.0,
         "25 % of samples drop to 0 V (loose connector)"),
    ]
    for name, ftype, limit, condition in fault_cases:
        sim = run_plan([Segment(FAULT_INJECT_S, MODE_NORMAL), Segment(45, MODE_SENSOR, ftype)])
        raised = [t for t in sim.raised_times(ALARM_SENSOR) if t >= FAULT_INJECT_S]
        latency = (raised[0] - FAULT_INJECT_S) if raised else None
        spurious = _pressure_alarm_raises(sim, FAULT_INJECT_S)
        df = sim.history_df()
        after = df[df["time_s"] >= (raised[0] if raised else 1e9)]
        invalid_ok = bool(len(after)) and bool(after["measured_bar"].isna().mean() > 0.5)
        passed = bool(raised) and latency <= limit and not spurious and invalid_ok
        add(name, condition + f" at t = {FAULT_INJECT_S:.0f} s",
            f"Sensor-fault alarm within {limit:g} s; reading flagged invalid; no spurious pressure alarms",
            (f"Fault at +{latency:.1f} s ({sim.fault_reason or 'latched'}); spurious pressure alarms: {len(spurious)}"
             if raised else "Sensor fault never detected"),
            latency, passed, sim)

    # 8. Recovery ---------------------------------------------------------
    name = "8. Sensor fault recovery"
    sim = run_plan([Segment(15, MODE_NORMAL), Segment(10, MODE_SENSOR, FAULT_OPEN), Segment(35, MODE_NORMAL)])
    raised = sim.raised_times(ALARM_SENSOR)
    cleared = [t for t in sim.cleared_times(ALARM_SENSOR) if t >= 25.0]
    spurious = _pressure_alarm_raises(sim, 0.0)
    latency = (cleared[0] - 25.0) if cleared else None
    add(name, "Open-circuit fault for 10 s, then the sensor is restored at t = 25 s",
        "Fault raised, then cleared within 3 s of recovery (2 s clean-signal hold + 1 s diagnostic window); no pressure alarms",
        (f"Raised at {raised[0]:.1f} s, cleared at {cleared[0]:.1f} s; spurious pressure alarms: {len(spurious)}"
         if raised and cleared else "Fault did not raise and clear as expected"),
        latency, bool(raised) and bool(cleared) and latency <= 3.0 and not spurious, sim)

    # 9. Configurable threshold -------------------------------------------
    name = "9. Lowered high threshold"
    sim = run_plan([Segment(30, MODE_NORMAL)], thresholds=Thresholds(high_bar=50.0))
    high = sim.raised_times(ALARM_HIGH)
    latency = (high[0] - FIRST_BRAKE_S) if high else None
    add(name, "Normal braking, but high threshold lowered to 50 bar",
        "High-pressure alarm raised even though the pressure is normal (limits are configurable)",
        f"High alarm at {high[0]:.1f} s" if high else "No alarm",
        latency, bool(high) and latency <= 2.5, sim)

    # 10. Alarm-delay demonstration ------------------------------------------
    name = "10. Alarm delay set to 0 s"
    sim = run_plan([Segment(30, MODE_NORMAL)], thresholds=Thresholds(alarm_delay_s=0.0))
    low = sim.raised_times(ALARM_LOW)
    add(name, "Normal braking with the alarm delay (debounce) removed",
        "Nuisance low-pressure alarms appear while pressure ramps up - shows why the delay exists",
        f"{len(low)} nuisance low-pressure alarms in 30 s", None, len(low) >= 1, sim)

    # 11. Calibration ---------------------------------------------------------
    cal_df, cal = run_calibration_demo()
    found = float(cal_df["as_found_error_pct_fs"].abs().max())
    left = float(cal_df["as_left_error_pct_fs"].abs().max())
    add("11. Two-point calibration",
        "Sensor with +3 % gain error, +1.5 bar offset and 0.3 %FS non-linearity",
        "As-left worst-case error <= 0.5 %FS and smaller than as-found",
        f"Worst-case error {found:.2f} %FS as-found -> {left:.2f} %FS as-left "
        f"(gain {cal.gain:.4f}, offset {cal.offset_bar:+.2f} bar)",
        None, left <= 0.5 and left < found)

    return pd.DataFrame(rows), traces

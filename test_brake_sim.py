"""
Unit tests for brake_sim.py (the simulation engine).

Run from the project folder:
    python -m unittest -v test_brake_sim

These tests check the SIMULATION LOGIC only. They do not validate anything
about real automotive braking systems, and the Streamlit user interface is not
covered by them.
"""

import math
import unittest

import numpy as np

import brake_sim as bs
from brake_sim import (
    ALARM_HIGH, ALARM_LOW, ALARM_SENSOR, FAULT_INTERMITTENT, FAULT_OPEN,
    FAULT_SHORT, FAULT_STUCK, MODE_HIGH, MODE_LOW, MODE_NORMAL, MODE_SENSOR,
    BrakePressureSimulator, Segment, SensorModel, Thresholds, run_plan,
)

IDEAL = SensorModel(gain_error_pct=0.0, offset_bar=0.0, nonlinearity_pct_fs=0.0, noise_bar_rms=0.0)


class TestMeasurementChain(unittest.TestCase):
    def test_sensor_transfer_function_endpoints(self):
        self.assertAlmostEqual(IDEAL.output_voltage(0.0), 0.5)
        self.assertAlmostEqual(IDEAL.output_voltage(80.0), 2.5)
        self.assertAlmostEqual(IDEAL.output_voltage(160.0), 4.5)

    def test_gain_and_offset_errors_move_the_output(self):
        s = SensorModel(gain_error_pct=3.0, offset_bar=1.5, nonlinearity_pct_fs=0.0, noise_bar_rms=0.0)
        # at 0 bar: only the offset shows; at 100 bar: 3 % of 100 plus the offset
        self.assertAlmostEqual(bs.voltage_to_bar_nominal(s.output_voltage(0.0)), 1.5, places=6)
        self.assertAlmostEqual(bs.voltage_to_bar_nominal(s.output_voltage(100.0)), 104.5, places=6)

    def test_adc_range_and_resolution(self):
        self.assertEqual(bs.adc_code(0.0), 0)
        self.assertEqual(bs.adc_code(5.0), 1023)
        self.assertEqual(bs.adc_code(9.0), 1023)     # clamps above the reference
        self.assertEqual(bs.adc_code(-1.0), 0)       # clamps below zero
        self.assertAlmostEqual(bs.code_to_voltage(1) - bs.code_to_voltage(0), 5.0 / 1023)

    def test_roundtrip_error_is_within_one_adc_count(self):
        # one count is ~0.195 bar, so half a count is ~0.1 bar
        for p in (0.0, 25.0, 60.0, 100.0, 160.0):
            v = IDEAL.output_voltage(p)
            back = bs.voltage_to_bar_nominal(bs.code_to_voltage(bs.adc_code(v)))
            self.assertLess(abs(back - p), 0.11)

    def test_hydraulic_lag_alpha(self):
        self.assertTrue(0.0 < bs.LAG_ALPHA < 1.0)


class TestCalibration(unittest.TestCase):
    def test_two_point_calibration_maps_endpoints_exactly(self):
        cal = bs.two_point_calibration(1.5, 169.7)   # reads 1.5 at 0 bar, 169.7 at 160 bar
        self.assertAlmostEqual(cal.apply(1.5), 0.0, places=9)
        self.assertAlmostEqual(cal.apply(169.7), 160.0, places=9)

    def test_calibration_reduces_error(self):
        df, cal = bs.run_calibration_demo()
        found = df["as_found_error_pct_fs"].abs().max()
        left = df["as_left_error_pct_fs"].abs().max()
        self.assertGreater(found, 3.0)     # +3 % gain and +1.5 bar offset are large
        self.assertLess(left, found)
        self.assertLessEqual(left, 0.5)

    def test_calibration_corrects_zero_and_span(self):
        df, _ = bs.run_calibration_demo()
        self.assertLess(abs(df["as_left_error_bar"].iloc[0]), 0.01)
        self.assertLess(abs(df["as_left_error_bar"].iloc[-1]), 0.01)

    def test_calibration_demo_is_repeatable(self):
        a, _ = bs.run_calibration_demo(seed=3)
        b, _ = bs.run_calibration_demo(seed=3)
        self.assertTrue(a.equals(b))

    def test_averaging_reduces_noise(self):
        s = SensorModel(gain_error_pct=0.0, offset_bar=0.0, nonlinearity_pct_fs=0.0, noise_bar_rms=0.4)
        rng = np.random.default_rng(0)
        single = [bs.read_through_chain(s, 60.0, rng, 1) for _ in range(300)]
        avg = [bs.read_through_chain(s, 60.0, rng, 25) for _ in range(300)]
        self.assertLess(np.std(avg), np.std(single) / 2.5)

    def test_live_calibration_toggle_changes_zero_reading(self):
        sim = BrakePressureSimulator()
        sim.set_calibration_enabled(False)
        sim.advance(15)   # pedal still released
        uncal = sim.history_df()["measured_bar"].mean()
        sim2 = BrakePressureSimulator()
        sim2.advance(15)
        cal = sim2.history_df()["measured_bar"].mean()
        self.assertGreater(uncal, 1.0)          # +1.5 bar zero error visible
        self.assertLess(abs(cal), 0.5)          # corrected


class TestPressureScenarios(unittest.TestCase):
    def test_normal_mode_has_no_alarms_across_seeds(self):
        for seed in range(10):
            sim = run_plan([Segment(120, MODE_NORMAL)], seed=seed)
            self.assertEqual(sim.raised_times(ALARM_LOW), [], f"seed {seed}")
            self.assertEqual(sim.raised_times(ALARM_HIGH), [], f"seed {seed}")
            self.assertEqual(sim.raised_times(ALARM_SENSOR), [], f"seed {seed}")

    def test_low_mode_raises_only_low_alarm(self):
        for seed in range(5):
            sim = run_plan([Segment(60, MODE_LOW)], seed=seed)
            self.assertTrue(sim.raised_times(ALARM_LOW), f"seed {seed}")
            self.assertEqual(sim.raised_times(ALARM_HIGH), [])
            self.assertEqual(sim.raised_times(ALARM_SENSOR), [])

    def test_high_mode_raises_only_high_alarm(self):
        for seed in range(5):
            sim = run_plan([Segment(60, MODE_HIGH)], seed=seed)
            self.assertTrue(sim.raised_times(ALARM_HIGH), f"seed {seed}")
            self.assertEqual(sim.raised_times(ALARM_LOW), [])
            self.assertEqual(sim.raised_times(ALARM_SENSOR), [])

    def test_alarms_clear_when_pedal_released(self):
        sim = run_plan([Segment(60, MODE_HIGH)])
        raised = sim.raised_times(ALARM_HIGH)
        cleared = sim.cleared_times(ALARM_HIGH)
        self.assertGreaterEqual(len(cleared), len(raised) - 1)
        self.assertLessEqual(len(raised) - len(cleared), 1)   # at most one still active at the end

    def test_alarm_delay_rejects_ramp_up_transient(self):
        with_delay = run_plan([Segment(30, MODE_NORMAL)], thresholds=Thresholds(alarm_delay_s=1.0))
        without = run_plan([Segment(30, MODE_NORMAL)], thresholds=Thresholds(alarm_delay_s=0.0))
        self.assertEqual(with_delay.raised_times(ALARM_LOW), [])
        self.assertTrue(without.raised_times(ALARM_LOW))

    def test_lowered_high_threshold_raises_alarm_in_normal_mode(self):
        sim = run_plan([Segment(30, MODE_NORMAL)], thresholds=Thresholds(high_bar=50.0))
        self.assertTrue(sim.raised_times(ALARM_HIGH))

    def test_hysteresis_keeps_alarm_latched_until_clear_band(self):
        # Hysteresis of 40 bar means the alarm cannot clear until pressure < 100 - 40 = 60 bar.
        wide = run_plan([Segment(12, MODE_HIGH)], thresholds=Thresholds(hysteresis_bar=40.0))
        narrow = run_plan([Segment(12, MODE_HIGH)], thresholds=Thresholds(hysteresis_bar=1.0))
        t_wide = wide.cleared_times(ALARM_HIGH)[0]
        t_narrow = narrow.cleared_times(ALARM_HIGH)[0]
        self.assertGreater(t_wide, t_narrow)

    def test_threshold_and_mode_changes_are_logged(self):
        sim = BrakePressureSimulator()
        sim.set_thresholds(Thresholds(high_bar=90.0))
        sim.set_mode(MODE_LOW)
        events = sim.events_df()["event"].tolist()
        self.assertIn("Thresholds updated", events)
        self.assertIn("Mode changed", events)


class TestSensorFaults(unittest.TestCase):
    def _fault_run(self, ftype, seed=42, inject=15.0, total=45.0):
        return run_plan([Segment(inject, MODE_NORMAL), Segment(total, MODE_SENSOR, ftype)], seed=seed)

    def test_open_circuit_detected_quickly(self):
        sim = self._fault_run(FAULT_OPEN)
        t = sim.raised_times(ALARM_SENSOR)[0]
        self.assertLessEqual(t - 15.0, 0.5)
        self.assertIn("diagnostic band", sim.events_df().query("code == 'SENSOR_FAULT'")["detail"].iloc[0])

    def test_short_to_supply_detected_quickly(self):
        sim = self._fault_run(FAULT_SHORT)
        self.assertLessEqual(sim.raised_times(ALARM_SENSOR)[0] - 15.0, 0.5)

    def test_stuck_output_detected_by_flatline_check(self):
        sim = self._fault_run(FAULT_STUCK)
        t = sim.raised_times(ALARM_SENSOR)[0]
        self.assertLessEqual(t - 15.0, 2.5)
        self.assertGreater(t - 15.0, 1.0)     # a range check could never catch this one
        self.assertIn("Flatline", sim.events_df().query("code == 'SENSOR_FAULT'")["detail"].iloc[0])

    def test_intermittent_fault_detected_for_many_seeds(self):
        for seed in range(30):
            sim = self._fault_run(FAULT_INTERMITTENT, seed=seed)
            raised = sim.raised_times(ALARM_SENSOR)
            self.assertTrue(raised, f"seed {seed}: intermittent fault never detected")
            self.assertLessEqual(raised[0] - 15.0, 5.0, f"seed {seed}")

    def test_reading_is_flagged_invalid_after_detection(self):
        sim = self._fault_run(FAULT_OPEN)
        df = sim.history_df()
        late = df[df["time_s"] > 16.0]
        self.assertTrue(late["measured_bar"].isna().all())
        self.assertFalse(late["valid"].any())

    def test_pressure_alarms_inhibited_during_sensor_fault(self):
        sim = self._fault_run(FAULT_OPEN)
        self.assertEqual([t for t in sim.raised_times(ALARM_LOW) if t >= 15.0], [])
        self.assertEqual([t for t in sim.raised_times(ALARM_HIGH) if t >= 15.0], [])

    def test_stuck_sensor_during_braking_freezes_reading_then_flags_it(self):
        # Fault begins at t = 3.5 s, mid-hold. The reading should freeze at a high value.
        sim = run_plan([Segment(3.5, MODE_NORMAL), Segment(10, MODE_SENSOR, FAULT_STUCK)])
        df = sim.history_df()
        window = df[(df["time_s"] >= 3.5) & (df["time_s"] < 5.0)]
        self.assertEqual(window["adc_code"].nunique(), 1)             # frozen output
        self.assertGreater(window["measured_bar"].iloc[0], 40.0)      # frozen at a braking value
        self.assertTrue(window["valid"].all())                        # not yet detected: latency
        self.assertTrue(sim.raised_times(ALARM_SENSOR))               # but detected eventually

    def test_fault_recovers_after_sensor_restored(self):
        sim = run_plan([Segment(15, MODE_NORMAL), Segment(10, MODE_SENSOR, FAULT_OPEN),
                        Segment(35, MODE_NORMAL)])
        cleared = [t for t in sim.cleared_times(ALARM_SENSOR) if t >= 25.0]
        self.assertTrue(cleared)
        # Design bound: 2.0 s clean-signal hold + 1.0 s for old bad samples to leave the window.
        self.assertLessEqual(cleared[0] - 25.0, 3.0)
        df = sim.history_df()
        self.assertTrue(df[df["time_s"] > 28.0]["valid"].all())

    def test_changing_fault_type_refreezes_stuck_output(self):
        sim = BrakePressureSimulator()
        sim.set_mode(MODE_SENSOR, FAULT_STUCK)
        sim.advance(5)
        first = sim._frozen_code
        sim.set_mode(MODE_SENSOR, FAULT_OPEN)
        sim.set_mode(MODE_SENSOR, FAULT_STUCK)
        sim.advance(1)
        self.assertIsNotNone(sim._frozen_code)
        self.assertIsNotNone(first)


class TestEngineBehaviour(unittest.TestCase):
    def test_runs_are_repeatable_for_a_fixed_seed(self):
        a = run_plan([Segment(30, MODE_NORMAL)], seed=5).history_df()
        b = run_plan([Segment(30, MODE_NORMAL)], seed=5).history_df()
        self.assertTrue(a.equals(b))

    def test_reset_clears_state_but_keeps_settings(self):
        sim = BrakePressureSimulator()
        sim.set_thresholds(Thresholds(high_bar=90.0))
        sim.set_mode(MODE_HIGH)
        sim.advance(200)
        self.assertTrue(len(sim.events_df()) > 0)
        sim.reset()
        self.assertEqual(sim.step_count, 0)
        self.assertEqual(len(sim.history_df()), 0)
        self.assertEqual(len(sim.events_df()), 0)
        self.assertEqual(sim.thresholds.high_bar, 90.0)
        self.assertEqual(sim.mode, MODE_HIGH)

    def test_history_dataframe_shape_and_window(self):
        sim = BrakePressureSimulator()
        sim.advance(100)
        self.assertEqual(list(sim.history_df().columns), bs.HISTORY_COLUMNS)
        self.assertEqual(len(sim.history_df()), 100)
        win = sim.history_df(window_s=5.0)
        self.assertLessEqual(len(win), 51)
        self.assertAlmostEqual(win["time_s"].iloc[-1], 9.9, places=6)

    def test_pressure_never_negative_physically_and_pedal_cycle_timing(self):
        sim = run_plan([Segment(16, MODE_NORMAL)])
        df = sim.history_df()
        self.assertTrue((df["true_bar"] >= 0).all())
        brake = df[df["brake_on"]]["time_s"]
        self.assertAlmostEqual(brake.iloc[0], 2.0, places=6)
        self.assertAlmostEqual(df[(df["time_s"] > 2.5) & (df["time_s"] < 4.9)]["true_bar"].min() > 50.0, True)

    def test_invalid_arguments_rejected(self):
        sim = BrakePressureSimulator()
        with self.assertRaises(ValueError):
            sim.set_mode("Nonsense")
        with self.assertRaises(ValueError):
            sim.set_mode(MODE_SENSOR, "Nonsense")

    def test_status_priority(self):
        sim = BrakePressureSimulator()
        self.assertEqual(sim.status, "NORMAL")
        sim.set_mode(MODE_SENSOR, FAULT_OPEN)
        sim.advance(10)
        self.assertEqual(sim.status, "SENSOR FAULT")


class TestScenarioSuite(unittest.TestCase):
    def test_every_scenario_passes(self):
        df, traces = bs.run_scenario_suite()
        self.assertEqual(len(df), 11)
        failed = df[df["Result"] != "PASS"]
        self.assertTrue(failed.empty, f"failed scenarios:\n{failed.to_string()}")
        self.assertEqual(len(traces), 10)     # calibration row has no time trace

    def test_scenario_columns(self):
        df, _ = bs.run_scenario_suite()
        self.assertEqual(list(df.columns), ["Scenario", "Condition applied", "Expected response",
                                            "Observed response", "Detection latency (s)", "Result"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

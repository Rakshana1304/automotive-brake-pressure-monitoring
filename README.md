# Automotive Brake Pressure Monitoring and Fault Detection System

A software-only instrumentation demonstration built with **Python, Streamlit, Plotly, Pandas and NumPy**.
It simulates a hydraulic brake-line pressure transducer and shows how a measurement system turns a
raw sensor signal into a trusted reading, raises alarms, and detects its own sensor failures.

> **Important:** this is an **educational software simulation**. All data is synthetic. Every threshold,
> sensor parameter and timing is **illustrative**. It is **not** a validated automotive braking safety
> system and must not be used for any real braking-system decision.

---

## What it shows

| Area | What the dashboard does |
|---|---|
| Live monitor | Dial gauge and numeric readout, pressure-versus-time trend that updates while the simulation runs, status banner, pedal state, raw sensor voltage and ADC code |
| Simulation modes | Normal pressure, low pressure, excessive pressure, sensor failure (open circuit, short to supply, stuck output, intermittent dropouts) |
| Alarms | Low-pressure, high-pressure and sensor-fault alarms, plus a time-stamped event history |
| Controls | Start, Stop and Reset buttons; mode can be switched while running |
| Thresholds | Adjustable low limit, high limit, hysteresis and alarm delay |
| Calibration | Adjustable sensor gain, offset and non-linearity errors; a calibration tab that runs a two-point (zero and span) calibration and shows the error before and after |
| Test results | Eleven scenarios run by the engine with no UI and graded against expected behaviour, plus a trace viewer |

## Quick start

Developed and tested on **Python 3.12**; Python 3.10 or newer is expected to work.

```bash
# 1. Go to the project folder
cd brake-pressure-monitor

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows (Command Prompt or PowerShell)

# 3. Install the dependencies
pip install -r requirements.txt

# 4. Start the dashboard
streamlit run app.py
```

Streamlit prints a local address (normally `http://localhost:8501`) and opens it in your browser.
Stop the server with `Ctrl+C`.

**Troubleshooting**

- `streamlit: command not found`: activate the virtual environment first, or use `python -m streamlit run app.py`.
- Live updates require **Streamlit 1.37 or newer** (`st.fragment(run_every=...)`). Check with `pip show streamlit`.
- Run the command from inside the project folder so `app.py` can import `brake_sim.py`.

## Running the tests

The engine tests need no extra packages:

```bash
python -m unittest -v test_brake_sim
```

They cover the sensor and ADC maths, calibration, each simulation mode, alarm delay and hysteresis,
every sensor-fault type (including detection-latency bounds and recovery), reset behaviour,
repeatability and the full scenario suite. They test the **simulation logic only**; the Streamlit
interface is not covered by them.

## A suggested demonstration (about five minutes)

1. Press **Start** in the sidebar. Normal braking cycles repeat every 8 seconds. Point out the pedal
   state, the 0.5-4.5 V sensor output and the ADC code beside the pressure reading.
2. Switch to **Low pressure**. After the pedal is pressed the pressure stays low and a low-pressure alarm
   appears in the alarm panel and the event history.
3. Switch to **Excessive pressure**. The high-pressure alarm raises, and clears when the pedal is released.
4. Switch to **Sensor failure** and try each type. Note that the open-circuit and short faults are caught
   almost at once by the range check, the stuck output takes about 2 seconds (flatline check), and the
   gauge shows "SIGNAL INVALID" instead of a misleading number.
5. In the sidebar, set **Alarm delay** to 0 s in Normal mode. Nuisance low-pressure alarms appear during
   each pressure ramp-up. This is why alarms need a delay.
6. Open **Sensor and calibration**, untick the calibration correction, and watch the zero reading shift
   by the offset error. Then open the **Sensor calibration** tab to see the as-found and as-left errors.
7. Open **Test results** to show the scenario table.

## Architecture

```
brake-pressure-monitor/
├── app.py             Streamlit dashboard: presentation only
├── brake_sim.py       Simulation engine: sensor, ADC, calibration, diagnostics, alarms
├── test_brake_sim.py  Unit tests for the engine
├── requirements.txt
├── README.md
└── .gitignore
```

**Why two modules?** Real instrumentation keeps the measurement-and-diagnostics firmware separate from
the display. `brake_sim.py` imports only NumPy and Pandas, so it runs and is tested without a browser.
`app.py` reads the engine's state and draws it; it holds no measurement logic.

### Signal chain (one pass per sample, 10 Hz)

```
Driver pedal --> Hydraulic dynamics --> Pressure sensor --> 10-bit ADC --> Fault injection
                                                                               |
        Alarms <-- Diagnostics <-- Calibration <-- Engineering-unit conversion <--
```

| Stage | What it models |
|---|---|
| Driver pedal | An 8 s cycle: 2 s rest, 0.6 s apply ramp, hold, 0.6 s release ramp. Each cycle draws a new demand of 58-72 bar. |
| Hydraulic dynamics | First-order lag (time constant 0.12 s): pressure follows the demand but cannot change instantly. |
| Pressure sensor | Ratiometric 0.5-4.5 V over 0-160 bar, with gain error, offset error, non-linearity and noise. |
| ADC | 10-bit, 5 V reference (about 4.9 mV, or about 0.2 bar, per count). |
| Fault injection | In Sensor-failure mode the ADC code is forced to 0, forced to full scale, frozen, or randomly dropped. |
| Engineering units | The ECU's nominal transfer function (it assumes a perfect sensor), then the optional two-point calibration. |
| Diagnostics | Range check, rate-of-change check, and flatline check decide whether the reading is trustworthy. |
| Alarms | Low, high and sensor-fault alarms. Pressure alarms are inhibited while the reading is invalid. |

### Data flow in the dashboard

1. The simulator object lives in `st.session_state`, so it survives Streamlit's script re-runs.
2. Each sidebar control is pushed into the engine on every re-run (`set_mode`, `set_thresholds`,
   `set_sensor`, `set_calibration_enabled`). The engine logs genuine changes to the event history.
3. The live panel is a `st.fragment` with `run_every=0.5`. Only this panel re-runs on the timer, advancing the
   simulation by 5 samples (0.5 s of simulated time) and redrawing. Pressing Stop removes the timer.
4. The Calibration and Test tabs call `run_calibration_demo()` and `run_scenario_suite()`, which are pure
   functions in the engine.

### Engine API (`brake_sim.py`)

| Item | Purpose |
|---|---|
| `BrakePressureSimulator` | Steps the chain one sample at a time; holds history, events and alarm state |
| `Thresholds`, `SensorModel`, `DiagnosticsConfig`, `Calibration` | Immutable configuration objects |
| `run_calibration_demo()` | Deadweight-tester style calibration demonstration |
| `run_plan()`, `run_scenario_suite()` | Headless scenario runner and grader |

## Instrumentation concepts

| Concept | Where |
|---|---|
| **Ratiometric output with a live zero** | 0.5 V at 0 bar means a broken wire (0 V) is distinguishable from a real 0 bar reading, like the 4 mA live zero of a 4-20 mA loop. |
| **ADC quantisation** | Resolution limits how finely pressure can be resolved. |
| **First-order lag** | Real hydraulic systems have compliance, so pressure cannot step instantly. |
| **Offset, gain and non-linearity errors** | The sensor model and the Calibration tab. |
| **Two-point calibration** | `gain = (ref_hi - ref_lo) / (read_hi - read_lo)`, `offset = ref_lo - gain * read_lo`. Exact at the two reference points; non-linearity, noise and quantisation remain in between. |
| **Averaging** | Averaging n readings cuts random noise by about the square root of n. |
| **Range check** | A healthy output stays within 0.25-4.75 V. Outside it means open or short circuit. |
| **Rate-of-change plausibility** | A change faster than the hydraulics allow marks the sample as bad. |
| **Flatline check** | A live signal always carries noise; identical ADC codes for 2 s means a frozen sensor. |
| **Debounce (alarm delay)** | A condition must persist before the alarm raises, rejecting the harmless ramp-up. |
| **Hysteresis** | An alarm clears only after the value moves back inside the limit by a margin, so it does not chatter. |
| **Pressure-versus-demand check** | A low reading is only abnormal while the pedal is pressed, because 0 bar at rest is normal. |
| **Alarm inhibit on invalid data** | When the sensor is faulty, pressure alarms are suppressed so the operator sees the real problem. |

### Fault detection logic

| Fault | How it is detected | Typical latency |
|---|---|---|
| Open circuit / short to ground (0 V) | Voltage below 0.25 V on 3 of the last 10 samples | about 0.2 s |
| Short to supply (5 V) | Voltage above 4.75 V on 3 of the last 10 samples | about 0.2 s |
| Stuck / frozen output | Identical ADC codes for 20 consecutive samples | about 2 s |
| Intermittent dropouts | The same 3-of-10 filter, counting dropouts and the jumps they cause | about 1 s |

A detected fault latches and is released only after 2 s of clean data. While a stuck sensor is still
undetected, its frozen reading looks valid; this detection latency is a real limitation of diagnostics.

### Default demonstration values (all illustrative)

| Parameter | Default | Range in the UI |
|---|---|---|
| Low-pressure alarm (pedal applied) | 40 bar | 10-70 |
| High-pressure alarm | 100 bar | 80-140 |
| Hysteresis | 5 bar | 1-15 |
| Alarm delay | 1.0 s | 0-3 |
| Sensor gain error | +3 % | -5 to +5 |
| Sensor offset error | +1.5 bar | -3 to +3 |
| Sensor non-linearity | 0.3 %FS | 0-1 |
| Noise | 0.4 bar rms | fixed |

Mode behaviour: **Low pressure** scales demand by 0.35 (about 20-25 bar); **Excessive pressure** scales it
by 1.85 (about 107-133 bar); **Sensor failure** leaves the true pressure normal and corrupts only the
sensor signal.

## Limitations

- All behaviour is simulated. There is no real sensor data, vehicle data, or hardware.
- The hydraulic model is a single first-order lag, not a model of a real brake circuit.
- Thresholds, tolerances and diagnostic timings are teaching values, not drawn from any standard,
  specification or product, and nothing here has been validated for automotive use.
- The test scenarios check that the simulation behaves as designed. They are not automotive validation.

## Possible extensions

- Add wheel-speed and ABS-style slip signals.
- Replace the pedal cycle with recorded or standard drive-cycle data.
- Add data export of the full history and event log.
- Model temperature drift of the sensor.

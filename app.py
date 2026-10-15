"""
app.py - Streamlit dashboard for the
"Automotive Brake Pressure Monitoring and Fault Detection System".

*** EDUCATIONAL SOFTWARE SIMULATION ***
Every value shown here is synthetic. All thresholds are ILLUSTRATIVE. This is
NOT a validated automotive braking safety system.

Run with:   streamlit run app.py

Architecture
------------
    brake_sim.py  - the simulation engine (sensor, ADC, calibration,
                    diagnostics, alarms). No UI code.
    app.py        - this file: presentation only. It reads the engine's state
                    and draws it; it contains no measurement logic.

How live updating works
-----------------------
Streamlit re-runs the whole script on every widget interaction. A plain re-run
would restart the page, so the live panel is wrapped in `st.fragment` with
`run_every`: only that panel re-runs on a timer, advancing the simulation a few
samples each tick. The simulator object lives in `st.session_state` so it
survives re-runs. Pressing Stop removes the timer; Start adds it back.
"""

import plotly.graph_objects as go
import streamlit as st

from brake_sim import (
    ALARM_HIGH, ALARM_LOW, ALARM_SENSOR, DT_S, FAULT_OPEN, FAULT_TYPES,
    FULL_SCALE_BAR, MODE_SENSOR, MODES,
    BrakePressureSimulator, Thresholds, run_calibration_demo, run_scenario_suite,
)

# ---------------------------------------------------------------------------
# Look and feel: an instrument-panel palette - graphite text on a light panel,
# with colour reserved for meaning (green normal, amber warning, red critical).
# ---------------------------------------------------------------------------
INK = "#1F2933"
STEEL = "#52606D"
TEAL = "#0B7285"
GREEN = "#1B8A5A"
AMBER = "#C77700"
RED = "#C62828"
GREY = "#9AA5B1"

DISCLAIMER = (
    "Educational software simulation. All data is synthetic and every threshold is "
    "illustrative. This is not a validated automotive braking safety system."
)

REFRESH_S = 0.5                                   # wall-clock refresh interval
STEPS_PER_TICK = int(round(REFRESH_S / DT_S))     # 5 samples per tick -> real time

CSS = """
<style>
.notice {border-left: 4px solid #C77700; background: #FBF3E2; color: #1F2933;
         padding: 10px 14px; border-radius: 4px; font-size: 0.92rem; margin-bottom: 14px;}
.status {padding: 12px 18px; border-radius: 4px; margin-bottom: 14px; color: #1F2933;}
.status-main {font-size: 1.5rem; font-weight: 700; letter-spacing: 0.01em;}
.status-sub {font-size: 0.88rem; opacity: 0.85;}
.alarm-row {display: flex; align-items: center; gap: 10px; padding: 7px 0;
            border-bottom: 1px solid rgba(128,128,128,0.25); font-size: 0.95rem;}
.alarm-row .dot {width: 12px; height: 12px; border-radius: 50%; display: inline-block;}
.alarm-row .state {margin-left: auto; font-weight: 600;}
</style>
"""


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------
def build_gauge(value_bar, status, valid, low_bar, high_bar):
    """Dial gauge. The coloured bands show the *operating* ranges at a glance:
    below the low limit (amber), acceptable (green), above the high limit (red).
    The low limit only applies while the pedal is pressed - see the caption."""
    bar_color = {"NORMAL": GREEN, "LOW PRESSURE": AMBER,
                 "HIGH PRESSURE": RED, "SENSOR FAULT": GREY}.get(status, GREEN)
    if not valid:
        bar_color = GREY
    fig = go.Figure(go.Indicator(
        mode="gauge+number" if valid else "gauge",
        value=value_bar if valid else 0,
        number={"suffix": " bar", "valueformat": ".1f"},
        title={"text": "Brake-line pressure"},
        gauge={
            "axis": {"range": [0, FULL_SCALE_BAR], "tickwidth": 1},
            "bar": {"color": bar_color, "thickness": 0.3},
            "steps": [
                {"range": [0, low_bar], "color": "#FBEBD0"},
                {"range": [low_bar, high_bar], "color": "#DDF0E6"},
                {"range": [high_bar, FULL_SCALE_BAR], "color": "#F6D9D7"},
            ],
            "threshold": {"line": {"color": RED, "width": 4}, "thickness": 0.8, "value": high_bar},
        },
    ))
    if not valid:
        fig.add_annotation(text="SIGNAL INVALID", x=0.5, y=0.2, showarrow=False,
                           font={"size": 22, "color": RED})
    fig.update_layout(height=290, margin={"l": 25, "r": 25, "t": 60, "b": 10})
    return fig


def _segments(df, column):
    """Contiguous (start, end) time spans where a boolean column is True."""
    if df.empty:
        return []
    flags = df[column].to_numpy(dtype=bool)
    times = df["time_s"].to_numpy()
    spans, start = [], None
    for t, f in zip(times, flags):
        if f and start is None:
            start = t
        elif not f and start is not None:
            spans.append((start, t))
            start = None
    if start is not None:
        spans.append((start, times[-1]))
    return spans


def build_trend(df, low_bar, high_bar, window_s=None, show_true=True, height=360):
    """Pressure-versus-time trend with threshold lines and shaded alarm periods."""
    fig = go.Figure()
    if not df.empty:
        if show_true:
            # The *true* pressure exists only because this is a simulation. A real
            # instrument never sees it - which is why sensor faults are dangerous.
            fig.add_trace(go.Scatter(
                x=df["time_s"].to_numpy(), y=df["true_bar"].to_numpy(),
                mode="lines", name="True pressure (simulation reference)",
                line={"color": GREY, "width": 1.5, "dash": "dot"}))
        fig.add_trace(go.Scatter(
            x=df["time_s"].to_numpy(), y=df["measured_bar"].to_numpy(),
            mode="lines", name="Measured pressure (ADC + calibration)",
            line={"color": TEAL, "width": 2.5}, connectgaps=False))   # gaps = invalid samples
        for col, colour in (("sensor_fault", RED), ("high_alarm", RED), ("low_alarm", AMBER)):
            for x0, x1 in _segments(df, col):
                fig.add_vrect(x0=x0, x1=x1, fillcolor=colour, opacity=0.10, line_width=0)
    fig.add_hline(y=high_bar, line_dash="dash", line_color=RED,
                  annotation_text=f"High alarm {high_bar:.0f} bar", annotation_position="top left")
    fig.add_hline(y=low_bar, line_dash="dash", line_color=AMBER,
                  annotation_text=f"Low alarm {low_bar:.0f} bar (pedal applied)",
                  annotation_position="bottom left")
    if window_s is not None:
        t_end = max(float(df["time_s"].iloc[-1]) if not df.empty else 0.0, window_s)
        fig.update_xaxes(range=[t_end - window_s, t_end])
    fig.update_xaxes(title="Simulation time (s)")
    fig.update_yaxes(title="Pressure (bar)", range=[-10, FULL_SCALE_BAR + 5])
    fig.update_layout(height=height, margin={"l": 50, "r": 20, "t": 30, "b": 40},
                      legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
                      hovermode="x unified")
    return fig


# ---------------------------------------------------------------------------
# Small HTML helpers
# ---------------------------------------------------------------------------
def status_banner(status, started, running, t_s, mode):
    if not started:
        colour, bg, main = STEEL, "#E8ECEF", "READY"
        sub = "Press Start in the sidebar to begin the simulation."
    else:
        colour, bg = {"NORMAL": (GREEN, "#E3F3EA"), "LOW PRESSURE": (AMBER, "#FBEBD0"),
                      "HIGH PRESSURE": (RED, "#F9DCDA"), "SENSOR FAULT": (RED, "#F9DCDA")}[status]
        main = status
        sub = f"Simulation time {t_s:.1f} s. Mode: {mode}. {'Running' if running else 'Paused'}."
    return (f'<div class="status" style="border-left: 8px solid {colour}; background: {bg};">'
            f'<div class="status-main">{main}</div><div class="status-sub">{sub}</div></div>')


def alarm_row(label, active, colour):
    dot = colour if active else "#B7C0C8"
    state = "ACTIVE" if active else "clear"
    return (f'<div class="alarm-row"><span class="dot" style="background:{dot}"></span>'
            f'<span>{label}</span><span class="state" style="color:{colour if active else STEEL}">'
            f'{state}</span></div>')


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
def render_calibration_tab(sim):
    st.subheader("Sensor calibration demonstration")
    st.markdown(
        "A real pressure transducer is never perfect. Its **zero** can be off (offset error), its "
        "**span** can be off (gain error), and its curve can bow slightly (non-linearity). "
        "Calibration applies *known* reference pressures, compares what the instrument reads with "
        "what it should read, and computes a correction. This demonstration uses a simulated "
        "deadweight tester at 9 reference points, the sensor error settings from the sidebar, and a "
        "**two-point (zero and span) calibration** computed from the first and last point."
    )
    n_avg = st.slider("Readings averaged per reference point", 1, 50, 20,
                      help="Averaging n readings reduces random noise by roughly the square root of n.")
    s = sim.sensor
    df, cal = run_calibration_demo(s.gain_error_pct, s.offset_bar, s.nonlinearity_pct_fs,
                                   s.noise_bar_rms, n_avg=n_avg)
    found = float(df["as_found_error_pct_fs"].abs().max())
    left = float(df["as_left_error_pct_fs"].abs().max())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Correction gain", f"{cal.gain:.4f}")
    m2.metric("Correction offset", f"{cal.offset_bar:+.2f} bar")
    m3.metric("Worst error as-found", f"{found:.2f} %FS")
    m4.metric("Worst error as-left", f"{left:.2f} %FS", delta=f"{left - found:+.2f} %FS",
              delta_color="inverse")

    c1, c2 = st.columns(2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["reference_bar"], y=df["reference_bar"], mode="lines",
                             name="Ideal (reading = reference)", line={"color": GREY, "dash": "dash"}))
    fig.add_trace(go.Scatter(x=df["reference_bar"], y=df["as_found_bar"], mode="lines+markers",
                             name="As-found (before calibration)", line={"color": RED}))
    fig.add_trace(go.Scatter(x=df["reference_bar"], y=df["as_left_bar"], mode="lines+markers",
                             name="As-left (after calibration)", line={"color": GREEN}))
    fig.update_xaxes(title="Reference pressure (bar)")
    fig.update_yaxes(title="Indicated pressure (bar)")
    fig.update_layout(height=340, margin={"l": 50, "r": 10, "t": 30, "b": 40},
                      legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0})
    c1.plotly_chart(fig, key="cal_transfer")

    fig2 = go.Figure()
    fig2.add_hrect(y0=-0.5, y1=0.5, fillcolor=GREEN, opacity=0.10, line_width=0)
    fig2.add_trace(go.Scatter(x=df["reference_bar"], y=df["as_found_error_pct_fs"], mode="lines+markers",
                              name="As-found error", line={"color": RED}))
    fig2.add_trace(go.Scatter(x=df["reference_bar"], y=df["as_left_error_pct_fs"], mode="lines+markers",
                              name="As-left error", line={"color": GREEN}))
    fig2.update_xaxes(title="Reference pressure (bar)")
    fig2.update_yaxes(title="Error (% of full scale)")
    fig2.update_layout(height=340, margin={"l": 50, "r": 10, "t": 30, "b": 40},
                       legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0})
    c2.plotly_chart(fig2, key="cal_error")

    verdict = "within" if left <= 0.5 else "outside"
    st.caption(
        f"Shaded band: +/-0.5 %FS (an illustrative target, not an industry specification). The as-left "
        f"worst-case error is {verdict} this band. The error is zero at both calibration points by "
        f"construction; what remains in between is non-linearity, noise and ADC quantisation, which a "
        f"two-point calibration cannot remove."
    )
    show = df.copy()
    show.columns = ["Reference (bar)", "As-found (bar)", "As-found error (bar)", "As-found error (%FS)",
                    "As-left (bar)", "As-left error (bar)", "As-left error (%FS)"]
    st.dataframe(show.round(3), hide_index=True)


@st.cache_data(show_spinner=False)
def cached_suite():
    return run_scenario_suite()


def render_test_tab():
    st.subheader("Scenario test results")
    st.markdown(
        "Each scenario below is run by the simulation engine with no user interface, using a fixed "
        "random seed and the default demonstration thresholds, and is then graded against the "
        "response listed. These results are produced when this page loads, so they always reflect the "
        "current code. They check that the **simulation logic behaves as designed**; they are not "
        "automotive validation."
    )
    suite, traces = cached_suite()
    n_pass = int((suite["Result"] == "PASS").sum())
    c1, c2 = st.columns(2)
    c1.metric("Scenarios passed", f"{n_pass} of {len(suite)}")
    c2.metric("Random seed", "42 (fixed)")
    st.dataframe(suite, hide_index=True, height=440)
    st.download_button("Download results (CSV)", suite.to_csv(index=False),
                       file_name="scenario_results.csv", mime="text/csv")

    st.markdown("**Inspect a scenario**")
    name = st.selectbox("Scenario trace", list(traces.keys()))
    hist, events = traces[name]
    th = Thresholds(high_bar=50.0) if name.startswith("9.") else Thresholds()
    fig = build_trend(hist, th.low_bar, th.high_bar, window_s=None, show_true=True)
    for _, ev in events[events["action"] == "RAISED"].iterrows():
        fig.add_vline(x=ev["time_s"], line_dash="dot", line_color=RED)
    for _, ev in events[events["action"] == "CLEARED"].iterrows():
        fig.add_vline(x=ev["time_s"], line_dash="dot", line_color=GREEN)
    st.plotly_chart(fig, key="scenario_trace")
    st.caption("Red dotted lines: alarm raised. Green dotted lines: alarm cleared. "
               "Shaded areas: alarm active. Gaps in the teal line: samples flagged invalid.")
    ev_show = events[["time_s", "severity", "event", "detail"]].rename(
        columns={"time_s": "Time (s)", "severity": "Severity", "event": "Event", "detail": "Detail"})
    st.dataframe(ev_show, hide_index=True, height=240)


def render_about_tab():
    st.subheader("Instrumentation concepts demonstrated")
    st.markdown(
        """
| Concept | Where it appears |
|---|---|
| **Ratiometric transducer, 0.5-4.5 V output** | The 0.5 V live zero separates a true 0 bar reading from a broken wire (0 V), like the 4 mA live zero of a 4-20 mA loop. |
| **ADC quantisation (10-bit)** | 5 V / 1023 is about 4.9 mV per count, or about 0.2 bar per count over this span. |
| **First-order hydraulic lag** | Pressure follows the pedal demand with a time constant; it cannot change instantly. |
| **Offset, gain and non-linearity errors** | Sidebar *Sensor and calibration* controls; Calibration tab. |
| **Two-point calibration** | Zero and span correction computed from two reference points. |
| **Diagnostic range check** | A healthy output stays within 0.25-4.75 V; outside means open or short circuit. |
| **Rate-of-change plausibility** | Pressure faster than the hydraulics allow is treated as a bad sample. |
| **Flatline check** | A real signal always carries noise; identical ADC codes for 2 s means a frozen sensor. |
| **Alarm delay (debounce)** | Rejects the harmless pressure ramp-up after the pedal is pressed. |
| **Hysteresis** | Stops alarms chattering when the reading hovers at a limit. |
| **Alarm inhibit on invalid data** | Pressure alarms are suppressed when the sensor is faulty, so the operator sees the real problem. |
        """
    )
    st.subheader("Signal chain")
    st.code(
        "Driver pedal -> Hydraulic dynamics -> Pressure sensor -> 10-bit ADC -> Fault injection\n"
        "                                                                              |\n"
        "     Alarms <- Diagnostics (range, rate, flatline) <- Calibration <- Engineering units",
        language="text",
    )
    st.subheader("Project files")
    st.markdown(
        "- `app.py`: this dashboard (presentation only)\n"
        "- `brake_sim.py`: the simulation engine (sensor, ADC, calibration, diagnostics, alarms)\n"
        "- `test_brake_sim.py`: unit tests for the engine\n"
        "- `README.md`: setup, architecture and usage"
    )
    st.warning(DISCLAIMER)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Brake Pressure Monitoring (Simulation)", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    # The simulator persists across Streamlit re-runs in session_state.
    if "sim" not in st.session_state:
        st.session_state.sim = BrakePressureSimulator(seed=42)
        st.session_state.running = False
    sim = st.session_state.sim

    # ---------------- Sidebar: controls ----------------
    sb = st.sidebar
    sb.header("Simulation controls")
    b1, b2, b3 = sb.columns(3)
    if b1.button("Start", type="primary"):
        if not st.session_state.running:
            sim.log("INFO", "Simulation started")
        st.session_state.running = True
    if b2.button("Stop"):
        if st.session_state.running:
            sim.log("INFO", "Simulation stopped")
        st.session_state.running = False
    if b3.button("Reset"):
        sim.reset()
        st.session_state.running = False
        sim.log("INFO", "Simulation reset")

    mode = sb.radio("Simulation mode", MODES,
                    help="Switch modes while running to see the system respond.")
    fault_type = FAULT_OPEN
    if mode == MODE_SENSOR:
        fault_type = sb.selectbox("Sensor failure type", FAULT_TYPES)

    sb.subheader("Alarm thresholds (illustrative)")
    low = sb.slider("Low-pressure alarm (bar)", 10.0, 70.0, 40.0, 1.0,
                    help="Applies only while the pedal is pressed.")
    high = sb.slider("High-pressure alarm (bar)", 80.0, 140.0, 100.0, 1.0)
    hyst = sb.slider("Hysteresis (bar)", 1.0, 15.0, 5.0, 1.0)
    delay = sb.slider("Alarm delay (s)", 0.0, 3.0, 1.0, 0.1,
                      help="Try 0 s to see nuisance alarms during the pressure ramp-up.")

    with sb.expander("Sensor and calibration"):
        gain = st.slider("Gain error (%)", -5.0, 5.0, 3.0, 0.5)
        offset = st.slider("Offset error (bar)", -3.0, 3.0, 1.5, 0.5)
        nonlin = st.slider("Non-linearity (% FS)", 0.0, 1.0, 0.3, 0.1)
        cal_on = st.checkbox("Apply two-point calibration correction", value=True)

    with sb.expander("Display"):
        window = st.slider("Trend window (s)", 20, 120, 60, 10)
        show_true = st.checkbox("Show true pressure (simulation reference)", value=True)

    sb.caption(DISCLAIMER)

    # Push the sidebar settings into the engine (it logs genuine changes only).
    sim.set_mode(mode, fault_type)
    sim.set_thresholds(Thresholds(low_bar=low, high_bar=high, hysteresis_bar=hyst, alarm_delay_s=delay))
    sim.set_sensor(gain, offset, nonlin)
    sim.set_calibration_enabled(cal_on)

    # ---------------- Header ----------------
    st.title("Automotive Brake Pressure Monitoring and Fault Detection System")
    st.markdown(f'<div class="notice">{DISCLAIMER}</div>', unsafe_allow_html=True)

    tab_live, tab_cal, tab_test, tab_about = st.tabs(
        ["Live monitor", "Sensor calibration", "Test results", "Concepts"])

    # ---------------- Live monitor (self-refreshing fragment) ----------------
    @st.fragment(run_every=REFRESH_S if st.session_state.running else None)
    def live_panel():
        if st.session_state.running:
            sim.advance(STEPS_PER_TICK)
        rec = sim.last_record
        started = rec is not None
        status = sim.status
        valid = bool(rec["valid"]) if started else True
        # Readings just below zero at rest are noise; clamp only for display.
        shown = max(0.0, rec["measured_bar"]) if (started and valid) else 0.0

        st.markdown(status_banner(status, started, st.session_state.running, sim.t, sim.mode),
                    unsafe_allow_html=True)

        left, right = st.columns([1.15, 1])
        left.plotly_chart(build_gauge(shown, status, valid, low, high), key="gauge")
        left.caption("Bands: amber below the low limit, green acceptable, red above the high limit. "
                     "The low limit applies only while the pedal is pressed.")

        r1, r2 = right.columns(2)
        r1.metric("Brake-line pressure", f"{shown:.1f} bar" if valid else "invalid")
        r2.metric("Pedal demand", ("Applied" if rec["brake_on"] else "Released") if started else "-")
        r3, r4 = right.columns(2)
        r3.metric("Sensor output", f"{rec['voltage_v']:.2f} V" if started else "-",
                  help="Raw transducer voltage at the ADC input (0.5 V = 0 bar, 4.5 V = full scale).")
        r4.metric("Sensor health", "FAULT" if (started and rec["sensor_fault"]) else "OK")
        if started:
            right.caption(f"ADC code {rec['adc_code']} of 1023. "
                          + (f"Diagnosis: {rec['fault_reason']}" if rec["sensor_fault"] else
                             ("Calibration correction applied." if sim.calibration_enabled
                              else "Calibration correction OFF: readings include sensor errors.")))

        st.markdown("**Pressure trend**")
        hist = sim.history_df(window_s=window)
        st.plotly_chart(build_trend(hist, low, high, window_s=window, show_true=show_true),
                        key="trend")

        a_col, e_col = st.columns([1, 2])
        with a_col:
            st.markdown("**Alarm status**")
            active = sim.active_alarms
            st.markdown(
                alarm_row("Low pressure (with pedal applied)", ALARM_LOW in active, AMBER)
                + alarm_row("High pressure", ALARM_HIGH in active, RED)
                + alarm_row("Sensor fault", ALARM_SENSOR in active, RED),
                unsafe_allow_html=True)
        with e_col:
            st.markdown("**Event history** (newest first)")
            ev = sim.events_df()
            if ev.empty:
                st.info("No events yet. Press Start, then try the other simulation modes.")
            else:
                shown_ev = ev[["time_s", "severity", "event", "detail"]].iloc[::-1].rename(
                    columns={"time_s": "Time (s)", "severity": "Severity",
                             "event": "Event", "detail": "Detail"})
                st.dataframe(shown_ev, hide_index=True, height=230)

    with tab_live:
        live_panel()
    with tab_cal:
        render_calibration_tab(sim)
    with tab_test:
        render_test_tab()
    with tab_about:
        render_about_tab()


main()

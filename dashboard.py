import os
import json
import time
from collections import deque
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import psutil
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

import torch
import torch.nn as nn


# =========================
# Page
# =========================
#st.set_page_config(page_title="Real-Time Monitoring + AI", layout="wide")
st.set_page_config(page_title="AI-Driven Real-Time Resource Monitoring and Usage Prediction", layout="wide")

# ---- Dark Theme + Glow CSS ----
st.markdown(
    """
<style>
/* overall background */
.stApp {
  background: radial-gradient(1200px 800px at 10% 10%, rgba(60, 120, 255, 0.12), transparent 60%),
              radial-gradient(900px 700px at 90% 20%, rgba(180, 80, 255, 0.12), transparent 60%),
              radial-gradient(900px 700px at 50% 90%, rgba(0, 255, 160, 0.08), transparent 60%),
              #0b1220;
  color: #e8eefc;
}

h1, h2, h3, h4, h5, h6, .stMarkdown { color: #e8eefc; }

/* expander style */
div[data-testid="stExpander"] {
  border-radius: 14px;
  border: 1px solid rgba(255,255,255,0.08);
  background: rgba(255,255,255,0.03);
}

/* card base */
.kpi-card {
  border-radius: 16px;
  padding: 16px 16px 14px 16px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.04);
  box-shadow: 0 10px 30px rgba(0,0,0,0.35);
  position: relative;
  overflow: hidden;
  min-height: 110px;
}

.kpi-glow {
  position: absolute;
  inset: -40px;
  background: radial-gradient(circle at 20% 20%, var(--glow), transparent 45%);
  filter: blur(10px);
  opacity: 0.9;
  pointer-events: none;
}

.kpi-title { font-size: 12px; letter-spacing: 0.10em; opacity: 0.85; }
.kpi-value { font-size: 30px; font-weight: 800; margin-top: 6px; }
.kpi-sub   { font-size: 12px; opacity: 0.85; margin-top: 6px; }

.kpi-badge {
  display:inline-block;
  font-size: 11px;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.05);
  margin-left: 8px;
  opacity: 0.95;
}

.alert-card {
  border-radius: 14px;
  padding: 14px 14px 12px 14px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.04);
  box-shadow: 0 10px 30px rgba(0,0,0,0.30);
  min-height: 70px;
}

.small-note { font-size: 12px; opacity: 0.80; }
hr { border-color: rgba(255,255,255,0.10); }
</style>
""",
    unsafe_allow_html=True,
)

#st.markdown("## Real-Time System Monitoring with AI Predictions")
st.markdown("## AI-Driven Real-Time Resource Monitoring and Usage Prediction")

# =========================
# Models
# =========================
@st.cache_resource
def load_rf():
    model = joblib.load(r"models\resource_model.pkl")
    scaler = joblib.load(r"models\scaler.pkl")
    return model, scaler

rf_model, rf_scaler = load_rf()


class LSTMRegressor(nn.Module):
    def __init__(self, input_size=3, hidden=64, layers=2, dropout=0.1, output_size=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)


@st.cache_resource
def load_lstm():
    cfg_path = r"models\lstm_config.json"
    model_path = r"models\lstm_model.pt"
    scaler_path = r"models\lstm_scaler.pkl"

    if not (os.path.exists(cfg_path) and os.path.exists(model_path) and os.path.exists(scaler_path)):
        return None, None, None

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    scaler = joblib.load(scaler_path)
    model = LSTMRegressor(
        input_size=3,
        hidden=int(cfg.get("hidden", 64)),
        layers=int(cfg.get("layers", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        output_size=3,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    seq_len = int(cfg.get("seq_len", 20))
    return model, scaler, seq_len


lstm_model, lstm_scaler, lstm_seq_len = load_lstm()
has_lstm = lstm_model is not None


# =========================
# Sidebar controls (INNOVATIVE UI)
# =========================
st.sidebar.markdown("## ⚙️ Control Center")
st.sidebar.caption("Tune monitoring, warnings, and prediction behavior.")

# Presets
st.sidebar.markdown("### 🧩 Presets")
preset = st.sidebar.radio(
    "Choose a mode",
    ["Balanced", "Aggressive (Fast)", "Power Saving"],
    horizontal=True
)

default_window = 60
default_refresh = 2
default_cpu = 80
default_mem = 85
default_disk = 90
default_batt = 20
default_anom = 10.0
default_alpha = 0.85

if preset == "Aggressive (Fast)":
    default_refresh = 1
    default_anom = 8.0
    default_alpha = 0.75
elif preset == "Power Saving":
    default_refresh = 4
    default_window = 90
    default_anom = 12.0
    default_alpha = 0.90

st.sidebar.markdown("### 🧠 Prediction")
engines = ["RandomForest"]
if has_lstm:
    engines.append("LSTM (PyTorch)")
engine = st.sidebar.selectbox("Prediction Engine", engines, index=(1 if has_lstm else 0))
st.sidebar.caption("LSTM learns sequences; RF predicts from recent lags.")

st.sidebar.divider()

st.sidebar.markdown("### ⏱️ Monitoring")
window_size = st.sidebar.slider("History window (seconds)", 20, 300, default_window, 10)
refresh_sec = st.sidebar.slider("Refresh rate (seconds)", 1, 10, default_refresh, 1)
st.sidebar.caption("Tip: 2s feels smooth. 1s is fastest but heavier.")

st.sidebar.divider()

st.sidebar.markdown("### 🚨 Warning thresholds")
c1, c2 = st.sidebar.columns(2)
with c1:
    cpu_warn = st.slider("CPU %", 50, 100, default_cpu, 1)
    mem_warn = st.slider("Memory %", 50, 100, default_mem, 1)
with c2:
    disk_warn = st.slider("Disk Used %", 50, 100, default_disk, 1)
    battery_warn = st.slider("Battery Low %", 5, 50, default_batt, 1)

st.sidebar.caption("Alerts turn ⚠️ when values cross these limits.")

st.sidebar.divider()

with st.sidebar.expander("🧪 Advanced (AI & Anomaly)", expanded=False):
    anomaly_thr = st.slider("Anomaly threshold (abs error in % units)", 1.0, 50.0, default_anom, 1.0)
    alpha = st.slider("Prediction responsiveness", 0.50, 0.95, default_alpha, 0.05)
    st.caption("Lower = faster change. Higher = smoother predictions.")


# ✅ smooth refresh (no while-loop, no st.rerun)
st_autorefresh(interval=refresh_sec * 1000, key="autorefresh_main")


# =========================
# State
# =========================
RF_LAGS = 5

if "history" not in st.session_state:
    st.session_state.history = deque(maxlen=3000)

if "rf_buf" not in st.session_state:
    st.session_state.rf_buf = deque(maxlen=RF_LAGS)

if "lstm_buf" not in st.session_state:
    st.session_state.lstm_buf = deque(maxlen=int(lstm_seq_len or 20))

if "last_io" not in st.session_state:
    st.session_state.last_io = psutil.disk_io_counters()
    st.session_state.last_io_t = time.time()

if "cpu_warm" not in st.session_state:
    psutil.cpu_percent(interval=1)
    st.session_state.cpu_warm = True


# =========================
# Helpers
# =========================
def disk_path():
    return os.getenv("SystemDrive", "C:") + "\\"


def read_metrics():
    cpu = psutil.cpu_percent(interval=0.2)
    mem = psutil.virtual_memory().percent
    dused = psutil.disk_usage(disk_path()).percent
    return float(cpu), float(mem), float(dused)


def read_disk_mbps():
    now = time.time()
    io = psutil.disk_io_counters()
    dt = max(1e-6, now - st.session_state.last_io_t)

    read_mb_s = (io.read_bytes - st.session_state.last_io.read_bytes) / dt / (1024 * 1024)
    write_mb_s = (io.write_bytes - st.session_state.last_io.write_bytes) / dt / (1024 * 1024)

    st.session_state.last_io = io
    st.session_state.last_io_t = now
    return float(read_mb_s), float(write_mb_s)


def read_battery():
    try:
        b = psutil.sensors_battery()
        if b is None:
            return None
        percent = float(b.percent)
        plugged = bool(b.power_plugged)
        secs = int(b.secsleft) if b.secsleft is not None else -1
        return percent, plugged, secs
    except Exception:
        return None


def fmt_time_left(secs: int) -> str:
    if secs is None or secs < 0 or secs == psutil.POWER_TIME_UNLIMITED or secs == psutil.POWER_TIME_UNKNOWN:
        return "Time left: N/A"
    h = secs // 3600
    m = (secs % 3600) // 60
    if h <= 0:
        return f"Time left: ~{m}m"
    return f"Time left: ~{h}h {m}m"


def trend_arrow(series: pd.Series, k: int = 6) -> str:
    if series is None or len(series) < k + 1:
        return "▬"
    a = float(series.iloc[-k])
    b = float(series.iloc[-1])
    if b - a > 0.8:
        return "▲"
    if a - b > 0.8:
        return "▼"
    return "▬"


def trim_history(hist, seconds):
    cutoff = time.time() - seconds
    while hist and hist[0]["t"] < cutoff:
        hist.popleft()


def kpi_card(title, value, sub, glow_rgba):
    return f"""
<div class="kpi-card" style="--glow:{glow_rgba};">
  <div class="kpi-glow"></div>
  <div class="kpi-title">{title}</div>
  <div class="kpi-value">{value}</div>
  <div class="kpi-sub">{sub}</div>
</div>
"""


def alert_card(title, msg, status_color_rgba):
    return f"""
<div class="alert-card" style="--glow:{status_color_rgba};">
  <div class="kpi-glow" style="opacity:0.6;"></div>
  <div style="font-weight:800; font-size:13px; letter-spacing:0.06em; opacity:0.9;">{title}</div>
  <div style="margin-top:6px; font-size:13px; opacity:0.9;">{msg}</div>
</div>
"""


# =========================
# Read now
# =========================
cpu_now, mem_now, disk_now = read_metrics()
disk_read, disk_write = read_disk_mbps()
bat = read_battery()

st.session_state.rf_buf.append((cpu_now, mem_now, disk_now))
st.session_state.lstm_buf.append((cpu_now, mem_now, disk_now))

# =========================
# Predict next
# =========================
cpu_pred, mem_pred, disk_pred = cpu_now, mem_now, disk_now
predict_ready = False

if engine == "RandomForest":
    predict_ready = len(st.session_state.rf_buf) >= 1

    if predict_ready:
        # Use latest values (NO lag features)
        c, m, d = st.session_state.rf_buf[-1]

        X_simple = pd.DataFrame([[c, m, d]], columns=[
            'cpu_usage', 'memory_usage', 'disk_usage'
        ])

        # Use already loaded model (DON'T reload again)
        X_scaled = rf_scaler.transform(X_simple)
        pred = rf_model.predict(X_scaled)[0]

        # Since model gives single output, reuse it
        cpu_pred = float(pred)
        mem_pred = float(pred)
        disk_pred = float(pred)

else:
    need = int(lstm_seq_len)
    predict_ready = len(st.session_state.lstm_buf) == need

    if predict_ready:
        seq = np.array(list(st.session_state.lstm_buf), dtype=np.float32)
        seq_scaled = lstm_scaler.transform(seq)
        xb = torch.tensor(seq_scaled[None, :, :], dtype=torch.float32)

        with torch.no_grad():
            pred_scaled = lstm_model(xb).numpy()[0]

        pred = lstm_scaler.inverse_transform(pred_scaled.reshape(1, -1))[0]

        cpu_pred = float(pred[0])
        mem_pred = float(pred[1])
        disk_pred = float(pred[2])

# clamp + smooth
cpu_pred = max(0.0, min(100.0, cpu_pred))
mem_pred = max(0.0, min(100.0, mem_pred))
disk_pred = max(0.0, min(100.0, disk_pred))

cpu_pred = alpha * cpu_pred + (1 - alpha) * cpu_now
mem_pred = alpha * mem_pred + (1 - alpha) * mem_now
disk_pred = alpha * disk_pred + (1 - alpha) * disk_now

# =========================
# Save history
# =========================
st.session_state.history.append(
    {
        "t": time.time(),
        "time": datetime.now().strftime("%H:%M:%S"),
        "CPU_now": cpu_now,
        "Memory_now": mem_now,
        "Disk_now": disk_now,
        "CPU_pred": cpu_pred,
        "Memory_pred": mem_pred,
        "Disk_pred": disk_pred,
    }
)

trim_history(st.session_state.history, window_size)
df = pd.DataFrame(list(st.session_state.history))

cpu_arrow = trend_arrow(df["CPU_now"])
mem_arrow = trend_arrow(df["Memory_now"])
disk_arrow = trend_arrow(df["Disk_now"])

# =========================
# Health score + insights
# =========================
risk = 0.45 * (cpu_now / 100.0) + 0.40 * (mem_now / 100.0) + 0.15 * (disk_now / 100.0)
health = int(max(0, min(100, round(100 - risk * 100))))

err_cpu = abs(cpu_now - cpu_pred)
err_mem = abs(mem_now - mem_pred)
err_disk = abs(disk_now - disk_pred)
anom = predict_ready and ((err_cpu > anomaly_thr) or (err_mem > anomaly_thr) or (err_disk > anomaly_thr))

ins_parts = []
if cpu_pred - cpu_now > 2.0:
    ins_parts.append("CPU is predicted to rise.")
elif cpu_now - cpu_pred > 2.0:
    ins_parts.append("CPU is predicted to fall.")
else:
    ins_parts.append("CPU looks stable.")

if mem_pred - mem_now > 2.0:
    ins_parts.append("Memory is trending upward.")
elif mem_now - mem_pred > 2.0:
    ins_parts.append("Memory is trending downward.")
else:
    ins_parts.append("Memory looks stable.")

suggest = "Suggestion: Close heavy background apps / browser tabs." if (cpu_now > cpu_warn or mem_now > mem_warn) else "Suggestion: System operating normally."

bat_line = "Battery: N/A (not detected)"
battery_ok = True
if bat is not None:
    b_pct, plugged, secs = bat
    b_state = "⚡ Charging" if plugged else "🔋 On Battery"
    bat_line = f"Battery: {b_pct:.0f}%  {b_state}  |  {fmt_time_left(secs)}"
    battery_ok = (b_pct >= battery_warn) or plugged

# =========================
# Collapsible Summary
# =========================
with st.expander("System Summary (click to open / close)", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"**Engine:** {engine}")
        st.markdown(f"**Refresh:** {refresh_sec}s")
        st.markdown(f"**Window:** {window_size}s")
    with c2:
        st.markdown(f"**System Health:** {health}/100")
        st.markdown(f"**Anomaly:** {'YES' if anom else 'NO'}")
        st.markdown(f"**Last Update:** {datetime.now().strftime('%H:%M:%S')}")
    with c3:
        st.markdown(f"**Disk Activity:** Read {disk_read:.2f} MB/s | Write {disk_write:.2f} MB/s")
        st.markdown(f"**Prediction Errors:** CPU {err_cpu:.2f} | MEM {err_mem:.2f} | DISK {err_disk:.2f}")
        st.markdown(f"**{bat_line}**")

st.markdown("---")

# =========================
# Current cards
# =========================
st.markdown("### Current Metrics (Live)")
colA, colB, colC, colD = st.columns(4)

with colA:
    st.markdown(
        kpi_card(
            "CPU NOW",
            f"{cpu_now:.1f}%",
            f"{cpu_arrow} mini trend",
            "rgba(0,255,160,0.55)" if cpu_now < cpu_warn else "rgba(255,90,90,0.60)",
        ),
        unsafe_allow_html=True,
    )

with colB:
    st.markdown(
        kpi_card(
            "MEMORY NOW",
            f"{mem_now:.1f}%",
            f"{mem_arrow} mini trend",
            "rgba(80,200,255,0.55)" if mem_now < mem_warn else "rgba(255,170,60,0.60)",
        ),
        unsafe_allow_html=True,
    )

with colC:
    st.markdown(
        kpi_card(
            "DISK USED NOW",
            f"{disk_now:.1f}%",
            f"{disk_arrow} mini trend",
            "rgba(255,210,80,0.55)" if disk_now < disk_warn else "rgba(255,90,90,0.60)",
        ),
        unsafe_allow_html=True,
    )

with colD:
    if bat is None:
        b_val = "N/A"
        b_sub = "Battery not detected"
        glow = "rgba(170,120,255,0.50)"
    else:
        b_pct, plugged, secs = bat
        b_val = f"{b_pct:.0f}%"
        b_sub = ("⚡ Charging" if plugged else "🔋 On Battery") + f" • {fmt_time_left(secs)}"
        glow = "rgba(170,120,255,0.55)" if battery_ok else "rgba(255,90,90,0.60)"

    st.markdown(
        kpi_card("BATTERY", b_val, b_sub, glow),
        unsafe_allow_html=True,
    )

st.markdown("---")

# =========================
# Prediction cards
# =========================
st.markdown("### AI Predictions (Next Moment)")
p1, p2, p3 = st.columns(3)

def pred_glow(delta):
    if abs(delta) < 1.5:
        return "rgba(80,140,255,0.55)"
    if delta > 0:
        return "rgba(130,200,255,0.60)"
    return "rgba(120,120,255,0.55)"

with p1:
    st.markdown(
        kpi_card(
            "CPU NEXT",
            f"{cpu_pred:.1f}%",
            f"Δ {cpu_pred - cpu_now:+.1f}% • {'ready' if predict_ready else 'warming'}",
            pred_glow(cpu_pred - cpu_now),
        ),
        unsafe_allow_html=True,
    )
with p2:
    st.markdown(
        kpi_card(
            "MEMORY NEXT",
            f"{mem_pred:.1f}%",
            f"Δ {mem_pred - mem_now:+.1f}% • {'ready' if predict_ready else 'warming'}",
            pred_glow(mem_pred - mem_now),
        ),
        unsafe_allow_html=True,
    )
with p3:
    st.markdown(
        kpi_card(
            "DISK NEXT",
            f"{disk_pred:.1f}%",
            f"Δ {disk_pred - disk_now:+.1f}% • {'ready' if predict_ready else 'warming'}",
            pred_glow(disk_pred - disk_now),
        ),
        unsafe_allow_html=True,
    )

if not predict_ready:
    need = RF_LAGS if engine == "RandomForest" else int(lstm_seq_len)
    have = len(st.session_state.rf_buf) if engine == "RandomForest" else len(st.session_state.lstm_buf)
    st.info(f"Warming up prediction buffer… {have}/{need} points collected.")

st.markdown("---")

# =========================
# Graph
# =========================
st.markdown("### Live Graph (Current vs Predicted)")
x = list(range(len(df)))
hover_time = df["time"]

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=df["CPU_now"], mode="lines", name="CPU_now",
                         hovertext=hover_time, hovertemplate="Time: %{hovertext}<br>CPU_now: %{y:.2f}%<extra></extra>"))
fig.add_trace(go.Scatter(x=x, y=df["Memory_now"], mode="lines", name="MEM_now",
                         hovertext=hover_time, hovertemplate="Time: %{hovertext}<br>MEM_now: %{y:.2f}%<extra></extra>"))
fig.add_trace(go.Scatter(x=x, y=df["Disk_now"], mode="lines", name="DISK_now",
                         hovertext=hover_time, hovertemplate="Time: %{hovertext}<br>DISK_now: %{y:.2f}%<extra></extra>"))

fig.add_trace(go.Scatter(x=x, y=df["CPU_pred"], mode="lines", name="CPU_pred", line=dict(dash="dash"),
                         hovertext=hover_time, hovertemplate="Time: %{hovertext}<br>CPU_pred: %{y:.2f}%<extra></extra>"))
fig.add_trace(go.Scatter(x=x, y=df["Memory_pred"], mode="lines", name="MEM_pred", line=dict(dash="dash"),
                         hovertext=hover_time, hovertemplate="Time: %{hovertext}<br>MEM_pred: %{y:.2f}%<extra></extra>"))
fig.add_trace(go.Scatter(x=x, y=df["Disk_pred"], mode="lines", name="DISK_pred", line=dict(dash="dash"),
                         hovertext=hover_time, hovertemplate="Time: %{hovertext}<br>DISK_pred: %{y:.2f}%<extra></extra>"))

fig.update_layout(
    height=420,
    margin=dict(l=10, r=10, t=10, b=10),
    legend=dict(orientation="h"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)
fig.update_xaxes(showticklabels=False, ticks="", showgrid=False, title_text=None)
fig.update_yaxes(range=[0, 100], showgrid=True, gridcolor="rgba(255,255,255,0.08)")
st.plotly_chart(fig, use_container_width=True, key="live_graph_main")

st.markdown("---")

# =========================
# Health + Insight
# =========================
left, right = st.columns([1, 1])

with left:
    st.markdown("### System Health Gauge")
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=health,
        number={"suffix": "/100"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"thickness": 0.25},
            "steps": [
                {"range": [0, 40], "color": "rgba(255,90,90,0.20)"},
                {"range": [40, 70], "color": "rgba(255,180,80,0.18)"},
                {"range": [70, 100], "color": "rgba(0,255,160,0.16)"},
            ],
        },
        title={"text": "Health Score"},
    ))
    gauge.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(gauge, use_container_width=True, key="health_gauge")

with right:
    st.markdown("### AI Insight Box 🤖")
    st.markdown(
        f"""
<div class="kpi-card" style="--glow:rgba(80,140,255,0.45); min-height: 280px;">
  <div class="kpi-glow"></div>
  <div style="font-weight:800; font-size:14px; letter-spacing:0.06em; opacity:0.95;">Insights</div>
  <div style="margin-top:10px; font-size:13px; opacity:0.92;">
    • {" ".join(ins_parts)}<br/>
    • {suggest}<br/>
    • {"Anomaly detected: prediction error high." if anom else "No anomaly detected."}<br/>
    • {bat_line}
  </div>
  <div style="margin-top:14px;" class="small-note">
    Notes: Predictions are next-moment forecasts, not copies of current values.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown("---")

# =========================
# Alerts
# =========================
st.markdown("### Alerts")

cpu_ok = cpu_now < cpu_warn
mem_ok = mem_now < mem_warn
disk_ok = disk_now < disk_warn

a1, a2, a3, a4 = st.columns(4)

with a1:
    st.markdown(
        alert_card("CPU", f"{'OK ✅' if cpu_ok else 'HIGH ⚠️'} • {cpu_now:.1f}% (limit {cpu_warn}%)",
                   "rgba(0,255,160,0.45)" if cpu_ok else "rgba(255,90,90,0.55)"),
        unsafe_allow_html=True,
    )
with a2:
    st.markdown(
        alert_card("MEMORY", f"{'OK ✅' if mem_ok else 'HIGH ⚠️'} • {mem_now:.1f}% (limit {mem_warn}%)",
                   "rgba(0,255,160,0.45)" if mem_ok else "rgba(255,170,60,0.60)"),
        unsafe_allow_html=True,
    )
with a3:
    st.markdown(
        alert_card("DISK USED", f"{'OK ✅' if disk_ok else 'HIGH ⚠️'} • {disk_now:.1f}% (limit {disk_warn}%)",
                   "rgba(0,255,160,0.45)" if disk_ok else "rgba(255,90,90,0.55)"),
        unsafe_allow_html=True,
    )
with a4:
    if bat is None:
        msg = "N/A • Battery not detected"
        ok = True
    else:
        b_pct, plugged, _ = bat
        ok = battery_ok
        msg = f"{'OK ✅' if ok else 'LOW ⚠️'} • {b_pct:.0f}% (limit {battery_warn}%)" + (" • Charging" if plugged else "")
    st.markdown(
        alert_card("BATTERY", msg, "rgba(0,255,160,0.45)" if ok else "rgba(255,90,90,0.55)"),
        unsafe_allow_html=True,
    )

st.markdown(
    f"""
<div class="alert-card" style="margin-top:12px; --glow:rgba(255,90,90,0.40);">
  <div class="kpi-glow" style="opacity:0.35;"></div>
  <div style="font-weight:800; font-size:13px; letter-spacing:0.06em; opacity:0.9;">
    ANOMALY STATUS: {"DETECTED ⚠️" if anom else "NONE ✅"}
    <span class="kpi-badge">err CPU {err_cpu:.2f}</span>
    <span class="kpi-badge">err MEM {err_mem:.2f}</span>
    <span class="kpi-badge">err DISK {err_disk:.2f}</span>
    <span class="kpi-badge">Disk R/W {disk_read:.2f}/{disk_write:.2f} MB/s</span>
  </div>
  <div class="small-note" style="margin-top:8px;">
    Last update: {datetime.now().strftime("%H:%M:%S")} • Engine: {engine} • Refresh: {refresh_sec}s • Window: {window_size}s
  </div>
</div>
""",
    unsafe_allow_html=True,
)

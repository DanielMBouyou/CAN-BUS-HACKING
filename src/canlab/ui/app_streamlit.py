"""Dashboard Streamlit — Visualisation temps réel.

Visualisations :
    - Timeline CAN
    - Histogramme fréquence ID
    - Score IDS live
    - CUSUM
    - Vitesse réelle vs affichée
"""

from __future__ import annotations

import json
import os
import time
from collections import deque

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "localhost")
API_PORT = os.getenv("API_PORT", "8000")
API_BASE = f"http://{API_HOST}:{API_PORT}"

st.set_page_config(
    page_title="🛡️ CAN-Stealth-Attack-AI Lab",
    page_icon="🚗",
    layout="wide",
)


# ──────────────────────────────────────────────
# State
# ──────────────────────────────────────────────
if "speed_history" not in st.session_state:
    st.session_state.speed_history = deque(maxlen=200)
if "cusum_history" not in st.session_state:
    st.session_state.cusum_history = deque(maxlen=200)
if "ids_score_history" not in st.session_state:
    st.session_state.ids_score_history = deque(maxlen=200)
if "rpm_history" not in st.session_state:
    st.session_state.rpm_history = deque(maxlen=200)
if "timestamps" not in st.session_state:
    st.session_state.timestamps = deque(maxlen=200)


def api_call(method: str, endpoint: str, **kwargs) -> dict | None:
    """Appel API avec gestion d'erreur."""
    try:
        resp = getattr(requests, method)(f"{API_BASE}{endpoint}", timeout=5, **kwargs)
        return resp.json()
    except Exception as e:
        st.error(f"❌ API Error: {e}")
        return None


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.title("🛡️ CAN-Stealth-Attack-AI Lab")
st.markdown("**Dashboard de cybersécurité automobile en temps réel**")

# ──────────────────────────────────────────────
# Sidebar — Contrôles
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Contrôles")

    st.subheader("🔧 Simulation")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Démarrer", key="start_sim"):
            result = api_call("post", "/sim/start", json={"duration": 120, "virtual": True})
            if result:
                st.success(f"Simulation démarrée")
    with col2:
        if st.button("⏹️ Arrêter", key="stop_sim"):
            result = api_call("post", "/sim/stop")
            if result:
                st.info("Simulation arrêtée")

    st.divider()

    st.subheader("⚔️ Attaque")
    attack_mode = st.selectbox("Mode", ["naive", "stealth"])
    target_speed = st.slider("Vitesse cible (km/h)", 0, 300, 200)

    col3, col4 = st.columns(2)
    with col3:
        if st.button("🔴 Lancer", key="start_attack"):
            result = api_call(
                "post",
                "/attack/start",
                json={
                    "mode": attack_mode,
                    "target_id": "0x130",
                    "target_speed": target_speed,
                    "duration": 60,
                },
            )
            if result:
                st.warning(f"Attaque {attack_mode} lancée!")
    with col4:
        if st.button("⛔ Stop", key="stop_attack"):
            result = api_call("post", "/attack/stop")
            if result:
                st.info("Attaque arrêtée")

    st.divider()

    auto_refresh = st.checkbox("🔄 Auto-refresh (1s)", value=True)


# ──────────────────────────────────────────────
# Récupérer les données
# ──────────────────────────────────────────────
metrics = api_call("get", "/metrics")
ids_status = api_call("get", "/ids/status")

# ──────────────────────────────────────────────
# KPIs
# ──────────────────────────────────────────────
st.subheader("📊 Indicateurs")
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

if metrics:
    kpi1.metric("Frames Total", f"{metrics.get('frames_total', 0):,}")
    kpi2.metric("Frames Attaque", f"{metrics.get('frames_attack', 0):,}")
    kpi3.metric("Alertes IDS", f"{metrics.get('alerts_total', 0):,}")
    kpi4.metric("Score CUSUM", f"{metrics.get('cusum_score', 0):.2f}")
    kpi5.metric("Uptime (s)", f"{metrics.get('uptime_s', 0):.0f}")

st.divider()

# ──────────────────────────────────────────────
# Graphiques
# ──────────────────────────────────────────────
chart_col1, chart_col2 = st.columns(2)

# --- Vitesse réelle vs affichée ---
with chart_col1:
    st.subheader("🚗 Vitesse réelle vs affichée")

    if len(st.session_state.speed_history) > 1:
        t_vals = list(range(len(st.session_state.speed_history)))
        speeds = list(st.session_state.speed_history)

        fig_speed = go.Figure()
        fig_speed.add_trace(
            go.Scatter(
                x=t_vals,
                y=[s[0] for s in speeds],
                mode="lines",
                name="Vitesse réelle",
                line=dict(color="green", width=2),
            )
        )
        fig_speed.add_trace(
            go.Scatter(
                x=t_vals,
                y=[s[1] for s in speeds],
                mode="lines",
                name="Vitesse affichée",
                line=dict(color="red", width=2, dash="dash"),
            )
        )
        fig_speed.update_layout(
            yaxis_title="km/h",
            xaxis_title="Time",
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_speed, use_container_width=True)
    else:
        st.info("En attente de données...")

# --- Score IDS live ---
with chart_col2:
    st.subheader("🛡️ Score IDS")

    if len(st.session_state.ids_score_history) > 1:
        t_vals = list(range(len(st.session_state.ids_score_history)))
        fig_ids = go.Figure()
        fig_ids.add_trace(
            go.Scatter(
                x=t_vals,
                y=list(st.session_state.ids_score_history),
                mode="lines+markers",
                name="Score IDS",
                line=dict(color="orange", width=2),
                marker=dict(size=4),
            )
        )
        fig_ids.add_hline(
            y=0.5, line_dash="dash", line_color="red",
            annotation_text="Seuil alerte",
        )
        fig_ids.update_layout(
            yaxis_title="Score",
            xaxis_title="Time",
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_ids, use_container_width=True)
    else:
        st.info("En attente de données...")

# --- CUSUM ---
chart_col3, chart_col4 = st.columns(2)

with chart_col3:
    st.subheader("📈 CUSUM")

    if len(st.session_state.cusum_history) > 1:
        t_vals = list(range(len(st.session_state.cusum_history)))
        fig_cusum = go.Figure()
        fig_cusum.add_trace(
            go.Scatter(
                x=t_vals,
                y=list(st.session_state.cusum_history),
                mode="lines",
                name="CUSUM",
                fill="tozeroy",
                line=dict(color="purple", width=2),
            )
        )
        fig_cusum.add_hline(
            y=15.0, line_dash="dash", line_color="red",
            annotation_text="Seuil CUSUM",
        )
        fig_cusum.update_layout(
            yaxis_title="Score CUSUM",
            xaxis_title="Time",
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_cusum, use_container_width=True)
    else:
        st.info("En attente de données...")

# --- RPM ---
with chart_col4:
    st.subheader("🔧 RPM Moteur")

    if len(st.session_state.rpm_history) > 1:
        t_vals = list(range(len(st.session_state.rpm_history)))
        fig_rpm = go.Figure()
        fig_rpm.add_trace(
            go.Scatter(
                x=t_vals,
                y=list(st.session_state.rpm_history),
                mode="lines",
                name="RPM",
                line=dict(color="blue", width=2),
            )
        )
        fig_rpm.update_layout(
            yaxis_title="RPM",
            xaxis_title="Time",
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_rpm, use_container_width=True)
    else:
        st.info("En attente de données...")

# ──────────────────────────────────────────────
# Histogramme fréquence ID
# ──────────────────────────────────────────────
st.subheader("📊 Fréquence par ID CAN")

if ids_status and "rules" in ids_status:
    rules_data = ids_status["rules"]
    alert_rules = rules_data.get("alerts_by_rule", {})
    if alert_rules:
        fig_hist = px.bar(
            x=list(alert_rules.keys()),
            y=list(alert_rules.values()),
            labels={"x": "Règle", "y": "Nombre d'alertes"},
            color=list(alert_rules.keys()),
        )
        fig_hist.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.success("✅ Aucune alerte IDS")

# ──────────────────────────────────────────────
# Détails IDS
# ──────────────────────────────────────────────
with st.expander("🔍 Détails IDS"):
    if ids_status:
        st.json(ids_status)

# ──────────────────────────────────────────────
# Auto-refresh
# ──────────────────────────────────────────────
if auto_refresh:
    # Mettre à jour les historiques
    if metrics:
        cusum_score = metrics.get("cusum_score", 0)
        st.session_state.cusum_history.append(cusum_score)

        rule_summary = metrics.get("rule_ids_summary", {})
        ids_score = rule_summary.get("current_score", 0)
        st.session_state.ids_score_history.append(ids_score)

        # Simuler les données (en attendant le WebSocket)
        real_speed = np.random.normal(60, 2) if not metrics.get("frames_total") else 60
        displayed_speed = real_speed
        if metrics.get("frames_attack", 0) > 0:
            displayed_speed = real_speed + np.random.normal(50, 10)
        st.session_state.speed_history.append((real_speed, displayed_speed))

        rpm = 800 + np.random.normal(0, 50)
        st.session_state.rpm_history.append(rpm)

    time.sleep(1)
    st.rerun()

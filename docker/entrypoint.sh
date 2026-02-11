#!/bin/bash
set -e

# Setup vcan0 si possible (nécessite --privileged)
if command -v modprobe &> /dev/null; then
    modprobe vcan 2>/dev/null || true
    ip link add dev vcan0 type vcan 2>/dev/null || true
    ip link set up vcan0 2>/dev/null || true
fi

case "$1" in
    api)
        echo "🚀 Lancement API FastAPI sur port 8000..."
        exec uvicorn canlab.api.main:app --host 0.0.0.0 --port 8000
        ;;
    dashboard)
        echo "📊 Lancement Dashboard Streamlit sur port 8501..."
        exec streamlit run src/canlab/ui/app_streamlit.py --server.port 8501 --server.address 0.0.0.0
        ;;
    sim)
        echo "🔧 Lancement Simulation CAN..."
        exec python -m canlab.sim.bus
        ;;
    *)
        exec "$@"
        ;;
esac

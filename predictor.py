class PhasePredictor:
    def __init__(self):
        pass

    def predict(self, episodes):
        if len(episodes) < 3:
            return {"risk": 0.0, "note": "Not enough data"}

        T_vals = [ep["tension"] for ep in episodes[-3:]]
        trend = (T_vals[-1] - T_vals[0]) / 3

        risk = min(1.0, max(0.0, (T_vals[-1] + trend)))

        return {
            "risk": float(risk),
            "trend": float(trend)
        }

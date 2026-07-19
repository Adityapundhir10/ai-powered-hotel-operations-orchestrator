import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib
from app.ml.occupancy_forecaster import FEATURES


def main():
    model_path = ROOT / "models" / "occupancy_xgb.joblib"
    data_path = ROOT / "data" / "synthetic_hotel_demand.csv"
    output = ROOT / "artifacts" / "evaluations" / "shap_summary.png"
    bundle = joblib.load(model_path)
    model = bundle["model"]
    frame = pd.read_csv(data_path)
    dates = pd.to_datetime(frame["forecast_date"])
    frame["day_of_week"] = dates.dt.dayofweek
    frame["month"] = dates.dt.month
    frame["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int)
    sample = frame[FEATURES].tail(min(300, len(frame)))
    explainer = shap.TreeExplainer(model)
    values = explainer(sample)
    output.parent.mkdir(parents=True, exist_ok=True)
    shap.plots.beeswarm(values, show=False)
    plt.tight_layout()
    plt.savefig(output, dpi=160, bbox_inches="tight")
    print(f"Saved SHAP summary to {output}")


if __name__ == "__main__":
    main()

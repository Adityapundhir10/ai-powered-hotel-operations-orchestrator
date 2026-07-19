import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from pathlib import Path
import json
import pandas as pd

from app.ml.occupancy_forecaster import OccupancyForecaster

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic_hotel_demand.csv"
MODEL = ROOT / "models" / "occupancy_xgb.joblib"
METRICS = ROOT / "artifacts" / "evaluations" / "forecast_metrics.json"


def main():
    if not DATA.exists():
        from scripts.generate_synthetic_data import generate
        DATA.parent.mkdir(parents=True, exist_ok=True)
        generate().to_csv(DATA, index=False)
    frame = pd.read_csv(DATA).sort_values("forecast_date")
    split = int(len(frame) * 0.80)
    train, validation = frame.iloc[:split], frame.iloc[split:]
    model = OccupancyForecaster(MODEL)
    metrics = model.fit(train, validation)
    model.save(MODEL)
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(metrics.__dict__, indent=2), encoding="utf-8")
    print(json.dumps(metrics.__dict__, indent=2))
    print(f"Saved model to {MODEL}")


if __name__ == "__main__":
    main()

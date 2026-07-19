import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from pathlib import Path
from datetime import date, timedelta
import argparse
import math
import random
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def generate(rows: int = 2400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = date(2023, 1, 1)
    records = []
    for i in range(rows):
        forecast_date = start + timedelta(days=i % 1095)
        month = forecast_date.month
        dow = forecast_date.weekday()
        weekend = int(dow >= 5)
        seasonal = 0.12 * math.sin(2 * math.pi * (forecast_date.timetuple().tm_yday / 365.25))
        event = float(rng.beta(1.4, 5.0))
        lead_time = int(rng.integers(0, 90))
        lag7 = float(np.clip(0.61 + seasonal + 0.07 * weekend + rng.normal(0, 0.08), 0.10, 0.98))
        lag28 = float(np.clip(0.59 + seasonal + rng.normal(0, 0.06), 0.10, 0.98))
        rolling = float(np.clip(0.58 + seasonal + 0.08 * weekend + 0.10 * event + rng.normal(0, 0.07), 0.05, 1.0))
        cancellation = float(np.clip(rng.normal(0.12 + 0.0007 * lead_time, 0.035), 0.01, 0.45))
        pickup = float(max(0, rng.normal(16 + 25 * rolling + 12 * event - 0.10 * lead_time, 6)))
        inventory = int(np.clip(rng.normal(48 - 37 * rolling - 10 * event, 8), 0, 80))
        occupancy = np.clip(
            0.13 + 0.39 * rolling + 0.20 * lag7 + 0.12 * lag28 + 0.0035 * pickup
            + 0.10 * event + 0.045 * weekend - 0.17 * cancellation - 0.0015 * inventory
            - 0.00025 * lead_time + rng.normal(0, 0.025),
            0.04, 0.99,
        )
        records.append({
            "hotel_id": f"HOTEL-{1 + (i % 12):03d}",
            "forecast_date": forecast_date.isoformat(),
            "lead_time": lead_time,
            "booking_pickup_7d": round(pickup, 3),
            "rolling_demand_14d": round(rolling, 4),
            "cancellation_rate_28d": round(cancellation, 4),
            "available_inventory": inventory,
            "event_intensity": round(event, 4),
            "lag_occupancy_7d": round(lag7, 4),
            "lag_occupancy_28d": round(lag28, 4),
            "occupancy_rate": round(float(occupancy), 4),
        })
    return pd.DataFrame(records).sort_values("forecast_date")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = ROOT / "data" / "synthetic_hotel_demand.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = generate(args.rows, args.seed)
    frame.to_csv(output, index=False)
    print(f"Wrote {len(frame)} synthetic rows to {output}")


if __name__ == "__main__":
    main()

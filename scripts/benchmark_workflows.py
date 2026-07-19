import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from pathlib import Path
import argparse
import asyncio
import json
import statistics
import time
from datetime import datetime, timezone

from app.schemas import ComplaintRequest
from app.workflows.hotel_ops_graph import HotelOperationsWorkflow

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = [
    "Water is leaking from the bathroom ceiling",
    "The room is dirty and towels are missing",
    "There is loud music from the next room",
    "My bill contains an incorrect charge",
    "Smoke is coming from the electrical panel",
]


async def run_one(engine, index, semaphore):
    async with semaphore:
        request = ComplaintRequest(
            complaint_id=f"BENCH-{index}", hotel_id=f"HOTEL-{index % 12:03d}",
            room_number=str(100 + index % 500), language="en", text=SAMPLES[index % len(SAMPLES)]
        )
        start = time.perf_counter()
        try:
            await engine.run(request)
            return True, (time.perf_counter() - start) * 1000
        except Exception:
            return False, (time.perf_counter() - start) * 1000


async def benchmark(events, concurrency):
    engine = HotelOperationsWorkflow()
    semaphore = asyncio.Semaphore(concurrency)
    started = time.perf_counter()
    results = await asyncio.gather(*(run_one(engine, i, semaphore) for i in range(events)))
    duration = time.perf_counter() - started
    latencies = [latency for _, latency in results]
    successes = sum(ok for ok, _ in results)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "events": events,
        "concurrency": concurrency,
        "successful_events": successes,
        "reliability": successes / events if events else 0,
        "duration_seconds": round(duration, 4),
        "throughput_events_per_second": round(events / duration, 2) if duration else 0,
        "latency_ms_mean": round(statistics.mean(latencies), 3),
        "latency_ms_p95": round(sorted(latencies)[int(0.95 * (len(latencies)-1))], 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=25)
    args = parser.parse_args()
    result = asyncio.run(benchmark(args.events, args.concurrency))
    output_dir = ROOT / "artifacts" / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"workflow_{args.events}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

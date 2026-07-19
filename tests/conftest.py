from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os
os.environ.setdefault("USE_TRANSFORMERS", "false")
os.environ.setdefault("USE_WEAVIATE", "false")
os.environ.setdefault("DATABASE_URL", "")
os.environ.setdefault("REDIS_URL", "")
os.environ.setdefault("KAFKA_BOOTSTRAP_SERVERS", "")

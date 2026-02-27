#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/4] compile check"
python -m compileall app/corpus/query_api.py src/make_submission.py

echo "[2/4] deterministic submission build"
python src/make_submission.py --verify-determinism --output outputs/submission_smoke.csv

echo "[3/4] app endpoint checks"
python app/corpus/query_api.py > outputs/mim_smoke_server.log 2>&1 &
APP_PID=$!
cleanup() {
  kill "$APP_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

python - <<'PY'
import json, time, urllib.request

def wait_for(url: str, timeout_sec: int = 30):
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout_sec:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                return r.read().decode("utf-8")
        except Exception as e:
            last = e
            time.sleep(0.5)
    raise RuntimeError(f"timeout waiting for {url}: {last}")

health_raw = wait_for("http://127.0.0.1:50001/corpus/health")
health = json.loads(health_raw)
assert health.get("ok") is True, health
print("health ok")

demo_raw = wait_for("http://127.0.0.1:50001/corpus/demo?limit=3")
demo = json.loads(demo_raw)
assert demo.get("ok") is True, demo
print("demo ok")
PY

echo "smoke test passed"

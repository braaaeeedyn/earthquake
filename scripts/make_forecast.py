"""Emit app/public/forecast.json — the contract the web app reads (MVP §3.5).

PLACEHOLDER: until the fused model exists (Phase 4-6), this writes a clearly-marked
placeholder forecast so the app surface can be built and tested. Phase 6 replaces
``compute_forecast`` with real inference over the latest geomagnetic window.
"""
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eq.config import Config  # noqa: E402


def compute_forecast(cfg: Config) -> dict:
    # PLACEHOLDER — no trained model yet.
    probability = 0.23
    return {
        "schema_version": 1,
        "region": "Southern California cluster (synthetic)",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "horizon_days": cfg.labeling.horizon_days,
        "probability": probability,
        "label": "likely" if probability >= 0.5 else "unlikely",
        "confidence": "low",
        "is_placeholder": True,
        "model_version": "none-placeholder",
        "stations": [s.code for s in cfg.stations],
        "disclaimer": (
            "Research prototype. Not for safety-critical decisions. Near-term earthquake "
            "prediction from geomagnetic precursors is scientifically unproven."
        ),
    }


def main() -> None:
    fc = compute_forecast(Config())
    out = Path(__file__).resolve().parents[1] / "app" / "public" / "forecast.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fc, indent=2))
    tag = "[PLACEHOLDER] " if fc["is_placeholder"] else ""
    print(f"{tag}Wrote {out}")


if __name__ == "__main__":
    main()

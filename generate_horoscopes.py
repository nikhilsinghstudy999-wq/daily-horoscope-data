#!/usr/bin/env python3
"""
Batch‑Aware Per‑Rashi Horoscope Generator — Groq (free tier), Token‑Safe.

Reads optional env vars BATCH_START, BATCH_END to generate a subset of
rashis.  Writes one JSON file per rashi into data/rashi/ (e.g. Mesha.json).
Also creates a combined data/horoscopes.json for your existing frontend.

Model: llama-3.1-8b-instant (1M TPD, 14.4k RPD — far above our needs).
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("horoscope")

# ---------------------------------------------------------------------------
# Dep checks
# ---------------------------------------------------------------------------
try:
    import swisseph as swe
    from groq import Groq
    from pydantic import BaseModel, Field, field_validator
    from json_repair import repair_json
    from tenacity import (
        retry, stop_after_attempt, wait_exponential,
        retry_if_exception_type, before_sleep_log,
    )
except ImportError as e:
    log.critical("Missing dependency: %s", e)
    sys.exit(1)

# ============================================================================
# 0. Ephemeris
# ============================================================================
def setup_ephemeris() -> None:
    ephe_dir = tempfile.mkdtemp(prefix="sweph_")
    swe.set_ephe_path(ephe_dir)
    swe.set_sid_mode(swe.SIDM_LAHIRI)
    log.info("Ephemeris: Moshier, Lahiri")

# ============================================================================
# 1. Configuration
# ============================================================================
RASHIS = [
    "Mesha (Aries)", "Vrishabha (Taurus)", "Mithuna (Gemini)",
    "Karka (Cancer)", "Simha (Leo)", "Kanya (Virgo)",
    "Tula (Libra)", "Vrishchika (Scorpio)", "Dhanu (Sagittarius)",
    "Makara (Capricorn)", "Kumbha (Aquarius)", "Meena (Pisces)"
]

RASHI_SHORT = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]

PLANETS = {
    swe.SUN: "Sun", swe.MOON: "Moon", swe.MARS: "Mars",
    swe.MERCURY: "Mercury", swe.JUPITER: "Jupiter", swe.VENUS: "Venus",
    swe.SATURN: "Saturn", swe.MEAN_NODE: "Rahu",
}

MODEL = "llama-3.1-8b-instant"      # 1M TPD free, fast
MAX_OUTPUT_TOKENS = 500              # enough for 2‑4 sentences per field
REQUEST_DELAY = 8                    # seconds between calls (well under 30 RPM)

# Batch control (env vars BATCH_START, BATCH_END are 0‑based indices)
try:
    BATCH_START = int(os.environ.get("BATCH_START", "0"))
    BATCH_END = int(os.environ.get("BATCH_END", "12"))
except ValueError:
    BATCH_START, BATCH_END = 0, 12
BATCH_END = min(BATCH_END, len(RASHIS))

# ---------------------------------------------------------------------------
# Pydantic model (same as before, with colour extractor)
# ---------------------------------------------------------------------------
KNOWN_COLORS = {
    "red","orange","yellow","green","blue","indigo","violet",
    "purple","pink","white","black","brown","grey","gray",
    "silver","gold","maroon","crimson","turquoise","teal",
    "magenta","cyan","navy","beige","coral","peach","mint",
    "lavender","sky","lime","rose","olive","ruby","sapphire",
    "emerald","amber","topaz","jade","cobalt","copper","bronze",
    "platinum","aqua","avocado","champagne","charcoal","chestnut",
    "chocolate","citrine","cream","ebony","forest","fuchsia",
    "ginger","honey","ivory","khaki","lemon","lilac","mahogany",
    "mustard","onyx","opal","periwinkle","rust","sand","scarlet",
    "sepia","sienna","tan","taupe","tomato","wheat",
}

class HoroscopeEntry(BaseModel):
    general: str = Field(..., min_length=20, max_length=2000)
    luck: str = Field(..., min_length=10, max_length=2000)
    scope: str = Field(..., min_length=10, max_length=2000)
    study: str = Field(..., min_length=10, max_length=2000)
    love: str = Field(..., min_length=10, max_length=2000)
    travel: str = Field(..., min_length=10, max_length=2000)
    lucky_number: int = Field(..., ge=1, le=100)
    lucky_color: str = Field(..., min_length=2, max_length=30)

    @field_validator("lucky_color", mode="before")
    @classmethod
    def extract_color(cls, v: Any) -> str:
        raw = str(v).strip().lower()
        words = re.findall(r"[a-zA-Z]+", raw)
        for w in words:
            if w in KNOWN_COLORS:
                return w.capitalize()
        return words[0].capitalize() if words else "Red"

    @field_validator("*", mode="before")
    @classmethod
    def coerce_str(cls, v: Any) -> str:
        return str(v)

# ============================================================================
# 2. Ephemeris helpers
# ============================================================================
def compute_planet_positions(jd: float) -> Dict[str, Dict[str, Any]]:
    positions = {}
    for pid, pname in PLANETS.items():
        xx, _ = swe.calc_ut(jd, pid, swe.FLG_SIDEREAL | swe.FLG_SPEED)
        lon = xx[0]
        sign_idx = int(lon // 30)
        degree = round(lon % 30, 2)
        positions[pname] = {"sign_idx": sign_idx, "sign_name": RASHI_SHORT[sign_idx], "degree": degree}
    rahu = positions["Rahu"]
    ketu_sign = (rahu["sign_idx"] + 6) % 12
    ketu_degree = (rahu["degree"] + 180) % 360
    positions["Ketu"] = {"sign_idx": ketu_sign, "sign_name": RASHI_SHORT[ketu_sign], "degree": round(ketu_degree, 2)}
    return positions

def build_transit_text(rashi_idx: int, positions: Dict[str, Dict[str, Any]]) -> str:
    parts = []
    for pname, data in positions.items():
        house = (data["sign_idx"] - rashi_idx) % 12 + 1
        parts.append(f"{pname} in house {house} ({data['sign_name']} {data['degree']}°)")
    return " | ".join(parts)

# ============================================================================
# 3. Groq client + retry
# ============================================================================
def _get_groq_client() -> Groq:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        log.critical("GROQ_API_KEY not set")
        sys.exit(1)
    return Groq(api_key=key)

SYSTEM_PROMPT = (
    "You are a Vedic astrologer. "
    "Return a JSON object with exactly these keys: general, luck, scope, study, love, travel, lucky_number, lucky_color. "
    "lucky_number is an integer 1‑100, lucky_color is a single colour name (e.g. 'Red'). "
    "Each text field: 2‑3 detailed, encouraging sentences grounded in the given real transits."
)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=30),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(log, logging.WARNING),
    reraise=True,
)
def call_groq(client: Groq, prompt: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,
        top_p=0.9,
        max_completion_tokens=MAX_OUTPUT_TOKENS,
    )
    content = resp.choices[0].message.content
    if not content:
        raise ValueError("Empty response from Groq")
    return content.strip()

def parse_response(raw: str) -> HoroscopeEntry:
    # Try direct repair first
    try:
        obj = repair_json(raw, return_objects=True)
        return HoroscopeEntry.model_validate(obj)
    except Exception:
        pass
    # Regex extract + repair
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            obj = repair_json(match.group(), return_objects=True)
            return HoroscopeEntry.model_validate(obj)
        except Exception:
            pass
    raise ValueError(f"Could not parse response: {raw[:200]}")

# ============================================================================
# 4. Per‑rashi generation
# ============================================================================
def generate_one_rashi(
    rashi_name: str,
    today_str: str,
    transit_text: str,
) -> HoroscopeEntry:
    prompt = (
        f"Rashi: {rashi_name}\nToday: {today_str}\n\n"
        f"Real planetary transits (house positions from {rashi_name}):\n{transit_text}\n\n"
        f"Based ONLY on these exact positions, generate the horoscope JSON."
    )

    client = _get_groq_client()
    for attempt in range(1, 4):
        try:
            raw = call_groq(client, prompt)
            return parse_response(raw)
        except Exception as e:
            log.warning("  Attempt %d failed: %s", attempt, e)
            if attempt < 3:
                time.sleep(4)
    raise RuntimeError(f"Failed after 3 attempts for {rashi_name}")

# ============================================================================
# 5. Main
# ============================================================================
def main() -> None:
    log.info("===== Batch Horoscope Generator START (rashi %d‑%d) =====",
             BATCH_START, BATCH_END - 1)
    setup_ephemeris()

    today = datetime.now()
    today_str = today.strftime("%B %d, %Y")
    today_iso = today.isoformat()
    jd = swe.julday(today.year, today.month, today.day, 0.0)

    positions = compute_planet_positions(jd)
    for name, data in positions.items():
        log.info("  %s: %s %.2f°", name, data["sign_name"], data["degree"])

    os.makedirs("data/rashi", exist_ok=True)

    rashi_horoscopes: Dict[str, HoroscopeEntry] = {}

    for idx in range(BATCH_START, BATCH_END):
        rashi = RASHIS[idx]
        log.info("--- %s ---", rashi)
        transit = build_transit_text(idx, positions)

        # Check if already generated today (safety for overlapping batches)
        out_file = Path("data/rashi") / f"{RASHI_SHORT[idx]}.json"
        if out_file.exists():
            try:
                existing = json.loads(out_file.read_text(encoding="utf-8"))
                if existing.get("date") == today_iso:
                    log.info("  Already generated today. Skipping.")
                    # Load existing entry for combined file
                    rashi_horoscopes[rashi] = HoroscopeEntry.model_validate(existing["data"])
                    continue
            except Exception:
                pass  # corrupted – regenerate

        entry = generate_one_rashi(rashi, today_str, transit)
        rashi_horoscopes[rashi] = entry

        # Write per‑rashi file
        rashi_data = {
            "date": today_iso,
            "rashi": rashi,
            "data": entry.model_dump()
        }
        out_file.write_text(json.dumps(rashi_data, ensure_ascii=False, indent=2), encoding="utf-8")

        # Git add the single file immediately
        os.system(f'git add "{out_file}"')

        log.info("  ✓ saved to %s", out_file.name)

        time.sleep(REQUEST_DELAY)

    # Write combined file (for backward compatibility)
    combined = {
        "date": today_iso,
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "rashi_horoscopes": {r: e.model_dump() for r, e in rashi_horoscopes.items()}
    }
    combined_path = Path("data/horoscopes.json")
    tmp = combined_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(combined_path)

    log.info("✅ Batch complete. Combined file updated.")

if __name__ == "__main__":
    main()

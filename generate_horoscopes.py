#!/usr/bin/env python3
"""
BATCHED DAILY HOROSCOPE GENERATOR — Bulletproof Edition.

- Real planetary positions (Moshier ephemeris, Lahiri ayanamsa)
- Groq API (llama-3.1-8b-instant, 1 M free tokens/day)
- Per‑rashi JSON files + combined output
- Multi‑stage JSON parser that handles nested description objects,
  markdown fences, preamble text, and minor syntax errors
- Prompt auto‑tightening on validation failure
- Full‑jitter exponential backoff for rate‑limit resilience
- Yesterday’s data as guaranteed fallback (logged prominently)
- Idempotent: skips already‑generated rashis for today
"""

import json
import logging
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("horoscope")

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
try:
    import swisseph as swe
    from groq import Groq
    from pydantic import BaseModel, Field, field_validator, ValidationError
    from json_repair import repair_json
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
    log.info("Ephemeris: Moshier built‑in, Lahiri ayanamsa")

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

MODEL = "llama-3.1-8b-instant"   # 1 M TPD free
MAX_OUTPUT_TOKENS = 600
REQUEST_DELAY = 8.0               # seconds between API calls (safe within 30 RPM)

# Batch control (env vars)
try:
    BATCH_START = int(os.environ.get("BATCH_START", "0"))
    BATCH_END = int(os.environ.get("BATCH_END", "12"))
except ValueError:
    BATCH_START, BATCH_END = 0, 12
BATCH_END = min(BATCH_END, len(RASHIS))

# Pydantic model
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
# 3. The Unbreakable JSON Parser
# ============================================================================
def _unwrap_description(obj: Any) -> Any:
    """Recursively replace {"description": "value"} with just "value"."""
    if isinstance(obj, dict):
        if list(obj.keys()) == ["description"] and isinstance(obj["description"], str):
            return obj["description"]
        return {k: _unwrap_description(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_unwrap_description(item) for item in obj]
    return obj

def _extract_json_candidate(text: str) -> Optional[str]:
    """Extract the most promising JSON substring from raw LLM output."""
    # 1. Strip markdown fences
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()

    # 2. Find the outermost braces
    start = text.find("{")
    if start == -1:
        return None
    # Find matching closing brace using simple stack
    stack = 0
    end = -1
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            stack += 1
        elif ch == "}":
            stack -= 1
            if stack == 0:
                end = i
                break
    if end != -1:
        return text[start:end+1]
    return None

def _try_parse_with_repair(json_str: str) -> Optional[dict]:
    """Try json_repair, then regex extract + repair."""
    try:
        obj = repair_json(json_str, return_objects=True)
        return _unwrap_description(obj) if isinstance(obj, dict) else None
    except Exception:
        pass
    # Regex extract the first JSON object
    match = re.search(r"\{.*\}", json_str, re.DOTALL)
    if match:
        try:
            obj = repair_json(match.group(), return_objects=True)
            return _unwrap_description(obj) if isinstance(obj, dict) else None
        except Exception:
            pass
    return None

def parse_response(raw: str) -> HoroscopeEntry:
    # Step 1: Extract the JSON candidate
    candidate = _extract_json_candidate(raw)
    if not candidate:
        raise ValueError("No JSON object found in response")

    # Step 2: Try repairing and unwrapping
    obj = _try_parse_with_repair(candidate)
    if obj is not None:
        try:
            return HoroscopeEntry.model_validate(obj)
        except ValidationError:
            pass

    # Step 3: Aggressive fallback – attempt to fix missing quotes around keys/values
    # (json_repair already did this, but just in case)
    # Then try again
    try:
        fixed = re.sub(r'(?<=\{|\,)\s*(\w+)\s*:', r'"\1":', candidate)
        obj = repair_json(fixed, return_objects=True)
        obj = _unwrap_description(obj)
        return HoroscopeEntry.model_validate(obj)
    except Exception:
        pass

    raise ValueError(f"Unable to parse horoscope JSON from response: {raw[:300]}")

# ============================================================================
# 4. Groq client with full‑jitter exponential backoff
# ============================================================================
def _get_groq_client() -> Groq:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        log.critical("GROQ_API_KEY not set")
        sys.exit(1)
    return Groq(api_key=key)

BASE_SYSTEM_PROMPT = (
    "You are a Vedic astrologer. "
    "Output RAW JSON. No markdown, no preamble, no code fences. "
    "The JSON must have exactly these top‑level keys: "
    "general, luck, scope, study, love, travel, lucky_number, lucky_color. "
    "All values must be STRINGS except lucky_number (integer 1‑100). "
    "lucky_color must be a single color name like 'Red', not a sentence. "
    "Each string value must be 2‑3 detailed sentences grounded in the given transits. "
    "DO NOT wrap values inside additional objects like {\"description\": \"...\"}. "
    "Just put the text directly as the value of the key."
)

def _build_prompt(rashi_name: str, today_str: str, transit_text: str, strict: bool = False) -> str:
    extra = ""
    if strict:
        extra = (
            "\nIMPORTANT: Your previous response was not parseable. "
            "Make absolutely sure the output is only a flat JSON object with string values. "
            "No nested objects, no markdown, no explanations."
        )
    return (
        f"Rashi: {rashi_name}\nToday: {today_str}\n\n"
        f"Real planetary transits (house positions from {rashi_name}):\n{transit_text}\n\n"
        f"Based ONLY on these exact positions, generate the horoscope JSON.{extra}"
    )

def call_groq(prompt: str, max_retries: int = 3) -> str:
    client = _get_groq_client()
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": BASE_SYSTEM_PROMPT},
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
        except Exception as e:
            last_error = e
            msg = str(e).lower()
            if "rate limit" in msg or "429" in msg:
                sleep_time = min(30, (2 ** attempt) + random.uniform(0, 1))
                log.warning("Rate‑limited, sleeping %.1fs", sleep_time)
                time.sleep(sleep_time)
            else:
                log.warning("Groq error (attempt %d): %s", attempt, e)
                time.sleep(2)
    raise RuntimeError(f"Groq call failed after {max_retries} attempts: {last_error}")

# ============================================================================
# 5. Per‑rashi generation with smart retries and fallback
# ============================================================================
def _load_previous_rashi(rashi_short: str) -> Optional[HoroscopeEntry]:
    """Try to load yesterday's per‑rashi file as a fallback."""
    path = Path("data/rashi") / f"{rashi_short}.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return HoroscopeEntry.model_validate(data["data"])
        except Exception as e:
            log.warning("Could not load previous data for %s: %s", rashi_short, e)
    # Absolute last resort: fallback_horoscopes.json
    try:
        fb = json.loads(Path("data/fallback_horoscopes.json").read_text(encoding="utf-8"))
        entry = fb["rashi_horoscopes"].get(RASHIS[RASHI_SHORT.index(rashi_short)])
        if entry:
            return HoroscopeEntry.model_validate(entry)
    except Exception:
        pass
    return None

def generate_one_rashi(
    rashi_name: str, today_str: str, transit_text: str, rashi_short: str
) -> HoroscopeEntry:
    # First try with normal prompt
    prompt = _build_prompt(rashi_name, today_str, transit_text, strict=False)
    for attempt in range(1, 4):
        try:
            raw = call_groq(prompt)
            return parse_response(raw)
        except Exception as e:
            log.warning("Attempt %d failed: %s", attempt, e)
            if attempt == 2:
                # On second failure, tighten the prompt
                prompt = _build_prompt(rashi_name, today_str, transit_text, strict=True)
            time.sleep(2)
    
    # All generations failed – fall back to yesterday's data (with loud warning)
    log.error("All 3 attempts failed for %s. Falling back to previous data.", rashi_name)
    prev = _load_previous_rashi(rashi_short)
    if prev is None:
        raise RuntimeError(f"Completely unable to generate or retrieve horoscope for {rashi_name}")
    return prev

# ============================================================================
# 6. Main
# ============================================================================
def main() -> None:
    log.info("===== BATCHED HOROSCOPE GENERATOR START (rashi %d‑%d) =====",
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
    # Collect any previously generated per‑rashi files for today
    for idx in range(BATCH_START, BATCH_END):
        rashi = RASHIS[idx]
        short = RASHI_SHORT[idx]
        out_file = Path("data/rashi") / f"{short}.json"
        if out_file.exists():
            try:
                existing = json.loads(out_file.read_text(encoding="utf-8"))
                if existing.get("date") == today_iso:
                    log.info("%s already generated today, reusing.", short)
                    rashi_horoscopes[rashi] = HoroscopeEntry.model_validate(existing["data"])
                    continue
            except Exception:
                log.warning("Corrupted file for %s, regenerating.", short)

        log.info("--- %s ---", rashi)
        transit = build_transit_text(idx, positions)
        try:
            entry = generate_one_rashi(rashi, today_str, transit, short)
        except Exception as e:
            log.critical("Unrecoverable failure for %s: %s", rashi, e)
            sys.exit(1)  # fail the workflow if even fallback is broken (should never happen)

        rashi_horoscopes[rashi] = entry

        # Write per‑rashi file atomically
        rashi_data = {"date": today_iso, "rashi": rashi, "data": entry.model_dump()}
        tmp = out_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(rashi_data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(out_file)
        log.info("  ✓ saved %s.json", short)

        time.sleep(REQUEST_DELAY)  # rate‑limit safety

    # Write combined file
    combined = {
        "date": today_iso,
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "rashi_horoscopes": {r: e.model_dump() for r, e in rashi_horoscopes.items()}
    }
    combined_path = Path("data/horoscopes.json")
    tmp_combined = combined_path.with_suffix(".tmp")
    tmp_combined.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_combined.replace(combined_path)

    log.info("✅ Batch complete.")

if __name__ == "__main__":
    import random  # for jitter
    main()

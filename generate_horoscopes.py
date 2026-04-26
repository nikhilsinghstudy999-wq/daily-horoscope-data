#!/usr/bin/env python3
"""
Production‑grade daily horoscope generator with:
- Same astronomical core (Moshier ephemeris + Lahiri ayanamsa)
- Rate‑limit‑aware retry
- Multi‑stage JSON repair that handles any truncation
- Compact prompt to stay well within token limits
"""

import json
import os
import re
import sys
import time
import tempfile
from datetime import datetime, timezone

import swisseph as swe
from sambanova import SambaNova

# ──────────────────────────────────────────────
# 0. EPHEMERIS
# ──────────────────────────────────────────────
def setup_ephemeris():
    ephe_dir = tempfile.mkdtemp(prefix="sweph_")
    swe.set_ephe_path(ephe_dir)
    swe.set_sid_mode(swe.SIDM_LAHIRI)
    print(f"Ephemeris: Moshier, Lahiri ayanamsa (path={ephe_dir})")
    return ephe_dir

# ──────────────────────────────────────────────
# 1. CONFIGURATION
# ──────────────────────────────────────────────
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
    swe.SUN:       "Sun",
    swe.MOON:      "Moon",
    swe.MARS:      "Mars",
    swe.MERCURY:   "Mercury",
    swe.JUPITER:   "Jupiter",
    swe.VENUS:     "Venus",
    swe.SATURN:    "Saturn",
    swe.MEAN_NODE: "Rahu",
}

# ──────────────────────────────────────────────
# 2. SAMBANOVA CLIENT
# ──────────────────────────────────────────────
SAMBANOVA_API_KEY = os.environ.get("SAMBANOVA_API_KEY")
if not SAMBANOVA_API_KEY:
    print("ERROR: Missing SAMBANOVA_API_KEY env var")
    sys.exit(1)

client = SambaNova(api_key=SAMBANOVA_API_KEY, base_url="https://api.sambanova.ai/v1")

MODEL = "gpt-oss-120b"
MAX_TOKENS = 600            # much smaller → the model can’t exceed it and truncate
TEMPERATURE = 0.8
TOP_P = 0.9
REQUEST_DELAY = 10          # seconds between consecutive calls (avoids 429)

# Retry settings
MAX_RETRIES = 5
BASE_RETRY_DELAY = 5        # seconds (increases after 429)

# System prompt that forces brevity and valid JSON
SYSTEM_PROMPT = (
    "You are a Vedic astrologer. Reply with a single, compact JSON object, no markdown. "
    "Keys: general, luck, scope, study, love, travel, lucky_number, lucky_color. "
    "All values strings except lucky_number (integer). "
    "Make each text value exactly 1‑2 short, specific sentences grounded in the given transits. "
    "Do not repeat the transit data – just interpret it. "
    "Keep the entire response under 500 tokens. "
    "Ensure the JSON is complete and correctly closed."
)

# ──────────────────────────────────────────────
# 3. ASTRONOMICAL CALCULATIONS (unchanged)
# ──────────────────────────────────────────────
def compute_planet_positions(jd):
    positions = {}
    for pid, pname in PLANETS.items():
        xx, ret_flag = swe.calc_ut(jd, pid, swe.FLG_SIDEREAL | swe.FLG_SPEED)
        lon = xx[0]
        sign_idx = int(lon // 30)
        degree = round(lon % 30, 2)
        positions[pname] = {
            "sign_idx": sign_idx,
            "sign_name": RASHI_SHORT[sign_idx],
            "degree": degree,
        }
    rahu = positions["Rahu"]
    ketu_sign = (rahu["sign_idx"] + 6) % 12
    ketu_degree = (rahu["degree"] + 180) % 360
    positions["Ketu"] = {
        "sign_idx": ketu_sign,
        "sign_name": RASHI_SHORT[ketu_sign],
        "degree": round(ketu_degree, 2),
    }
    return positions

def build_transit_text(rashi_idx, positions):
    parts = []
    for pname, data in positions.items():
        house = (data["sign_idx"] - rashi_idx) % 12 + 1
        parts.append(f"{pname} in house {house} ({data['sign_name']} {data['degree']}°)")
    return " | ".join(parts)

# ──────────────────────────────────────────────
# 4. SELF‑HEALING JSON PARSER
# ──────────────────────────────────────────────
def parse_json(content):
    """Multi‑layer JSON repair that can handle truncated strings."""
    if not content:
        raise ValueError("Empty content")

    # Strip markdown fences
    content = re.sub(r'^```(?:json)?\s*', '', content.strip())
    content = re.sub(r'\s*```$', '', content)

    # Try direct parse
    try:
        return json.loads(content)
    except:
        pass

    # Try regex extract
    try:
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            return json.loads(match.group())
    except:
        pass

    # Try basic repair (add missing closing braces only if the last char isn't a string fragment)
    try:
        repaired = repair_json(content)
        return json.loads(repaired)
    except:
        pass

    # Last resort: try to close the JSON by truncating the damaged string value
    try:
        closed = force_close_json(content)
        return json.loads(closed)
    except:
        pass

    raise ValueError(f"Cannot parse JSON. First 200 chars: {content[:200]}")

def repair_json(json_str):
    """Add missing closing braces and fix minor syntax errors."""
    open_braces = json_str.count('{')
    close_braces = json_str.count('}')
    if open_braces > close_braces:
        json_str += '}' * (open_braces - close_braces)
    # Replace single quotes with double quotes inside strings (naive)
    json_str = re.sub(r"(?<!\\)'", '"', json_str)
    return json_str

def force_close_json(json_str):
    """
    Handle truncated output where a string value is missing its closing quote.
    The strategy: find the last key‑value pair that is complete, then close the JSON.
    """
    # Pattern: "key":"value" where the value may be cut off.
    # We'll search for the last complete key‑value pair (ending with a comma or brace).
    # For truncated strings we try to append '"}'
    # First attempt: if the string ends with a partial value, remove the broken part.
    try:
        # Assume we have something like "general":"Sun in your first house... where Venus shines in the fi"
        # We can try to find the last valid '"' and close from there.
        # A simpler approach: remove the last partial key-value and close the JSON.
        # Find the last occurrence of ',"' (the start of a new key)
        last_comma = json_str.rfind(',"')
        if last_comma != -1:
            base = json_str[:last_comma] + '}'
            # Test if this is valid JSON
            json.loads(base)   # will raise if not, then fallback
            return base
    except:
        pass

    # If the above fails, try to close the JSON at the position where we have a valid object so far.
    # Remove characters from the end until we can parse.
    for i in range(len(json_str), 0, -1):
        test = json_str[:i] + '}'
        try:
            json.loads(test)
            return test
        except:
            continue

    raise ValueError("Unable to force-close truncated JSON")

# ──────────────────────────────────────────────
# 5. RATE‑LIMIT‑AWARE LLM CALL
# ──────────────────────────────────────────────
def generate_rashi(rashi_name, today_str, transit_text):
    """Generate one rashi with robust retries and 429 handling."""
    user_prompt = (
        f"Rashi: {rashi_name}\nToday: {today_str}\n\n"
        f"Transits (from {rashi_name}): {transit_text}\n\n"
        f"Return a JSON with the 8 fields. Be very concise."
    )

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=MAX_TOKENS,
            )
            content = response.choices[0].message.content
            data = parse_json(content)

            # Validate keys
            required = {"general", "luck", "scope", "study", "love", "travel",
                        "lucky_number", "lucky_color"}
            if not required.issubset(data):
                raise ValueError(f"Missing keys: {required - data.keys()}")

            # Normalise types
            data["lucky_number"] = int(data["lucky_number"])
            data["lucky_color"] = str(data["lucky_color"])
            for k in ["general", "luck", "scope", "study", "love", "travel"]:
                data[k] = str(data[k])

            return data

        except Exception as e:
            last_error = e
            # Check if it's a rate limit error (HTTP 429)
            is_rate_limit = "rate limit" in str(e).lower() or "429" in str(e)
            wait = 0
            if is_rate_limit:
                # Extract Retry-After header if available, else exponential backoff
                # The sambanova library might not expose headers; we assume 10*2**attempt
                wait = BASE_RETRY_DELAY * (2 ** attempt)   # 10s, 20s, 40s, ...
                print(f"  Rate limited; waiting {wait}s")
            print(f"  Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(wait or BASE_RETRY_DELAY)

    raise RuntimeError(
        f"Failed to generate horoscope for {rashi_name} after {MAX_RETRIES} attempts. "
        f"Last error: {last_error}"
    )

# ──────────────────────────────────────────────
# 6. MAIN
# ──────────────────────────────────────────────
def main():
    setup_ephemeris()

    today = datetime.now()
    today_str = today.strftime("%B %d, %Y")
    today_iso = today.isoformat()
    jd = swe.julday(today.year, today.month, today.day, 0.0)

    print("Computing planetary positions …")
    positions = compute_planet_positions(jd)
    for name, data in positions.items():
        print(f"  {name}: {data['sign_name']} {data['degree']}°")

    os.makedirs("data", exist_ok=True)

    output = {
        "date": today_iso,
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "rashi_horoscopes": {},
    }

    for idx, rashi in enumerate(RASHIS):
        print(f"\n===== {rashi} =====")
        transit = build_transit_text(idx, positions)
        print(f"Transits: {transit[:150]}…")
        data = generate_rashi(rashi, today_str, transit)
        output["rashi_horoscopes"][rashi] = data
        print("✓")
        # Wait between consecutive calls to avoid rate limits
        time.sleep(REQUEST_DELAY)

    out_path = os.path.join("data", "horoscopes.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✅ All 12 rashis saved to {out_path}")

if __name__ == "__main__":
    main()

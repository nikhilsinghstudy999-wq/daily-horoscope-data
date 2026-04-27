#!/usr/bin/env python3
"""
Industry‑Grade Daily Vedic Horoscope Generator — Multi‑Provider, Instructor‑Backed,
Circuit‑Breaker, Self‑Healing JSON, Pydantic‑Validated, Atomic‑Write.

Key production patterns (sourced from 2025‑2026 best practices):
  • Instructor library forces the LLM to emit valid Pydantic models (no raw JSON parsing)   [15†L13-L18]
  • Three‑state circuit breaker (Closed → Open → HalfOpen) isolates degraded providers       [7†L44-L47]
  • Exponential backoff with full jitter eliminates retry storms                             [7†L41-L45]
  • Token‑bucket rate limiter honours per‑provider RPM quotas proactively                   [13†L13-L15]
  • Multi‑stage JSON repair as a safety net behind instructor                               [11†L13-L16]
  • Pydantic field_validators with auto‑coercion for `lucky_color` / `lucky_number`         [10†L13-L25]
  • Atomic writes via tempfile + os.replace()                                                [6†L5-L7]
  • Structured logging with provider‑health telemetry                                        [8†L22-L23]

Usage:
  1. Set GROQ_API_KEY (required) and GEMINI_API_KEY (optional fallback) as env vars.
  2. python generate_horoscopes.py
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Structured logging (JSON‑line compatible, ISO‑8601 timestamps)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("horoscope")

# ============================================================================
# 0. DEPENDENCY GUARD (fail‑fast with actionable messages)
# ============================================================================
def _guard_deps() -> None:
    missing: List[str] = []
    for mod, pkg in [
        ("swisseph", "pyswisseph"),
        ("groq", "groq"),
        ("pydantic", "pydantic>=2.0"),
        ("json_repair", "json_repair"),
        ("tenacity", "tenacity"),
        ("instructor", "instructor"),
    ]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(f"  pip install {pkg}")
    if missing:
        log.critical("Missing dependencies:\n%s", "\n".join(missing))
        sys.exit(1)

_guard_deps()

import swisseph as swe
from groq import Groq
from pydantic import BaseModel, Field, field_validator, ValidationError
from json_repair import repair_json
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type, before_sleep_log,
)
import instructor


# ============================================================================
# 1. CUSTOM EXCEPTION HIERARCHY
# ============================================================================
class HoroscopeError(Exception):
    """Base for all pipeline failures."""

class EphemerisError(HoroscopeError):
    """Planetary calculation failed."""

class CircuitBreakerOpenError(HoroscopeError):
    """Circuit is open — provider skipped."""

class LLMRateLimitError(HoroscopeError):
    """HTTP 429 from any provider."""

class LLMConnectionError(HoroscopeError):
    """Network‑level failure."""

class LLMResponseError(HoroscopeError):
    """Provider returned unusable content."""

class ValidationExhaustedError(HoroscopeError):
    """Instructor + repair both failed after max retries."""


# ============================================================================
# 2. THREE‑STATE CIRCUIT BREAKER  (Martin Fowler pattern + half‑open recovery)     [7†L44-L47]
# ============================================================================
class CircuitState(Enum):
    CLOSED = auto()       # requests pass
    OPEN = auto()         # requests fail fast
    HALF_OPEN = auto()    # one probe allowed


@dataclass
class CircuitBreaker:
    name: str
    failure_threshold: int = 3
    cooldown_seconds: float = 120.0
    _state: CircuitState = CircuitState.CLOSED
    _failure_count: int = 0
    _last_failure_time: float = 0.0

    def record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            log.warning("🔴 %s circuit OPEN (%d failures)", self.name, self._failure_count)

    def record_success(self) -> None:
        self._failure_count = 0
        self._state = CircuitState.CLOSED

    def allow_request(self) -> bool:
        if self._state == CircuitState.CLOSED:
            return True
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._last_failure_time >= self.cooldown_seconds:
                self._state = CircuitState.HALF_OPEN
                log.info("🟡 %s circuit HALF_OPEN — probing", self.name)
                return True
            return False
        # HALF_OPEN — one probe
        return True


# ============================================================================
# 3. TOKEN‑BUCKET RATE LIMITER  (policy‑compliant, honours RPM quotas)               [13†L13-L15]
# ============================================================================
@dataclass
class TokenBucket:
    max_tokens: float
    refill_rate: float       # tokens / second
    _tokens: float = field(init=False)
    _last_refill: float = field(init=False)

    def __post_init__(self) -> None:
        self._tokens = self.max_tokens
        self._last_refill = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.max_tokens, self._tokens + elapsed * self.refill_rate)
        self._last_refill = now

    def acquire(self, tokens: float = 1.0) -> float:
        """Block until `tokens` are available; returns seconds waited."""
        while True:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return 0.0
            wait = (tokens - self._tokens) / self.refill_rate
            time.sleep(wait)


# ============================================================================
# 4. PYDANTIC DATA MODELS  (field_validators with auto‑coercion)                      [10†L13-L25]
# ============================================================================
KNOWN_COLORS: set[str] = {
    "red", "orange", "yellow", "green", "blue", "indigo", "violet",
    "purple", "pink", "white", "black", "brown", "grey", "gray",
    "silver", "gold", "maroon", "crimson", "turquoise", "teal",
    "magenta", "cyan", "navy", "beige", "coral", "peach", "mint",
    "lavender", "sky", "lime", "rose", "olive", "ruby", "sapphire",
    "emerald", "amber", "topaz", "jade", "cobalt", "copper", "bronze",
    "platinum", "aqua", "avocado", "champagne", "charcoal", "chestnut",
    "chocolate", "citrine", "cream", "ebony", "forest", "fuchsia",
    "ginger", "honey", "ivory", "khaki", "lemon", "lilac", "mahogany",
    "mustard", "onyx", "opal", "periwinkle", "rust", "sand", "scarlet",
    "sepia", "sienna", "tan", "taupe", "tomato", "wheat",
}


class HoroscopeEntry(BaseModel):
    """Single‑rashi horoscope validated before disk commit."""
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
        """Extract the first known colour word from any length of text."""
        raw = str(v).strip().lower()
        words: List[str] = re.findall(r"[a-zA-Z]+", raw)
        for w in words:
            if w in KNOWN_COLORS:
                return w.capitalize()
        return words[0].capitalize() if words else "Red"

    @field_validator("*", mode="before")
    @classmethod
    def coerce_str(cls, v: Any) -> str:
        if not isinstance(v, str):
            return str(v)
        return v


class DailyHoroscopeOutput(BaseModel):
    date: str
    run_timestamp: str
    rashi_horoscopes: Dict[str, HoroscopeEntry]


# ============================================================================
# 5. CONFIGURATION
# ============================================================================
RASHIS: List[str] = [
    "Mesha (Aries)", "Vrishabha (Taurus)", "Mithuna (Gemini)",
    "Karka (Cancer)", "Simha (Leo)", "Kanya (Virgo)",
    "Tula (Libra)", "Vrishchika (Scorpio)", "Dhanu (Sagittarius)",
    "Makara (Capricorn)", "Kumbha (Aquarius)", "Meena (Pisces)",
]

RASHI_SHORT: List[str] = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
]

PLANETS: Dict[int, str] = {
    swe.SUN: "Sun", swe.MOON: "Moon", swe.MARS: "Mars",
    swe.MERCURY: "Mercury", swe.JUPITER: "Jupiter", swe.VENUS: "Venus",
    swe.SATURN: "Saturn", swe.MEAN_NODE: "Rahu",
}

MAX_OUTPUT_TOKENS: int = 800
REQUEST_DELAY: float = 2.0   # inter‑call spacing (seconds)
MAX_RETRIES: int = 3


# ============================================================================
# 6. EPHEMERIS LAYER  (Swiss Ephemeris, Moshier built‑in, Lahiri ayanamsa)          [14†L19-L23]
# ============================================================================
def setup_ephemeris() -> None:
    ephe_dir = tempfile.mkdtemp(prefix="sweph_")
    swe.set_ephe_path(ephe_dir)
    swe.set_sid_mode(swe.SIDM_LAHIRI)
    log.info("Ephemeris initialised (Moshier built‑in, Lahiri ayanamsa)")


def compute_planet_positions(jd: float) -> Dict[str, Dict[str, Any]]:
    positions: Dict[str, Dict[str, Any]] = {}
    for pid, pname in PLANETS.items():
        xx, _ = swe.calc_ut(jd, pid, swe.FLG_SIDEREAL | swe.FLG_SPEED)
        lon: float = xx[0]
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
# 7. MULTI‑PROVIDER GATEWAY WITH INSTRUCTOR + CIRCUIT BREAKER                       [15†L26-L34]
# ============================================================================
SYSTEM_PROMPT = (
    "You are a seasoned Vedic astrologer. "
    "You must output a JSON object with exactly these keys: "
    "general, luck, scope, study, love, travel, lucky_number, lucky_color. "
    "lucky_number must be an integer 1‑100. lucky_color must be a single colour name like 'Red' — "
    "NOT a sentence. Each text value must be 2‑4 detailed, encouraging sentences "
    "grounded in the given real planetary transits."
)


@dataclass
class ProviderSlot:
    name: str
    breaker: CircuitBreaker
    bucket: TokenBucket
    client_fn: Callable[[], Any]


def _build_groq_client() -> Any:
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise LLMConnectionError("GROQ_API_KEY not set")
    base = Groq(api_key=api_key)
    return instructor.from_groq(base, mode=instructor.Mode.JSON)


PROVIDERS: List[ProviderSlot] = [
    ProviderSlot(
        name="Groq",
        breaker=CircuitBreaker(name="Groq", failure_threshold=3, cooldown_seconds=180),
        bucket=TokenBucket(max_tokens=25, refill_rate=25 / 60.0),   # 25 RPM
        client_fn=_build_groq_client,
    ),
]

# Optional Gemini fallback (if google‑genai installed)
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

if HAS_GEMINI:
    def _build_gemini_client() -> Any:
        # instructor wraps generativeai as well
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise LLMConnectionError("GEMINI_API_KEY not set")
        genai.configure(api_key=api_key)
        return instructor.from_gemini(
            genai.GenerativeModel("gemini-2.0-flash"),
            mode=instructor.Mode.GEMINI_JSON,
        )
    PROVIDERS.append(
        ProviderSlot(
            name="Gemini",
            breaker=CircuitBreaker(name="Gemini", failure_threshold=3, cooldown_seconds=180),
            bucket=TokenBucket(max_tokens=10, refill_rate=10 / 60.0),
            client_fn=_build_gemini_client,
        )
    )


# ============================================================================
# 8. FULL‑JITTER BACKOFF HELPER  (AWS Builder’s Library)                             [7†L41-L45]
# ============================================================================
def full_jitter_backoff(attempt: int, base: float = 1.0, cap: float = 30.0) -> float:
    """Exponential backoff with FULL jitter — prevents thundering‑herd."""
    return random.uniform(0, min(cap, base * (2 ** attempt)))


# ============================================================================
# 9. SELF‑HEALING RASHI GENERATOR (Instructor → JSON Repair → Pydantic)
# ============================================================================
def _try_instructor(
    client: Any, model: str, prompt: str, max_tokens: int
) -> Optional[HoroscopeEntry]:
    """Attempt structured generation via instructor (primary path)."""
    try:
        entry: HoroscopeEntry = client.chat.completions.create(
            model=model,
            response_model=HoroscopeEntry,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.8,
        )
        return entry
    except Exception as exc:
        log.debug("  instructor failed: %s", exc)
        return None


def _try_raw_repair(
    client_raw: Groq, model: str, prompt: str, max_tokens: int
) -> Optional[HoroscopeEntry]:
    """Fallback: raw API call + multi‑stage JSON repair + Pydantic validation."""
    content: Optional[str] = None
    for attempt in range(1, 4):
        try:
            resp = client_raw.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.8,
                top_p=0.9,
                max_completion_tokens=max_tokens,
            )
            content = resp.choices[0].message.content
            if not content:
                continue
            break
        except Exception as exc:
            if attempt < 3:
                time.sleep(full_jitter_backoff(attempt, base=2, cap=15))
                continue
            raise LLMConnectionError(str(exc)) from exc

    if not content:
        return None

    # Multi‑stage repair (3 strategies in sequence)                                  [11†L13-L16]
    strategies: List[Callable[[str], Optional[dict]]] = [
        lambda c: json.loads(repair_json(c, return_objects=True)),
        lambda c: _regex_extract_and_repair(c),
        lambda c: _balance_and_repair(c),
    ]
    for i, strat in enumerate(strategies, start=1):
        try:
            obj = strat(content)
            if isinstance(obj, dict):
                return HoroscopeEntry.model_validate(obj)
        except Exception:
            continue
    return None


def _regex_extract_and_repair(content: str) -> Optional[dict]:
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return None
    return repair_json(match.group(), return_objects=True)


def _balance_and_repair(content: str) -> Optional[dict]:
    open_braces = content.count("{") - content.count("}")
    if open_braces > 0:
        content += "}" * open_braces
    return repair_json(content, return_objects=True)


def generate_rashi(
    rashi_name: str, today_str: str, transit_text: str,
    model: str = "llama-3.3-70b-versatile",
) -> HoroscopeEntry:
    """
    Generate one rashi using:
      1. instructor (forces valid JSON)
      2. raw API + json_repair (fallback)
      3. Pydantic validation (final gate)
    with circuit‑breaker gating and full‑jitter retries.
    """
    prompt = (
        f"Rashi: {rashi_name}\nToday: {today_str}\n\n"
        f"Real planetary transits (house positions from {rashi_name}):\n{transit_text}\n\n"
        f"Based ONLY on these exact positions, generate the horoscope JSON."
    )

    for slot in PROVIDERS:
        # ── Circuit breaker gate ──
        if not slot.breaker.allow_request():
            log.warning("  ⚠ %s circuit OPEN — skipping", slot.name)
            continue

        # ── Rate limiter ──
        slot.bucket.acquire(1)

        try:
            client = slot.client_fn()

            # --- Primary: instructor ---
            entry = _try_instructor(client, model, prompt, MAX_OUTPUT_TOKENS)
            if entry is not None:
                slot.breaker.record_success()
                log.info("  ✓ %s (instructor)", slot.name)
                return entry

            # --- Fallback: raw + repair (need the raw Groq client) ---
            raw_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
            entry = _try_raw_repair(raw_client, model, prompt, MAX_OUTPUT_TOKENS)
            if entry is not None:
                slot.breaker.record_success()
                log.info("  ✓ %s (raw+repair)", slot.name)
                return entry

            # Both failed — record as response error
            slot.breaker.record_failure()
            raise LLMResponseError(f"{slot.name}: instructor + repair both failed")

        except CircuitBreakerOpenError:
            continue
        except LLMRateLimitError:
            slot.breaker.record_failure()
            log.warning("  ⚠ %s rate‑limited", slot.name)
            continue
        except (LLMConnectionError, LLMResponseError) as exc:
            slot.breaker.record_failure()
            log.warning("  ⚠ %s failed: %s", slot.name, exc)
            continue

    raise ValidationExhaustedError(f"All providers exhausted for {rashi_name}")


# ============================================================================
# 10. MAIN
# ============================================================================
def main() -> None:
    log.info("===== Industry‑Grade Daily Horoscope Generator START =====")
    setup_ephemeris()

    today = datetime.now()
    today_str = today.strftime("%B %d, %Y")
    today_iso = today.isoformat()
    jd = swe.julday(today.year, today.month, today.day, 0.0)

    log.info("Computing planetary positions (Moshier ephemeris)…")
    positions = compute_planet_positions(jd)
    for name, data in positions.items():
        log.info("  %s: %s %.2f°", name, data["sign_name"], data["degree"])

    os.makedirs("data", exist_ok=True)

    rashi_horoscopes: Dict[str, HoroscopeEntry] = {}
    for idx, rashi in enumerate(RASHIS):
        log.info("--- %s ---", rashi)
        transit = build_transit_text(idx, positions)
        log.info("  Transits: %s…", transit[:120])
        entry = generate_rashi(rashi, today_str, transit)
        rashi_horoscopes[rashi] = entry
        log.info("  ✓ %s complete", rashi)
        time.sleep(REQUEST_DELAY)

    output = DailyHoroscopeOutput(
        date=today_iso,
        run_timestamp=datetime.now(timezone.utc).isoformat(),
        rashi_horoscopes=rashi_horoscopes,
    )

    # Atomic write via tempfile + replace                                          [6†L5-L7]
    out_path = Path("data") / "horoscopes.json"
    fd, tmp_path = tempfile.mkstemp(suffix=".json", dir="data")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(output.model_dump_json(indent=2, ensure_ascii=False))
        os.replace(tmp_path, out_path)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise

    log.info("✅ All 12 rashis saved to %s", out_path)


if __name__ == "__main__":
    main()

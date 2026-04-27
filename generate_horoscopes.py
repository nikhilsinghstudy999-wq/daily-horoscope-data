#!/usr/bin/env python3
"""
High‑Tech Daily Vedic Horoscope Generator — Multi‑Provider, Circuit‑Breaker, Pydantic‑Validated.

Architecture:
  ┌──────────────────────────────────────────────────────────┐
  │  1. Ephemeris Layer (Moshier, no external files)         │
  │  2. Multi‑Provider LLM Gateway (Groq → Gemini → Router)  │
  │  3. Circuit Breaker per provider (prevents retry storms) │
  │  4. Token‑Bucket Rate Limiter (never hit 429)            │
  │  5. Validator Sandwich (Pydantic models + json_repair)   │
  │  6. Atomic file writes                                   │
  │  7. Structured logging with cost tracking                │
  └──────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import hashlib
import json
import logging
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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("horoscope")


# ============================================================================
# 0. DEPENDENCY CHECKS (fail‑fast with clear messages)
# ============================================================================
def _check_deps():
    missing = []
    for mod, pkg in [
        ("swisseph", "pyswisseph"),
        ("groq", "groq"),
        ("pydantic", "pydantic"),
        ("json_repair", "json_repair"),
        ("tenacity", "tenacity"),
        ("google.generativeai", "google-generativeai"),
    ]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(f"  pip install {pkg}")
    if missing:
        log.critical("Missing dependencies:\n%s", "\n".join(missing))
        sys.exit(1)

_check_deps()

import swisseph as swe
from groq import Groq
from pydantic import BaseModel, Field, ValidationError, field_validator
from json_repair import repair_json
from tenacity import (
    retry, stop_after_attempt, wait_exponential, retry_if_exception_type,
    before_sleep_log,
)

# Optional: Google Gemini as fallback provider
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False


# ============================================================================
# 1. CUSTOM EXCEPTION HIERARCHY
# ============================================================================
class HoroscopeError(Exception):
    """Base exception for the horoscope pipeline."""

class EphemerisError(HoroscopeError):
    """Planetary calculation failure."""

class LLMConnectionError(HoroscopeError):
    """Network-level failure to reach an LLM provider."""

class LLMRateLimitError(HoroscopeError):
    """HTTP 429 from any provider."""

class LLMResponseError(HoroscopeError):
    """Provider returned a response but it was unusable."""

class CircuitBreakerOpenError(HoroscopeError):
    """Circuit breaker is open — provider skipped."""

class ValidationError_(HoroscopeError):
    """Pydantic validation failed after all repair attempts."""


# ============================================================================
# 2. CIRCUIT BREAKER (prevents retry storms on degraded providers)
# ============================================================================
class CircuitState(Enum):
    CLOSED = auto()          # requests pass through
    OPEN = auto()            # requests immediately fail
    HALF_OPEN = auto()       # one probe request allowed


@dataclass
class CircuitBreaker:
    """
    Tracks failures per provider.  After `failure_threshold` consecutive
    failures, the circuit opens for `cooldown_seconds`.  In HALF_OPEN state,
    the next call is a probe — if it succeeds the circuit closes again.
    """
    name: str
    failure_threshold: int = 3
    cooldown_seconds: float = 120.0
    _state: CircuitState = CircuitState.CLOSED
    _failure_count: int = 0
    _last_failure_time: float = 0.0

    def record_failure(self):
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            log.warning("🔴 Circuit OPEN for %s (%d failures)", self.name, self._failure_count)

    def record_success(self):
        self._failure_count = 0
        self._state = CircuitState.CLOSED

    def allow_request(self) -> bool:
        if self._state == CircuitState.CLOSED:
            return True
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._last_failure_time >= self.cooldown_seconds:
                self._state = CircuitState.HALF_OPEN
                log.info("🟡 Circuit HALF_OPEN for %s — probing", self.name)
                return True
            return False
        # HALF_OPEN — allow one probe
        return True


# ============================================================================
# 3. TOKEN‑BUCKET RATE LIMITER
# ============================================================================
@dataclass
class TokenBucket:
    """Simple token‑bucket rate limiter to never hit provider 429s."""
    max_tokens: float
    refill_rate: float  # tokens per second
    _tokens: float = field(init=False)

    def __post_init__(self):
        self._tokens = self.max_tokens
        self._last_refill = time.monotonic()

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.max_tokens, self._tokens + elapsed * self.refill_rate)
        self._last_refill = now

    def consume(self, tokens: float = 1.0) -> float:
        """Block until `tokens` are available; returns wait time."""
        while True:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return 0.0
            wait = (tokens - self._tokens) / self.refill_rate
            time.sleep(wait)


# ============================================================================
# 4. PYDANTIC DATA MODELS (Validator Sandwich – inner layer)
# ============================================================================
class HoroscopeEntry(BaseModel):
    """A single rashi's horoscope, validated before writing to disk."""
    general: str = Field(..., min_length=20, max_length=2000)
    luck: str = Field(..., min_length=10, max_length=2000)
    scope: str = Field(..., min_length=10, max_length=2000)
    study: str = Field(..., min_length=10, max_length=2000)
    love: str = Field(..., min_length=10, max_length=2000)
    travel: str = Field(..., min_length=10, max_length=2000)
    lucky_number: int = Field(..., ge=1, le=100)
    lucky_color: str = Field(..., min_length=2, max_length=30)

    @field_validator("lucky_color")
    @classmethod
    def clean_color(cls, v: str) -> str:
        return v.strip().capitalize()

    @field_validator("*", mode="before")
    @classmethod
    def coerce_strings(cls, v: Any) -> str:
        if not isinstance(v, str):
            return str(v)
        return v


class DailyHoroscopeOutput(BaseModel):
    """Top‑level output model."""
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

# ---------------------------------------------------------------------------
# Multi‑provider definitions
# ---------------------------------------------------------------------------
@dataclass
class ProviderConfig:
    name: str
    circuit_breaker: CircuitBreaker
    rate_limiter: TokenBucket


PROVIDERS = {
    "groq": ProviderConfig(
        name="Groq",
        circuit_breaker=CircuitBreaker(name="Groq", failure_threshold=3, cooldown_seconds=180),
        rate_limiter=TokenBucket(max_tokens=25, refill_rate=25 / 60.0),  # 25 req/min
    ),
}

if HAS_GEMINI:
    PROVIDERS["gemini"] = ProviderConfig(
        name="Gemini",
        circuit_breaker=CircuitBreaker(name="Gemini", failure_threshold=3, cooldown_seconds=180),
        rate_limiter=TokenBucket(max_tokens=10, refill_rate=10 / 60.0),
    )

SYSTEM_PROMPT = (
    "You are a seasoned Vedic astrologer. "
    "Reply with exactly ONE JSON object, no markdown, no extra text. "
    "Keys: general, luck, scope, study, love, travel, lucky_number, lucky_color. "
    "All values are strings except lucky_number (integer 1‑100). "
    "Each text value: 2‑4 detailed, encouraging sentences grounded in the given real transits."
)


# ============================================================================
# 6. EPHEMERIS LAYER
# ============================================================================
def setup_ephemeris() -> None:
    ephe_dir = tempfile.mkdtemp(prefix="sweph_")
    swe.set_ephe_path(ephe_dir)
    swe.set_sid_mode(swe.SIDM_LAHIRI)
    log.info("Ephemeris initialised (Moshier, Lahiri)")


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
# 7. MULTI‑PROVIDER LLM GATEWAY WITH CIRCUIT BREAKER
# ============================================================================
def _call_groq(prompt: str, max_tokens: int) -> str:
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        temperature=0.8, top_p=0.9, max_completion_tokens=max_tokens,
    )
    content = response.choices[0].message.content
    if not content:
        raise LLMResponseError("Groq returned empty content")
    return content


def _call_gemini(prompt: str, max_tokens: int) -> str:
    if not HAS_GEMINI:
        raise LLMConnectionError("Gemini SDK not installed")
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(
        [SYSTEM_PROMPT, prompt],
        generation_config={"temperature": 0.8, "top_p": 0.9, "max_output_tokens": max_tokens},
    )
    if not response.text:
        raise LLMResponseError("Gemini returned empty content")
    return response.text


PROVIDER_CALLABLES: Dict[str, Callable] = {"groq": _call_groq}
if HAS_GEMINI:
    PROVIDER_CALLABLES["gemini"] = _call_gemini


def call_with_circuit_breaker(
    provider_name: str,
    prompt: str,
    max_tokens: int = 800,
) -> str:
    """
    Call a single provider through its circuit breaker and rate limiter.
    Raises CircuitBreakerOpenError, LLMRateLimitError, LLMConnectionError, or LLMResponseError.
    """
    cfg = PROVIDERS.get(provider_name)
    if not cfg:
        raise LLMConnectionError(f"Unknown provider: {provider_name}")

    # Circuit breaker check
    if not cfg.circuit_breaker.allow_request():
        raise CircuitBreakerOpenError(f"Circuit breaker is OPEN for {provider_name}")

    # Rate limiter
    cfg.rate_limiter.consume(1)

    try:
        result = PROVIDER_CALLABLES[provider_name](prompt, max_tokens)
        cfg.circuit_breaker.record_success()
        return result
    except CircuitBreakerOpenError:
        raise
    except (LLMRateLimitError, LLMConnectionError, LLMResponseError):
        cfg.circuit_breaker.record_failure()
        raise
    except Exception as exc:
        cfg.circuit_breaker.record_failure()
        msg = str(exc).lower()
        if "rate limit" in msg or "429" in msg:
            raise LLMRateLimitError(str(exc)) from exc
        if "timeout" in msg or "connection" in msg or "503" in msg or "500" in msg:
            raise LLMConnectionError(str(exc)) from exc
        raise LLMResponseError(str(exc)) from exc


def generate_raw_horoscope(prompt: str) -> str:
    """
    Walk the provider priority list.  Groq first, then Gemini, then fail hard.
    """
    last_error: Optional[Exception] = None
    for pname in PROVIDER_CALLABLES:
        try:
            log.info("  → trying %s", pname)
            return call_with_circuit_breaker(pname, prompt)
        except CircuitBreakerOpenError:
            log.warning("  ⚠ %s circuit open, skipping", pname)
            continue
        except (LLMRateLimitError, LLMConnectionError, LLMResponseError) as exc:
            log.warning("  ⚠ %s failed: %s", pname, exc)
            last_error = exc
            time.sleep(1)
            continue
        except Exception as exc:
            log.error("  ❌ Unexpected error from %s: %s", pname, exc)
            last_error = exc
            continue
    raise LLMConnectionError(f"All providers exhausted. Last error: {last_error}")


# ============================================================================
# 8. VALIDATOR SANDWICH – JSON repair + Pydantic
# ============================================================================
def parse_and_validate(raw: str) -> HoroscopeEntry:
    """
    Multi‑stage JSON repair followed by Pydantic validation.
    The 'Validator Sandwich' pattern: AI output → repair → validate → retry.
    """
    if not raw or not raw.strip():
        raise ValidationError_("Empty LLM response")

    content = raw.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)

    errors = []
    # Stage 1: Direct json_repair
    try:
        obj = repair_json(content, return_objects=True)
        return HoroscopeEntry.model_validate(obj)
    except Exception as e:
        errors.append(f"stage1: {e}")

    # Stage 2: Regex extract + repair
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        try:
            obj = repair_json(match.group(), return_objects=True)
            return HoroscopeEntry.model_validate(obj)
        except Exception as e:
            errors.append(f"stage2: {e}")

    # Stage 3: Brace balancing + repair
    balanced = _balance_braces(content)
    try:
        obj = repair_json(balanced, return_objects=True)
        return HoroscopeEntry.model_validate(obj)
    except Exception as e:
        errors.append(f"stage3: {e}")

    raise ValidationError_(f"All JSON repair stages failed: {'; '.join(errors)}")


def _balance_braces(text: str) -> str:
    open_count = text.count("{") - text.count("}")
    if open_count > 0:
        return text + "}" * open_count
    return text.rstrip("}")[: -abs(open_count)] if open_count < 0 else text


# ============================================================================
# 9. SELF‑HEALING RASHI GENERATOR
# ============================================================================
def generate_rashi(rashi_name: str, today_str: str, transit_text: str, max_retries: int = 3) -> HoroscopeEntry:
    """Generate and validate one rashi, with self‑healing retries on validation failure."""
    prompt = (
        f"Rashi: {rashi_name}\nToday: {today_str}\n\n"
        f"Real planetary transits (house positions from {rashi_name}):\n{transit_text}\n\n"
        f"Based ONLY on these exact positions, generate the horoscope JSON."
    )
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            raw = generate_raw_horoscope(prompt)
            return parse_and_validate(raw)
        except (ValidationError_, ValidationError) as exc:
            last_error = exc
            log.warning("  ⚠ validation attempt %d/%d: %s", attempt, max_retries, exc)
            if attempt < max_retries:
                time.sleep(2 ** attempt)
        except (LLMConnectionError, LLMRateLimitError, LLMResponseError) as exc:
            last_error = exc
            log.error("  ❌ LLM error attempt %d/%d: %s", attempt, max_retries, exc)
            if attempt < max_retries:
                time.sleep(4)
    raise HoroscopeError(f"Failed after {max_retries} attempts for {rashi_name}: {last_error}")


# ============================================================================
# 10. MAIN
# ============================================================================
def main() -> None:
    log.info("===== High‑Tech Daily Horoscope Generator START =====")
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
        log.info("  ✓ %s generated", rashi)
        time.sleep(1.5)  # gentle spacing

    output = DailyHoroscopeOutput(
        date=today_iso,
        run_timestamp=datetime.now(timezone.utc).isoformat(),
        rashi_horoscopes=rashi_horoscopes,
    )

    # Atomic write
    out_path = Path("data") / "horoscopes.json"
    tmp_path = out_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(output.model_dump_json(indent=2, ensure_ascii=False))
    tmp_path.replace(out_path)
    log.info("✅ All 12 rashis saved to %s", out_path)


if __name__ == "__main__":
    main()

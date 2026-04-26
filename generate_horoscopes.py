#!/usr/bin/env python3
"""
Fast local horoscope generation using transformers + optional HF Inference API fallback.
"""
import json
import os
import time
from datetime import datetime

from transformers import pipeline, set_seed

# ---------- CONFIGURATION ----------
RASHIS = [
    "Mesha (Aries)", "Vrishabha (Taurus)", "Mithuna (Gemini)",
    "Karka (Cancer)", "Simha (Leo)", "Kanya (Virgo)",
    "Tula (Libra)", "Vrishchika (Scorpio)", "Dhanu (Sagittarius)",
    "Makara (Capricorn)", "Kumbha (Aquarius)", "Meena (Pisces)"
]

# HuggingFace Inference API (fallback only)
HF_TOKEN = os.environ.get("HF_TOKEN")
PRIMARY_MODEL_URL = "https://api-inference.huggingface.co/models/shahp7575/gpt2-horoscopes"

MIN_LENGTH = 100

# ---------- HELPERS ----------
def generate_locally(model_name, rashi_prompts):
    """
    Load the model locally (cached on disk) and generate horoscopes for all rashis.
    Returns dictionary {rashi: text} or None if loading fails.
    """
    print(f"Loading model {model_name} locally...")
    try:
        generator = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
            device=-1,           # CPU
            framework="pt"
        )
        # Warm-up dummy call (optional)
        _ = generator("test", max_new_tokens=5)
        print("Model loaded. Generating...")
        results = {}
        for rashi, prompt in rashi_prompts.items():
            output = generator(
                prompt,
                max_new_tokens=250,
                temperature=0.8,
                do_sample=True,
                pad_token_id=generator.tokenizer.eos_token_id
            )[0]["generated_text"]
            # Remove the prompt from the output if it includes it
            if output.startswith(prompt):
                horoscope = output[len(prompt):].strip()
            else:
                horoscope = output.strip()
            results[rashi] = horoscope
        return results
    except Exception as e:
        print(f"Local generation failed: {e}")
        return None

def call_inference_api(prompt, retries=2):
    """Fallback: call HF Inference API."""
    if not HF_TOKEN:
        return None
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "x-wait-for-model": "true"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 250, "temperature": 0.8, "return_full_text": False}
    }
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(PRIMARY_MODEL_URL, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                result = resp.json()
                if isinstance(result, list) and "generated_text" in result[0]:
                    return result[0]["generated_text"].strip()
        except Exception:
            pass
        time.sleep(5)
    return None

def quality_check(text):
    if not text or len(text) < MIN_LENGTH:
        return False
    bad = ["unable to generate", "i cannot", "as an ai", "error", "loading"]
    lower = text.lower()
    return not any(b in lower for b in bad)

def load_previous_horoscopes():
    try:
        with open("data/horoscopes.json", "r", encoding="utf-8") as f:
            return json.load(f).get("rashi_horoscopes", {})
    except Exception:
        pass
    try:
        with open("data/fallback_horoscopes.json", "r", encoding="utf-8") as f:
            return json.load(f).get("rashi_horoscopes", {})
    except Exception:
        pass
    return {r: "Horoscope will be available soon." for r in RASHIS}

# ---------- MAIN ----------
def main():
    today_str = datetime.now().strftime("%B %d, %Y")
    today_iso = datetime.now().isoformat()

    # Skip if today already done
    try:
        with open("data/horoscopes.json", "r", encoding="utf-8") as f:
            if json.load(f).get("date", "").startswith(today_iso[:10]):
                print("Today's horoscopes already exist. Exiting.")
                return
    except FileNotFoundError:
        pass

    # Build prompts for local generation
    prompts = {}
    for rashi in RASHIS:
        prompts[rashi] = (
            f"<|category|> general <|horoscope|> "
            f"Generate a detailed Vedic daily horoscope for {rashi}. "
            f"Today is {today_str}. Include predictions for career, love, "
            f"health, and a lucky tip. Keep it 150-200 words."
        )

    # ---- STEP 1: Try local generation (fast) ----
    output = {"date": today_iso, "rashi_horoscopes": {}}
    local_generated = generate_locally("shahp7575/gpt2-horoscopes", prompts)

    for rashi in RASHIS:
        if local_generated and quality_check(local_generated.get(rashi)):
            output["rashi_horoscopes"][rashi] = local_generated[rashi]
        else:
            # ---- STEP 2: Fallback to Inference API ----
            print(f"Local generation insufficient for {rashi}, trying Inference API...")
            horoscope = call_inference_api(prompts[rashi])
            if quality_check(horoscope):
                output["rashi_horoscopes"][rashi] = horoscope
            else:
                # ---- STEP 3: Use previous day ----
                print(f"Both methods failed for {rashi}, reusing previous horoscope.")
                previous = load_previous_horoscopes()
                output["rashi_horoscopes"][rashi] = previous.get(
                    rashi, f"{rashi} horoscope is being updated. Check back shortly."
                )

    with open("data/horoscopes.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print("Done.")

if __name__ == "__main__":
    main()

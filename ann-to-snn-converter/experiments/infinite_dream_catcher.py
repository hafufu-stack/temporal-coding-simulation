"""
Infinite Dream Catcher — Mass Vaccine Production
==================================================
Generate 1000+ hallucination vaccine samples automatically.
Uses a large pool of random factual questions.
Runs in a loop, saving JSONL incrementally.
"""

import torch
import numpy as np
import json
import os
import time
import random
from transformers import AutoModelForCausalLM, AutoTokenizer

RESULTS_DIR = "results_v10_canary"
os.makedirs(RESULTS_DIR, exist_ok=True)
OUTPUT_JSONL = os.path.join(RESULTS_DIR, "vaccine_1000.jsonl")

CANARY_LAYER = 10
CANARY_HEAD = 17
NOISE_SIGMA = 0.10
NIGHTMARE_THRESHOLD = 3.0
SURGICAL_COT = " Wait, let me think about this carefully. The correct answer is:"

# Large pool of factual questions with expected keywords
QUESTION_BANK = [
    ("What is the capital of France?", "paris"),
    ("What is 2 + 2?", "4"),
    ("Who wrote Romeo and Juliet?", "shakespeare"),
    ("What is the boiling point of water in Celsius?", "100"),
    ("What planet is closest to the Sun?", "mercury"),
    ("What is the largest mammal?", "whale"),
    ("How many continents are there?", "7"),
    ("What color is the sky on a clear day?", "blue"),
    ("What gas do plants absorb?", "carbon dioxide"),
    ("What is the speed of light in km/s?", "300"),
    ("What is the capital of Japan?", "tokyo"),
    ("What is the chemical symbol for water?", "h2o"),
    ("How many legs does a spider have?", "8"),
    ("What is the largest planet in our solar system?", "jupiter"),
    ("What is the freezing point of water in Celsius?", "0"),
    ("Who painted the Mona Lisa?", "vinci"),
    ("What is the capital of Germany?", "berlin"),
    ("How many days are in a year?", "365"),
    ("What is the tallest mountain on Earth?", "everest"),
    ("What ocean is the largest?", "pacific"),
    ("What is the capital of Italy?", "rome"),
    ("How many bones are in the human body?", "206"),
    ("What is the chemical symbol for gold?", "au"),
    ("What is the capital of Australia?", "canberra"),
    ("How many planets are in our solar system?", "8"),
    ("What is the smallest country in the world?", "vatican"),
    ("Who invented the telephone?", "bell"),
    ("What is the capital of Brazil?", "brasilia"),
    ("How many hours are in a day?", "24"),
    ("What is the hardest natural substance?", "diamond"),
    ("What is the capital of Canada?", "ottawa"),
    ("How many sides does a hexagon have?", "6"),
    ("What is the capital of Spain?", "madrid"),
    ("What element does O represent?", "oxygen"),
    ("How many weeks are in a year?", "52"),
    ("What is the largest bird in the world?", "ostrich"),
    ("What is the capital of Russia?", "moscow"),
    ("How many teeth does an adult human have?", "32"),
    ("What is the fastest land animal?", "cheetah"),
    ("What is the capital of China?", "beijing"),
    ("How many chromosomes do humans have?", "46"),
    ("What is the longest river in the world?", "nile"),
    ("What is the capital of India?", "delhi"),
    ("What is the chemical formula for salt?", "nacl"),
    ("What is the capital of Egypt?", "cairo"),
    ("How many minutes are in an hour?", "60"),
    ("What is the largest desert in the world?", "sahara"),
    ("What is the capital of Mexico?", "mexico city"),
    ("How many seconds are in a minute?", "60"),
    ("What metal is liquid at room temperature?", "mercury"),
    ("What is the capital of South Korea?", "seoul"),
    ("How many strings does a standard guitar have?", "6"),
    ("What is the smallest bone in the human body?", "stapes"),
    ("What is the capital of Argentina?", "buenos aires"),
    ("What gas makes up most of Earth's atmosphere?", "nitrogen"),
    ("What is the capital of Turkey?", "ankara"),
    ("What is the largest organ in the human body?", "skin"),
    ("What is the capital of Thailand?", "bangkok"),
    ("How many players are on a soccer team?", "11"),
    ("What is the capital of Indonesia?", "jakarta"),
    ("What is the chemical symbol for iron?", "fe"),
    ("What is the capital of Poland?", "warsaw"),
    ("How many continents does the equator cross?", "3"),
    ("What is the capital of Netherlands?", "amsterdam"),
    ("What is the hardest mineral on the Mohs scale?", "diamond"),
    ("What is the capital of Sweden?", "stockholm"),
    ("How many chambers does the human heart have?", "4"),
    ("What is the capital of Norway?", "oslo"),
    ("What planet is known as the Red Planet?", "mars"),
    ("What is the capital of Greece?", "athens"),
    ("What is the largest continent?", "asia"),
    ("What is the capital of Portugal?", "lisbon"),
    ("What is the currency of Japan?", "yen"),
    ("What is the capital of Austria?", "vienna"),
    ("How many colors are in a rainbow?", "7"),
    ("What is the capital of Switzerland?", "bern"),
    ("What is the deepest ocean?", "pacific"),
    ("What is the capital of Finland?", "helsinki"),
    ("What gas do we breathe in?", "oxygen"),
    ("What is the capital of Denmark?", "copenhagen"),
    ("How many sides does a triangle have?", "3"),
    ("What is the capital of Belgium?", "brussels"),
    ("What is the currency of the UK?", "pound"),
    ("What is the capital of Ireland?", "dublin"),
    ("What planet has the most moons?", "saturn"),
    ("What is the capital of Scotland?", "edinburgh"),
    ("How many months have 31 days?", "7"),
    ("What is the capital of New Zealand?", "wellington"),
    ("What is the chemical symbol for silver?", "ag"),
    ("What is the capital of Malaysia?", "kuala lumpur"),
    ("How many zeros are in a million?", "6"),
    ("What is the capital of Vietnam?", "hanoi"),
    ("What is the tallest animal?", "giraffe"),
    ("What is the capital of Peru?", "lima"),
    ("What is the largest fish in the world?", "whale shark"),
    ("What is the capital of Chile?", "santiago"),
    ("How many years are in a century?", "100"),
    ("What is the capital of Colombia?", "bogota"),
    ("What is the smallest planet in our solar system?", "mercury"),
    ("What is the capital of Nigeria?", "abuja"),
]

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def compute_canary_entropy(model, input_ids):
    with torch.no_grad():
        out = model(input_ids, output_attentions=True, use_cache=False)
    attn = out.attentions[CANARY_LAYER]
    a = attn[0, CANARY_HEAD, -1, :].float()
    a = torch.where(torch.isnan(a), torch.zeros_like(a), a)
    a = a.clamp(min=1e-10)
    h = -(a * torch.log2(a)).sum().item()
    if np.isnan(h) or np.isinf(h):
        h = 0.0
    del out
    return h


def make_noise_hook(std):
    def pre_hook(module, args):
        hidden_states = args[0]
        noise = torch.randn_like(hidden_states) * std
        return (hidden_states + noise,) + args[1:]
    return pre_hook


def run_mass_production():
    print("="*70)
    print("INFINITE DREAM CATCHER — Mass Vaccine Production")
    print(f"Target: 1000+ vaccine samples from {len(QUESTION_BANK)} questions")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "mistralai/Mistral-7B-v0.1"
    
    print(f"\nLoading {model_name}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16,
        attn_implementation="eager",
        device_map=device
    )
    model.eval()
    print(f"Loaded in {time.time()-t0:.1f}s")
    
    # Shuffle and repeat questions to reach 1000+
    all_questions = list(QUESTION_BANK) * (1000 // len(QUESTION_BANK) + 1)
    random.shuffle(all_questions)
    # Use first 334 (giving ~1000 samples: 334 clean + 334 nightmare + 334 healed)
    target_count = 334
    all_questions = all_questions[:target_count]
    
    total_saved = 0
    stats = {"clean": 0, "nightmare": 0, "healed": 0, 
             "nightmare_detected": 0, "healed_correct": 0}
    
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for qi, (question, expected_keyword) in enumerate(all_questions):
            prompt_base = f"Question: {question}\nAnswer:"
            inputs = tokenizer(prompt_base, return_tensors='pt', truncation=True, max_length=256)
            input_ids = inputs['input_ids'].to(device)
            
            # === CLEAN ===
            with torch.no_grad():
                clean_out = model.generate(
                    input_ids, max_new_tokens=30, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            clean_text = tokenizer.decode(clean_out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
            clean_entropy = compute_canary_entropy(model, input_ids)
            clean_correct = expected_keyword.lower() in clean_text.lower()
            
            clean_sample = {
                "type": "clean", "prompt": prompt_base,
                "response": clean_text[:200],
                "canary_entropy": round(float(clean_entropy), 4),
                "correct": bool(clean_correct),
                "label": "safe",
                "question": question,
            }
            f.write(json.dumps(clean_sample, ensure_ascii=False, cls=NumpyEncoder) + "\n")
            total_saved += 1
            stats["clean"] += 1
            
            # === NIGHTMARE ===
            hook = model.model.layers[CANARY_LAYER].register_forward_pre_hook(
                make_noise_hook(NOISE_SIGMA)
            )
            nightmare_entropy = compute_canary_entropy(model, input_ids)
            hook.remove()
            
            hook = model.model.layers[CANARY_LAYER].register_forward_pre_hook(
                make_noise_hook(NOISE_SIGMA)
            )
            with torch.no_grad():
                nm_out = model.generate(
                    input_ids, max_new_tokens=30, do_sample=True,
                    temperature=0.7, pad_token_id=tokenizer.eos_token_id
                )
            nm_text = tokenizer.decode(nm_out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
            hook.remove()
            
            nm_correct = expected_keyword.lower() in nm_text.lower()
            is_nightmare = nightmare_entropy > NIGHTMARE_THRESHOLD
            
            nm_text_safe = ''.join(c if ord(c) < 128 else '?' for c in nm_text[:200])
            
            nm_sample = {
                "type": "nightmare", "prompt": prompt_base,
                "response": nm_text_safe,
                "canary_entropy": round(float(nightmare_entropy), 4),
                "correct": bool(nm_correct),
                "noise_sigma": NOISE_SIGMA,
                "is_nightmare": bool(is_nightmare),
                "label": "hallucination",
                "question": question,
            }
            f.write(json.dumps(nm_sample, ensure_ascii=False, cls=NumpyEncoder) + "\n")
            total_saved += 1
            stats["nightmare"] += 1
            if is_nightmare:
                stats["nightmare_detected"] += 1
            
            # === HEAL ===
            if is_nightmare:
                prompt_healed = prompt_base + SURGICAL_COT
                h_inputs = tokenizer(prompt_healed, return_tensors='pt', truncation=True, max_length=256)
                h_ids = h_inputs['input_ids'].to(device)
                
                with torch.no_grad():
                    h_out = model.generate(
                        h_ids, max_new_tokens=30, do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                h_text = tokenizer.decode(h_out[0][h_ids.shape[1]:], skip_special_tokens=True).strip()
                h_entropy = compute_canary_entropy(model, h_ids)
                h_correct = expected_keyword.lower() in h_text.lower()
                
                h_sample = {
                    "type": "healed", "prompt": prompt_healed,
                    "response": h_text[:200],
                    "canary_entropy": round(float(h_entropy), 4),
                    "correct": bool(h_correct),
                    "healing_method": "surgical_cot",
                    "label": "recovered",
                    "question": question,
                }
                f.write(json.dumps(h_sample, ensure_ascii=False, cls=NumpyEncoder) + "\n")
                total_saved += 1
                stats["healed"] += 1
                if h_correct:
                    stats["healed_correct"] += 1
            
            torch.cuda.empty_cache()
            
            if (qi + 1) % 10 == 0:
                f.flush()
                elapsed = time.time() - t0
                rate = (qi + 1) / (elapsed - 17.5)  # subtract load time approx
                eta = (target_count - qi - 1) / rate if rate > 0 else 0
                print(f"  [{qi+1}/{target_count}] saved={total_saved} | "
                      f"nightmares={stats['nightmare_detected']} | "
                      f"healed={stats['healed_correct']}/{stats['healed']} | "
                      f"ETA: {eta/60:.1f}min")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"DREAM CATCHER MASS PRODUCTION COMPLETE!")
    print(f"{'='*70}")
    print(f"  Total samples: {total_saved}")
    print(f"  Clean: {stats['clean']}")
    print(f"  Nightmare: {stats['nightmare']} (detected: {stats['nightmare_detected']})")
    print(f"  Healed: {stats['healed']} (correct: {stats['healed_correct']})")
    if stats['healed'] > 0:
        heal_rate = stats['healed_correct'] / stats['healed'] * 100
        print(f"  Healing rate: {heal_rate:.1f}%")
    print(f"  File: {OUTPUT_JSONL}")
    print(f"  Size: {os.path.getsize(OUTPUT_JSONL) / 1024:.1f} KB")
    
    # Save stats
    stats_path = os.path.join(RESULTS_DIR, "vaccine_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats: {stats_path}")


if __name__ == '__main__':
    run_mass_production()

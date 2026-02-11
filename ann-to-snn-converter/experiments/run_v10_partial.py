"""Run v10 experiments: EXP2 (Llama Canary) + EXP3 (Electric Dreams) only."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

from metacognition_v10_dreams import (
    run_llama_canary, run_electric_dreams, plot_results
)
import json

print("="*60)
print("v10: Running EXP2 + EXP3 (Instruct model skipped)")
print("="*60)

results = {}
llama_diff = None
llama_result = None
dreams_result = None

# EXP 2: Llama Canary
try:
    llama_result, llama_diff = run_llama_canary()
    results["llama_canary"] = llama_result
    print(f"\nEXP 2 DONE: Ace = {llama_result['ace_canary']['label']}")
except Exception as e:
    print(f"EXP 2 FAILED: {e}")
    import traceback; traceback.print_exc()

# EXP 3: Electric Dreams
try:
    dreams_result = run_electric_dreams()
    results["electric_dreams"] = dreams_result
    nightmares = sum(1 for d in dreams_result['dreams'] if d['is_nightmare'])
    print(f"\nEXP 3 DONE: {nightmares} nightmares detected")
except Exception as e:
    print(f"EXP 3 FAILED: {e}")
    import traceback; traceback.print_exc()

# Save
os.makedirs("results_v10_canary", exist_ok=True)
with open("results_v10_canary/v10_partial_results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# Plot
try:
    plot_results(None, None, llama_result, llama_diff, dreams_result)
except Exception as e:
    print(f"Plot: {e}")
    import traceback; traceback.print_exc()

print("\nAll done!")

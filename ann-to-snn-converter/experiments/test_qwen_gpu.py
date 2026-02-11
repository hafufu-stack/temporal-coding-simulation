"""
Qwen2.5-1.5B GPU Re-verification
RTX 5080 (Blackwell) + attn_implementation="eager" test

Previous attempt: CUDA assertion crash.
Re-testing with safeguards.
"""
import sys
import torch
print("=" * 60)
print("  Qwen2.5-1.5B GPU Test (eager attention)")
print("=" * 60)
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B"
print(f"\n  Loading {MODEL} with attn_implementation='eager'...")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager"
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("  Model loaded OK!")
    print(f"  Layers: {model.config.num_hidden_layers}")
    print(f"  Heads: {model.config.num_attention_heads}")
    print(f"  KV Heads: {model.config.num_key_value_heads}")
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  VRAM used: {vram:.2f} GB")
except Exception as e:
    print(f"  LOAD FAILED: {e}")
    sys.exit(1)

# Test inference with output_attentions
print("\n  Testing inference with output_attentions=True...")
try:
    prompt = "Question: What is the capital of France?\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        out = model(**inputs, output_attentions=True, use_cache=False)
    
    print(f"  Output logits shape: {out.logits.shape}")
    print(f"  Attention layers: {len(out.attentions)}")
    attn_0 = out.attentions[0]
    print(f"  Attn[0] shape: {attn_0.shape}")
    
    # Compute entropy on attention
    import numpy as np
    a = attn_0[0, :, -1, :].float()  # (heads, src)
    a_log = torch.log2(a + 1e-10)
    head_ent = -(a * a_log).sum(dim=-1)
    print(f"  L0 head entropy: mean={head_ent.mean().item():.4f}, std={head_ent.std().item():.4f}")
    
    print("\n  ★ GPU TEST PASSED! Qwen2.5 works with eager attention on RTX 5080! ★")
    
    # Multi-step generation test
    print("\n  Testing multi-step generation (5 tokens)...")
    past = None
    input_ids = inputs["input_ids"]
    for step in range(5):
        with torch.no_grad():
            if past is None:
                out2 = model(input_ids, output_attentions=True, use_cache=True)
            else:
                out2 = model(input_ids[:, -1:], past_key_values=past,
                            output_attentions=True, use_cache=True)
            past = out2.past_key_values
        
        next_id = out2.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        token = tokenizer.decode(next_id[0])
        input_ids = torch.cat([input_ids, next_id], dim=1)
        print(f"    Step {step+1}: '{token}'")
        del out2.attentions
    
    print("\n  ★★ FULL GPU TEST PASSED! Multi-step generation OK! ★★")

except torch.cuda.OutOfMemoryError:
    print("  FAILED: CUDA OOM")
    sys.exit(1)
except RuntimeError as e:
    if "CUDA" in str(e) or "assert" in str(e).lower():
        print(f"  FAILED: CUDA assertion error: {e}")
        print("  This confirms the RTX 5080 Blackwell driver bug.")
    else:
        print(f"  FAILED: {e}")
    sys.exit(1)
except Exception as e:
    print(f"  FAILED: {e}")
    sys.exit(1)

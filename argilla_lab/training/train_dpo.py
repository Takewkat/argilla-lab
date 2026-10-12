"""DPO on out/<dataset>/dpo.jsonl with a LoRA adapter, plus a before/after probe.

The probe prints, for every pair, how much more likely the model finds the chosen
answer than the rejected one (log-probability margin). DPO pushes that number up.
It also generates an answer to one held-out prompt before and after training.
"""

import json
import sys

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from argilla_lab.core.paths import OUT_DIR

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET = sys.argv[1] if len(sys.argv) > 1 else "code-preference"
DATA = OUT_DIR / DATASET / "dpo.jsonl"
OUTPUT = OUT_DIR / DATASET / "dpo-adapter"
HELD_OUT = "Write a Python function that returns the median of a list of numbers. It must raise ValueError on an empty list."

pairs = [json.loads(line) for line in DATA.read_text().splitlines()]
dataset = Dataset.from_list([
    {
        "prompt": [{"role": "user", "content": p["prompt"]}],
        "chosen": [{"role": "assistant", "content": p["chosen"]}],
        "rejected": [{"role": "assistant", "content": p["rejected"]}],
    }
    for p in pairs
])

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
model.to("mps" if torch.backends.mps.is_available() else "cpu")


def answer_logprob(m, prompt, answer):
    prefix = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids
    full_ids = tokenizer(prefix + answer, return_tensors="pt").input_ids.to(m.device)
    with torch.no_grad():
        logits = m(full_ids).logits[0, :-1]
    targets = full_ids[0, 1:]
    logps = torch.log_softmax(logits.float(), dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1)
    return logps[prefix_ids.shape[1] - 1:].sum().item()


def margins(m):
    m.eval()
    return [answer_logprob(m, p["prompt"], p["chosen"]) - answer_logprob(m, p["prompt"], p["rejected"]) for p in pairs]


def generate(m, prompt):
    m.eval()
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, return_tensors="pt").to(m.device)
    with torch.no_grad():
        out = m.generate(**ids, max_new_tokens=200, do_sample=False)
    return tokenizer.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)


before = margins(model)
before_answer = generate(model, HELD_OUT)

config = DPOConfig(
    output_dir=str(OUTPUT),
    beta=0.1,
    learning_rate=5e-5,
    num_train_epochs=6,
    per_device_train_batch_size=2,
    max_length=512,
    logging_steps=5,
    save_strategy="no",
    bf16=False,
    report_to="none",
)
trainer = DPOTrainer(
    model=model,
    args=config,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"),
)
trainer.train()
trainer.save_model(str(OUTPUT))

after = margins(trainer.model)
after_answer = generate(trainer.model, HELD_OUT)

print("\nlog p(chosen) - log p(rejected), before -> after")
for p, b, a in zip(pairs, before, after):
    print(f"  {p['id']}  {b:+8.1f} -> {a:+8.1f}")
print(f"\nheld-out prompt: {HELD_OUT}")
print(f"\n--- before\n{before_answer}\n\n--- after\n{after_answer}")

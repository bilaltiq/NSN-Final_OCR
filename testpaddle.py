import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from cb_dataloader import CampbellDataset
from torch.utils.data import DataLoader
from metrics import cer, wer

# -----------------------------
# Custom collate so PIL images pass through
# -----------------------------
def collate_nopad(batch):
    return batch[0]   # batch_size = 1, simplest fix

# -----------------------------
# Dataset + Loader
# -----------------------------
dataset = CampbellDataset("Dataset/GT-pairs")
loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_nopad)

print("Total samples:", len(dataset))

# -----------------------------
# Model + Processor
# -----------------------------
model_name = "microsoft/Florence-2-base"

processor = AutoProcessor.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,      # <- fix the deprecation + ensures half precision
    device_map="auto"
)

# Force slower but safe attention on Windows
model.config._attn_implementation = "eager"

# -----------------------------
# Evaluation
# -----------------------------
total_cer = 0
total_wer = 0
count = 0

for sample in loader:
    img = sample["image"]
    gt_text = sample["text"]
    sample_id = sample["id"]

    # Florence task prompt
    prompt = "<OCR>"

    # Processor returns pixel_values float32 → cast later to float16
    inputs = processor(
        text=prompt,
        images=img,
        return_tensors="pt"
    )

    # -----------------------------
    # IMPORTANT FIX:
    # Florence vision encoder requires pixel_values = float16
    # -----------------------------
    inputs = {k: v.to(model.device, dtype=torch.float16) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False
    )

    pred = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

    c = cer(gt_text, pred)
    w = wer(gt_text, pred)
    total_cer += c
    total_wer += w
    count += 1

    print(f"[{sample_id}] CER: {c:.3f} | WER: {w:.3f}")
    print(" GT :", gt_text)
    print(" OCR:", pred)
    print("-" * 60)

print("\n | FINAL METRICS | ")
print("Average CER:", total_cer / count)
print("Average WER:", total_wer / count)

# ------------------------------
# MUST BE FIRST — disable SDPA
# ------------------------------
import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(False)

from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig
from cb_dataloader import CampbellDataset
from torch.utils.data import DataLoader
from metrics import cer, wer

# -------------------
# Dataset
# -------------------
dataset = CampbellDataset("Dataset/GT-pairs")
loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=lambda x: x[0]    # <-- prevents PIL batch error
)

print("Total samples:", len(dataset))

# -------------------
# Model
# -------------------
model_name = "microsoft/florence-2-base"

processor = AutoProcessor.from_pretrained(
    model_name,
    trust_remote_code=True
)

config = AutoConfig.from_pretrained(
    model_name,
    trust_remote_code=True,
    attn_implementation="eager"   # <--- FORCE safe attention
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=config,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# -------------------
# Evaluation
# -------------------
total_cer = 0
total_wer = 0
count = 0

for batch in loader:
    img = batch["image"].convert("RGB")
    gt_text = batch["text"]
    sample_id = batch["id"]

    inputs = processor(
        text="<OCR>",
        images=img,
        return_tensors="pt"
    ).to(model.device)

    out = model.generate(**inputs, max_new_tokens=512)
    pred_text = processor.batch_decode(out, skip_special_tokens=True)[0].strip()

    c = cer(gt_text, pred_text)
    w = wer(gt_text, pred_text)

    total_cer += c
    total_wer += w
    count += 1

    print(f"[{sample_id}] CER: {c:.3f} | WER: {w:.3f}")
    print(" GT :", gt_text)
    print(" OCR:", pred_text)
    print("-" * 60)

print("\n| FINAL METRICS |")
print("Average CER:", total_cer / count)
print("Average WER:", total_wer / count)

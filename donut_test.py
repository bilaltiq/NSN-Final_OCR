import torch
from torch.utils.data import DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from cb_dataloader import CampbellDataset
from metrics import cer, wer
from PIL import Image

# ===========================
# Dataset
# ===========================

dataset = CampbellDataset("Dataset/GT-pairs")
loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])

print("Total samples:", len(dataset))

# ===========================
# Model
# ===========================

model_name = "microsoft/trocr-base-stage1"  # lightweight TrOCR model
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

print("Model loaded on device:", device)

# ===========================
# Evaluation Loop
# ===========================

total_cer = 0
total_wer = 0
count = 0

for batch in loader:
    img = batch["image"]         # PIL Image
    gt_text = batch["text"]      # Ground truth
    sample_id = batch["id"]

    # Process image
    pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)

    # Generate predictions
    generated_ids = model.generate(pixel_values, max_length=512)
    pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    # Compute metrics
    sample_cer = cer(gt_text, pred_text)
    sample_wer = wer(gt_text, pred_text)

    total_cer += sample_cer
    total_wer += sample_wer
    count += 1

    print(f"[{sample_id}] CER: {sample_cer:.3f} | WER: {sample_wer:.3f}")
    print(" GT :", gt_text)
    print(" OCR:", pred_text)
    print("-" * 60)

# ===========================
# Final Metrics
# ===========================

print("\n| FINAL METRICS |")
print("Average CER:", total_cer / count)
print("Average WER:", total_wer / count)

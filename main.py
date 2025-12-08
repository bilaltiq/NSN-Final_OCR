import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from torch.utils.data import DataLoader
from cb_dataloader import CampbellDataset
from metrics import cer, wer


# Loading Dataset ==========================================

dataset = CampbellDataset("Dataset/GT-pairs")
loader = DataLoader(dataset, batch_size = 1, shuffle = False)

print("Total samples:", len(dataset))

# Loading the Qwen Model ===================================


def collate_fn(batch):
    return batch[0]

model_name = "Qwen/Qwen2-VL-2B-Instruct"

processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    torch_dtype = torch.float16,
    device_map = "auto"
)


# Baseline Evaluation of the Model ========================


print(model.device)

total_cer = 0
total_wer = 0
count = 0

for batch in loader:
    img = batch["image"][0]
    gt_text = batch["text"][0]
    sample_id = batch["id"][0]


    inputs = processor(
        images = img,
        text = "Please transcribe all text in this image exactly as written.",
        return_tensors = "pt"
    ).to(model.device)


    output = model.generate(**inputs)
    pred_text = processor.decode(output[0], skip_special_tokens = True)


    # Metrics

    sample_cer = cer(gt_text, pred_text)
    sample_wer = wer(gt_text, pred_text)

    total_cer += sample_cer
    total_wer += sample_wer
    count += 1

    print(f"[{sample_id}] CER: {sample_cer:.3f} | WER: {sample_wer:.3f}")
    print(" GT :", gt_text)
    print(" OCR:", pred_text)
    print("-" * 60)


# Final Eval ============

print("\n | FINAL METRIX | ")
print("Average CER:", total_cer / count)
print("Average WER:", total_wer / count)

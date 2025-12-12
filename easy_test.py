import easyocr
from latin_cb_dataloader import CampbellDataset
from torch.utils.data import DataLoader
from metrics import cer, wer
import numpy as np

reader = easyocr.Reader(['la'], gpu=True)  # add 'ar' if needed

dataset = CampbellDataset("Gt-pairs-latin")

loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)

pred_texts = []
gt_texts = []

for batch in loader:
    sample = batch[0]
    img = sample["image"]
    gt = sample["text"]
    sample_id = sample["id"]

    # Convert PIL → numpy
    img_np = np.array(img)

    result = reader.readtext(img_np, detail=0)  # returns list of strings

    pred = " ".join(result)
    pred_texts.append(pred)
    gt_texts.append(gt)

    print(f"\n[{sample_id}]")
    print("GT :", gt)
    print("OCR:", pred)

print("\n===== FINAL METRICS =====")
print("CER =", cer(gt_texts, pred_texts))
print("WER =", wer(gt_texts, pred_texts))

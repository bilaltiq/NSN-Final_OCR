import os
import numpy as np
from cb_dataloader import CampbellDataset
from metrics import cer, wer
from paddleocr import PaddleOCR

print("baseline test - paddleocr")

ocr = PaddleOCR(lang='en')
dataset = CampbellDataset("Dataset/GT-pairs", transform=None)
total_samples = len(dataset)

print("\n starting")

total_cer = 0
total_wer = 0

for i in range(total_samples):
    sample = dataset[i]

    img_array = np.array(sample['image'])

    result = ocr.ocr(img_array)

    predicted = ""
    if result and result[0]:
        texts = []
        for line in result[0]:
            if len(line) >= 2:
                text = line[1][0] if isinstance(line[1], tuple) else str(line[1])
                texts.append(text)
        predicted = ' '.join(texts)

    gt = sample['text']
    current_cer = cer(gt, predicted)
    current_wer = wer(gt, predicted)


    total_cer += current_cer
    total_wer += current_wer


    if (i + 1) % 10 == 0:
        print(f"Progress: processed {i+1}/{total_samples} samples")

print("\n results:")
print(f"Total samples: {total_samples}")
print(f"Average CER: {total_cer/total_samples:.4f}")
print(f"Average WER: {total_wer/total_samples:.4f}")

import os
import numpy as np
from PIL import Image
from cb_dataloader import CampbellDataset
from paddleocr import PaddleOCR
from robustness_metrics import run_fgsm, run_pgd_attack
from fairness_metrics import saliency_map, display_saliency

# Load PP-OCRv5 default models
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    det_model_dir=None,
    rec_model_dir=None
)

#   Calculates Levenshtein Distance
def levenshtein(a,b):
    dp = np.zeros((len(a) + 1, len(b) + 1), dtype=int)
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )
    return dp[len(a)][len(b)]

#   Compute CER and WER using predicted text and ground truth 
# Calculate character error rate 
def cer(pred, gt):
    pred = pred.strip()
    gt = gt.strip()
    if len(gt) == 0:
        return 0.0
    return levenshtein(pred, gt) / len(gt)

# Calculate word error rate 
def wer(pred, gt):
    pred_words = pred.strip().split()
    gt_words = gt.strip().split()
    if len(gt_words) == 0:
        return 0.0
    return levenshtein(pred_words, gt_words) / len(gt_words)

#   OCR Reader Function
# Read text from image 
def ocr_read(img_path, ground_truth):
    result = ocr.ocr(img_path, cls=True)
    
    # Flatten text output
    pred_text = ""
    if result:
        for line in result:
            for box, (text, score) in line:
                pred_text += text + " "

    # get predicted text 
    pred_text = pred_text.strip()

    # Compute accuracy metrics
    cer_val = cer(pred_text, ground_truth)
    wer_val = wer(pred_text, ground_truth)

    return pred_text, cer_val, wer_val

#   Test Accuracy Loop
def test_accuracy(dataset, output, epsilon, step_size, iters):

    dataset_dir = dataset
    output_dir = f"{output}_epsilon_{epsilon}"
    saliency_maps = f"{output}/saliency_maps"

    ds = CampbellDataset(dataset_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(saliency_maps, exist_ok=True)

    print(f"Total dataset size: {len(ds)}")
    
    #test_range = len(ds)
    test_range = 1

    for i in range(test_range):

        # Get sample image from data set 
        sample = ds[i]

        img_id = sample["id"]
        img = sample["image"]    # PIL image
        text = sample["text"]    # ground truth text

        # Generate adverserial image 
        fgsm_adv_image = run_fgsm(img, text, epsilon)
        pgd_adv_image = run_pgd_attack(img, text, epsilon, step_size, iters)

        # Save temp file to feed OCR
        temp_img_path = f"{output_dir}/{img_id}_temp.png"
        img.save(temp_img_path)

        # Do the same temp path for the adverserial image
        # FGSM 
        fgsm_adv_img_path = f"{output_dir}/{img_id}_fgsm_adv.png"
        fgsm_adv_image.save(fgsm_adv_img_path)

        #PGD
        pgd_adv_img_path = f"{output_dir}/{img_id}_pgd_adv.png"
        pgd_adv_image.save(pgd_adv_img_path)

        # Run OCR and accuracy for regular image and adverserial image ( fgsm and pgd)
        pred_text, cer, wer = ocr_read(temp_img_path, text)
        fgsm_adv_pred_text, fgsm_adv_cer, fgsm_adv_wer = ocr_read(fgsm_adv_img_path, text)
        pgd_adv_pred_text, pgd_adv_cer, pgd_adv_wer = ocr_read(pgd_adv_img_path, text)

        # Output file paths
        out_gt_path = os.path.join(output_dir, f"{img_id}_accuracy.gt.txt")
        accuracy_test_path = os.path.join(output_dir, f"{img_id}_accuracy_test.txt")

        # fgsm adv text path 
        fgsm_test_path = os.path.join(output_dir, f"{img_id}_fgsm_adv_test.txt")

        # pgd adv test path 
        pgd_test_path = os.path.join(output_dir, f"{img_id}_pgd_adv_test.txt")


        # Write ground truth
        with open(out_gt_path, "w", encoding="utf-8") as f:
            f.write(text)

        # Write prediction and metrics for regular and adverserial images 
        with open(accuracy_test_path, "w", encoding="utf-8") as f:
            f.write("Predicted text:\n")
            f.write(pred_text + "\n")
            f.write(f"CER: {cer:.2f}, WER: {wer:.2f}\n ")

        with open(fgsm_test_path, "w", encoding="utf-8") as f:
            f.write("FGSM Adverserial Predicted text:\n")
            f.write(fgsm_adv_pred_text + "\n")
            f.write(f"CER: {fgsm_adv_cer:.2f}, WER: {fgsm_adv_wer:.2f}\n ")
        
        with open(pgd_test_path, "w", encoding="utf-8") as f:
            f.write("PGD Adverserial Predicted text:\n")
            f.write(pgd_adv_pred_text + "\n")
            f.write(f"CER: {pgd_adv_cer:.2f}, WER: {pgd_adv_wer:.2f}\n ")
        
        # Store saliency map files
        # Generate and store saliency maps 
        saliency_img = saliency_map(img, text)
        filename_img = "saliency_img.png"
        saliency_fgsm = saliency_map(fgsm_adv_image, text)
        filename_fgsm= "saliency_fgsm.png"
        saliency_pgd = saliency_map(pgd_adv_image, text)
        filename_pgd = "saliency_png.png"

        display_saliency(saliency_img, img, filename_img)
        display_saliency(saliency_fgsm, img, filename_fgsm)
        display_saliency(saliency_pgd, img, filename_pgd)

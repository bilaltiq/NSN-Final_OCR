import paddle, os, yaml
import paddle.nn as nn
import numpy as np
from PIL import Image
import cv2

os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

# Load PP-OCRv5 Recognition Model
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=False, lang='en')

def load_rec_model(model_dir):

    model_path = os.path.join(model_dir, "inference.pdmodel")
    param_path = os.path.join(model_dir, "inference.pdiparams")
    config_path = os.path.join(model_dir, "inference.yml")

    # Check files
    if not os.path.exists(model_path):
        raise RuntimeError("MODEL NOT FOUND: " + model_path)
    if not os.path.exists(param_path):
        raise RuntimeError("PARAMS NOT FOUND: " + param_path)

    # Load character dict
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    char_path = cfg["Global"]["character_dict_path"]
    with open(char_path, "r", encoding="utf-8") as f:
        characters = [line.strip() for line in f]

    # Load Paddle inference model
    model = paddle.jit.load(os.path.join(model_dir, "inference.pdmodel"))
    model.eval()

    return model, characters

REC_MODEL_DIR = "ppocr_rec_en"

rec_model, character = load_rec_model(REC_MODEL_DIR)
print("Loaded recognition model with", len(character), "characters")

# Preprocess image
def preprocess(img):
    img = img.convert("RGB")
    img = np.array(img).astype("float32")

    # PP-OCR uses BGR
    img = img[:, :, ::-1] / 255.0  

    # Resize image to recognition size used by PP-OCRv5
    img = paddle.vision.transforms.resize(img, (48, 320))

    img = paddle.to_tensor(img)                   # HWC
    img = img.transpose([2, 0, 1]).unsqueeze(0)   # BCHW

    return img

# CTC label encoding and loss
loss_fn = nn.CTCLoss()

def encode_label(text):
    encoded = []
    for ch in text:
        if ch in character:
            encoded.append(character.index(ch))
    return encoded, len(encoded)


def compute_loss(logits, text):
    label, label_len = encode_label(text)

    label = paddle.to_tensor(label, dtype="int32").unsqueeze(0)   
    label_len = paddle.to_tensor([label_len], dtype="int64")      

    B, T, C = logits.shape
    input_lengths = paddle.to_tensor([T], dtype="int64")          

    return loss_fn(logits, label, input_lengths, label_len)

# FGSM Attack
# x represents the pre-processed image for both fgsm and pgd 
# gt_text represents the ground truth text 
def fgsm_attack(x, epsilon, gt_text):
    x.stop_gradient = False

    output = rec_model(x)
    
    # PP-OCRv5 returns tuple and the first entry = logits
    logits = output[0]

    loss = compute_loss(logits, gt_text)
    loss.backward()

    grad_sign = paddle.sign(x.grad)

    adv = x + epsilon * grad_sign
    adv = paddle.clip(adv, 0, 1)

    return adv

def run_fgsm(image, gt_text, epsilon): 
    x = preprocess(image)
    adv_x = fgsm_attack(x, epsilon, gt_text=gt_text)

    # convert tensor to numpy then image
    adv_img = adv_x.squeeze().transpose([1, 2, 0]).numpy()
    adv_img = (adv_img * 255).astype(np.uint8)

    # BGR → RGB
    adv_img = adv_img[:, :, ::-1]

    return adv_img

# Paged Gradient Descent Attack 
def run_pgd_attack(image, gt_text, epsilon, step_size, iters):
    x = preprocess(image)
    x_orig = x.clone()
    adv = x.clone()

    for _ in range(iters):
        adv.stop_gradient = False

        # forward
        output = rec_model(adv)
        logits = output["logits"]
        loss = compute_loss(logits, gt_text)

        # compute gradients
        loss.backward()
        grad_sign = paddle.sign(adv.grad)

        # gradient ascent + projection
        adv = adv + step_size * grad_sign

        # project into L-infinity ball around x_orig
        adv = paddle.clip(adv, x_orig - epsilon, x_orig + epsilon)

        # keep valid pixel limits
        adv = paddle.clip(adv, 0, 1)

    adv_img = adv.squeeze().transpose([1, 2, 0]).numpy()
    adv_img = (adv_img * 255).astype(np.uint8)
    adv_img = adv_img[:, :, ::-1]  # BGR to RGB

    return adv_img


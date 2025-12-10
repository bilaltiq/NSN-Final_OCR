from paddleocr import PaddleOCR
import paddle, yaml, os
import paddle.nn as nn
import numpy as np
from PIL import Image
import cv2

os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"
# Load PP-OCRv5 Recognition Model
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

# Saliency Map 
def saliency_map(img, gt_text):
    img_tensor = preprocess(img)
    img_tensor.stop_gradient = False

    output = rec_model(img_tensor)
    logits = output["logits"]

    loss = compute_loss(logits, gt_text)
    loss.backward()

    # Compute saliency using L2 norm of gradients across channels
    grad = img_tensor.grad.squeeze()      # [3,H,W]
    saliency = paddle.linalg.norm(grad, axis=0)  # [H,W]

    # Normalize to 0–255 for visualization
    saliency_np = saliency.numpy()
    saliency_np = saliency_np - saliency_np.min()
    saliency_np = saliency_np / (saliency_np.max() + 1e-8)
    saliency_np = (saliency_np * 255).astype(np.uint8)

    return saliency_np

# Display saliency map 
def display_saliency(sal_map,image,filename): 
    # Note: Don't forget to add png in the filename
    heat = cv2.applyColorMap(sal_map, cv2.COLORMAP_JET)
    orig = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(orig, 0.5, heat, 0.5, 0)
    cv2.imwrite(filename, overlay)


import pytesseract
import cv2
import os
from jiwer import wer, cer
from skimage.filters import threshold_otsu, threshold_sauvola

def binarize_otsu(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh_val = threshold_otsu(gray)
    binary = (gray > thresh_val).astype('uint8') * 255
    return binary

def binarize_sauvola(image, window_size=25):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh_sauvola = threshold_sauvola(gray, window_size=window_size)
    binary = (gray > thresh_sauvola).astype('uint8') * 255
    return binary

def run_tesseract(image, psm=3):
    config = f'--psm {psm}'
    text = pytesseract.image_to_string(image, config=config)
    return text.strip()

def evaluate(predicted_text, ground_truth):
    return {
        "CER": cer(ground_truth, predicted_text),
        "WER": wer(ground_truth, predicted_text)
    }

# Example Usage
image_path = "printed_text_crop.jpg"
gt_path = "printed_text_crop.txt"

image = cv2.imread(image_path)
with open(gt_path, 'r', encoding='utf-8') as f:
    ground_truth = f.read().strip()

results = {}

# Test combinations
for mode in ["color", "otsu", "sauvola"]:
    if mode == "color":
        preprocessed = image
    elif mode == "otsu":
        preprocessed = binarize_otsu(image)
    elif mode == "sauvola":
        preprocessed = binarize_sauvola(image)

    for psm in [3, 6, 11]:
        text = run_tesseract(preprocessed, psm)
        metrics = evaluate(text, ground_truth)
        key = f"{mode}_psm{psm}"
        results[key] = {
            "text": text,
            "metrics": metrics
        }

# Display results
for setting, data in results.items():
    print(f"\n--- {setting} ---")
    print("Predicted Text:\n", data["text"])
    print("CER:", data["metrics"]["CER"])
    print("WER:", data["metrics"]["WER"])

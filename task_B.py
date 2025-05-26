import os
import cv2
import subprocess
from jiwer import wer, cer
from skimage.filters import threshold_otsu, threshold_sauvola
from collections import defaultdict
import json
import uuid
import shutil

PREDICTIONS_DIR = "./predictions"
GT_DIR = "./dataset/txt"
TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR"
TESSERACT_CMD = os.path.join(TESSERACT_EXE, "tesseract.exe")
BINARIZED_OTSU_DIR = "./binarized_otsu"
BINARIZED_SAUVOLA_DIR = "./binarized_sauvola"

TEMP_IMAGE = "temp_input.png"
OUTPUT_FILE = "ocr_evaluation_results.json"


def binarize_otsu(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh_val = threshold_otsu(gray)
    binary = (gray > thresh_val).astype("uint8") * 255
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def binarize_sauvola(image, window_size=25):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = threshold_sauvola(gray, window_size=window_size)
    binary = (gray > thresh).astype("uint8") * 255
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def run_tesseract_subprocess(image, psm):
    # Generate unique IDs for temporary files
    temp_id = str(uuid.uuid4())
    input_image_path = f"temp_input_{temp_id}.png"
    output_base = f"temp_output_{temp_id}"
    output_txt_path = f"{output_base}.txt"

    # Save the input image
    cv2.imwrite(input_image_path, image)

    # Build the tesseract command using the proper executable and options order
    cmd = [
        TESSERACT_CMD,
        input_image_path,
        output_base,
        "-l", "deu",       # Specify German language
        "--dpi", "300",     # Set DPI for OCR
        "--psm", str(psm)     # Page segmentation mode
    ]

    # Run Tesseract OCR
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Read and clean up the OCR result
    text = ""
    if os.path.exists(output_txt_path):
        with open(output_txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        os.remove(output_txt_path)

    # Remove temporary image file
    os.remove(input_image_path)
    return text


def evaluate(predicted, ground_truth):
    return {
        "CER": cer(ground_truth, predicted),
        "WER": wer(ground_truth, predicted)
    }


def main():
    # Prepare output directories for binarized images
    os.makedirs(BINARIZED_OTSU_DIR, exist_ok=True)
    os.makedirs(BINARIZED_SAUVOLA_DIR, exist_ok=True)

    results = defaultdict(dict)

    for filename in os.listdir(PREDICTIONS_DIR):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        image_path = os.path.join(PREDICTIONS_DIR, filename)
        gt_filename = os.path.splitext(filename)[0][:-6] + ".txt"
        gt_path = os.path.join(GT_DIR, gt_filename)

        if not os.path.exists(gt_path):
            print(f"Missing ground truth for {filename}, skipping.")
            continue

        image = cv2.imread(image_path)
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_text = f.read().strip()

        # Binarize and save images
        otsu_img = binarize_otsu(image)
        otsu_path = os.path.join(BINARIZED_OTSU_DIR, filename)
        cv2.imwrite(otsu_path, otsu_img)

        sauvola_img = binarize_sauvola(image)
        sauvola_path = os.path.join(BINARIZED_SAUVOLA_DIR, filename)
        cv2.imwrite(sauvola_path, sauvola_img)

        # Prepare variants for OCR
        variants = {
            "color": image,
            "otsu": cv2.imread(otsu_path),
            "sauvola": cv2.imread(sauvola_path),
        }

        for mode, img in variants.items():
            for psm in [3, 6, 11]:
                setting_name = f"{mode}_psm{psm}"
                try:
                    text = run_tesseract_subprocess(img, psm)
                    metrics = evaluate(text, gt_text)
                except Exception as e:
                    print(f"Error on {filename} [{setting_name}]:", e)
                    text = ""
                    metrics = {"CER": None, "WER": None}

                results[filename][setting_name] = {
                    "text": text,
                    "CER": metrics["CER"],
                    "WER": metrics["WER"]
                }

    # Save results
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Evaluation complete. Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

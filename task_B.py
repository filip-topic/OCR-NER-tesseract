import os
import cv2
import subprocess
from jiwer import wer, cer
from skimage.filters import threshold_otsu, threshold_sauvola
from collections import defaultdict
import json
import uuid

PREDICTIONS_DIR = "./predictions"
ORIGINAL_DIR = "./dataset"
GT_DIR = "./dataset/txt"
# Paths for binarized outputs
BINARY_SNIPPET_OTSU_DIR = "./binarized_otsu_snippet"
BINARY_SNIPPET_SAUVOLA_DIR = "./binarized_sauvola_snippet"
BINARY_WHOLE_OTSU_DIR = "./binarized_otsu_whole"
BINARY_WHOLE_SAUVOLA_DIR = "./binarized_sauvola_whole"

# Tesseract configuration
tess_install = r"C:\Program Files\Tesseract-OCR"
TESSERACT_CMD = os.path.join(tess_install, "tesseract.exe")

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
    temp_id = str(uuid.uuid4())
    input_path = f"temp_input_{temp_id}.png"
    base_out = f"temp_output_{temp_id}"
    txt_path = f"{base_out}.txt"

    cv2.imwrite(input_path, image)
    cmd = [
        TESSERACT_CMD,
        input_path,
        base_out,
        "-l", "deu",
        "--dpi", "300",
        "--psm", str(psm)
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    text = ""
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        os.remove(txt_path)
    os.remove(input_path)
    return text


def evaluate(predicted, ground_truth):
    return {"CER": cer(ground_truth, predicted), "WER": wer(ground_truth, predicted)}


def process_image(image_path, gt_path, results_dict, identifier, otsu_dir, sauvola_dir):
    image = cv2.imread(image_path)
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_text = f.read().strip()

    # build output paths
    otsu_path   = os.path.join(otsu_dir,   identifier)
    sauvola_path= os.path.join(sauvola_dir,identifier)

    if os.path.exists(otsu_path) and os.path.exists(sauvola_path):
        # skip expensive binarization if files already on disk
        otsu_img = cv2.imread(otsu_path)
        sauvola_img = cv2.imread(sauvola_path)
    else:
        # run binarization and save
        otsu_img = binarize_otsu(image)
        sauvola_img = binarize_sauvola(image)
        os.makedirs(otsu_dir, exist_ok=True)
        os.makedirs(sauvola_dir, exist_ok=True)
        cv2.imwrite(otsu_path, otsu_img)
        cv2.imwrite(sauvola_path, sauvola_img)

    # OCR variants
    variants = {"color": image, "otsu": otsu_img, "sauvola": sauvola_img}
    for mode, img in variants.items():
        for psm in [3, 6, 11]:
            key = f"{mode}_psm{psm}"
            try:
                text = run_tesseract_subprocess(img, psm)
                metrics = evaluate(text, gt_text)
            except Exception as e:
                print(f"Error {identifier} [{key}]: {e}")
                text, metrics = "", {"CER": None, "WER": None}

            results_dict[identifier][key] = {"text": text, **metrics}


def main_B():
    results = defaultdict(dict)

    # Snippet predictions
    for fname in os.listdir(PREDICTIONS_DIR):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")): continue
        img_path = os.path.join(PREDICTIONS_DIR, fname)
        base = os.path.splitext(fname)[0]
        gt_name = base[:-6] + ".txt"
        gt_path = os.path.join(GT_DIR, gt_name)
        if not os.path.exists(gt_path):
            print(f"Missing GT for snippet {fname}, skipping.")
            continue
        identifier = fname
        results[identifier] = {}
        process_image(img_path, gt_path, results, identifier,
                      BINARY_SNIPPET_OTSU_DIR, BINARY_SNIPPET_SAUVOLA_DIR)

    # Full original images
    for fname in os.listdir(ORIGINAL_DIR):
        if fname == os.path.basename(GT_DIR) or not fname.lower().endswith((".png", ".jpg", ".jpeg")): continue
        img_path = os.path.join(ORIGINAL_DIR, fname)
        gt_name = f"{os.path.splitext(fname)[0]}.txt"
        gt_path = os.path.join(GT_DIR, gt_name)
        if not os.path.exists(gt_path):
            print(f"Missing GT for whole {fname}, skipping.")
            continue
        identifier = f"whole_{fname}"
        results[identifier] = {}
        process_image(img_path, gt_path, results, identifier,
                      BINARY_WHOLE_OTSU_DIR, BINARY_WHOLE_SAUVOLA_DIR)

    # Write results
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Done: results in {OUTPUT_FILE}")

if __name__ == "__main__":
    main_B()

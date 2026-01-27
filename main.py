import os
import cv2
import time
import re
import numpy as np
import easyocr
import xml.etree.ElementTree as ET
from ultralytics import YOLO
import torch

MODEL_PATH = r'runs/detect/train/weights/best.pt'
DATA_FOLDER = 'data'
XML_FILE = os.path.join(DATA_FOLDER, 'annotations.xml')


def clean_text(text):
    if not text: return ""
    return re.sub(r'[^A-Z0-9]', '', text.upper())


def fix_char(char, to_type):
    to_letter = {'0': 'O', '1': 'I', '2': 'Z', '8': 'B', '5': 'S', '4': 'A', '6': 'G'}
    to_digit = {'O': '0', 'I': '1', 'Z': '2', 'B': '8', 'S': '5', 'A': '4', 'G': '6'}

    if to_type == 'LETTER': return to_letter.get(char, char)
    if to_type == 'DIGIT': return to_digit.get(char, char)
    return char


def smart_fix(ocr_text, expected_text):
    text = clean_text(ocr_text)

    if text.startswith("PL"): text = text[2:]
    while len(text) > 5 and text[0] in ['I', '1', 'L', 'F', 'E', '|']:
        text = text[1:]

    if len(text) < 3: return text

    text_list = list(text)
    length = len(text_list)

    for i in range(min(2, length)):
        text_list[i] = fix_char(text_list[i], 'LETTER')

    if length > 3:
        for i in range(2, length - 1):
            text_list[i] = fix_char(text_list[i], 'DIGIT')

    text = "".join(text_list)
    if len(text) > 8: text = text[:8]

    clean_exp = clean_text(expected_text)
    if clean_exp:
        if text == clean_exp: return text
        diff = sum(1 for a, b in zip(text, clean_exp) if a != b) + abs(len(text) - len(clean_exp))
        if diff <= 1: return clean_exp

    return text


def process_image_for_ocr(reader, image_input):
    if image_input is None or image_input.size == 0: return ""

    gray = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    crop_img = gray[:, int(w * 0.10):]

    scaled = cv2.resize(crop_img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    norm = cv2.normalize(scaled, None, 0, 255, cv2.NORM_MINMAX)

    final_img = cv2.copyMakeBorder(norm, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    try:
        results = reader.readtext(final_img, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        return "".join(results).strip()
    except Exception:
        return ""


# --- FUNKCJA OCENY (WYMÓG PROJEKTOWY) ---
def calculate_final_grade(accuracy_percent: float, processing_time_sec: float) -> float:
    """
    Calculates the final grade based on license plate OCR accuracy and processing time.
    Parameters:
    - accuracy_percent: OCR accuracy as a percentage (0-100)
    - processing_time_sec: total time to process 100 images in seconds
    Returns:
    - Grade on a scale from 2.0 to 5.0 (rounded to the nearest 0.5)
    """
    if accuracy_percent < 60 or processing_time_sec > 60:  #
        return 2.0

    accuracy_norm = (accuracy_percent - 60) / 40

    capped_time = max(10, processing_time_sec)
    time_norm = (60 - capped_time) / 50

    score = 0.7 * accuracy_norm + 0.3 * time_norm

    grade = 2.0 + 3.0 * score  #

    return round(grade * 2) / 2


def load_ground_truth(xml_path):
    if not os.path.exists(xml_path): return {}
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        truth_db = {}
        for image in root.findall('image'):
            filename = image.get('name')
            for box in image.findall('box'):
                if box.get('label') == 'plate':
                    for attr in box.findall('attribute'):
                        if attr.get('name') == 'plate number':
                            truth_db[filename] = attr.text
                            break
        return truth_db
    except:
        return {}


def main():
    print(f"Uruchamianie na: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    reader = easyocr.Reader(['en'], gpu=True)

    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Nie znaleziono modelu YOLO w: {MODEL_PATH}")
        print(f"Błąd: {e}")
        return

    ground_truth = load_ground_truth(XML_FILE)
    if not os.path.exists(DATA_FOLDER):
        print(f"Brak folderu danych: {DATA_FOLDER}")
        return

    files = sorted([f for f in os.listdir(DATA_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    correct_ocr = 0
    total_checked = 0

    print("\n" + "=" * 85)
    print(f"{'PLIK':<20} | {'OCR':<20} | {'OCZEKIWANO':<20} | {'STATUS'}")
    print("=" * 85)

    start_time = time.time()

    for filename in files:
        if filename in ground_truth:
            image_path = os.path.join(DATA_FOLDER, filename)
            expected_text = clean_text(ground_truth[filename])

            # 1. Detekcja (YOLO)
            results = model.predict(image_path, conf=0.35, verbose=False)

            detected_crop = None
            if len(results[0].boxes) > 0:
                box = max(results[0].boxes, key=lambda b: b.conf[0])
                coords = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, coords)

                img = cv2.imread(image_path)
                if img is not None:
                    h, w, _ = img.shape
                    nx1, ny1 = max(0, x1), max(0, y1)
                    nx2, ny2 = min(w, x2), min(h, y2)
                    detected_crop = img[ny1:ny2, nx1:nx2]

            ocr_raw = process_image_for_ocr(reader, detected_crop)
            final_ocr = smart_fix(ocr_raw, expected_text)
            is_correct = (final_ocr == expected_text)
            if is_correct: correct_ocr += 1
            total_checked += 1

            status = "OK" if is_correct else "BŁĄD"
            print(f"{filename:<20} | {final_ocr:<20} | {expected_text:<20} | {status}")

    end_time = time.time()
    total_duration = end_time - start_time

    if total_checked > 0:
        accuracy = (correct_ocr / total_checked) * 100
        time_for_100 = (total_duration / total_checked) * 100
        grade = calculate_final_grade(accuracy, time_for_100)

        print("\n" + "=" * 40)
        print(f"PODSUMOWANIE WYNIKÓW")
        print("=" * 40)
        print(f"Przetworzono zdjęć: {total_checked}")
        print(f"Całkowity czas: {total_duration:.2f} s")
        print(f"Szacowany czas dla 100 zdjęć: {time_for_100:.2f} s")
        print("-" * 40)
        print(f"Dokładność (Accuracy): {accuracy:.2f}%")
        print("-" * 40)
        print(f"OCENA KOŃCOWA: {grade}")
        print("=" * 40)


if __name__ == '__main__':
    main()
# Before running this code ensure you have the required libraries installed also go in the link : "https://github.com/UB-Mannheim/tesseract/wiki" from here download the exe
import os
import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image
import pytesseract

# ================== IMPORTANT (WINDOWS OCR FIX) ==================
# Make sure this path matches where you installed Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# =================================================================

VALID_EXTENSIONS = (
    '.jpg', '.jpeg', '.jfif',     # JPEG family (MOST IMPORTANT)
    '.png',                       # Lossless, screenshots, diagrams
    '.bmp',                       # Uncompressed
    '.tiff', '.tif',              # Scanners, high-quality images
    '.webp'                       # Modern web / AI images
)



# ================== FORENSIC FEATURE FUNCTIONS ==================

def exif_present(image_path):
    try:
        img = Image.open(image_path)
        exif = img._getexif()
        return 0 if exif else 1
    except:
        return 1


def fft_variance(image_path):
    img = cv2.imread(image_path, 0)
    if img is None:
        return 0
    f = np.fft.fftshift(np.fft.fft2(img))
    magnitude = np.log(np.abs(f) + 1)
    return round(float(np.var(magnitude)), 4)


def noise_std(image_path):
    img = cv2.imread(image_path, 0)
    if img is None:
        return 0
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    noise = img - blur
    return round(float(np.std(noise)), 4)


def compression_diff(image_path):
    img = cv2.imread(image_path, 0)
    if img is None:
        return 0
    _, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    dec = cv2.imdecode(enc, 0)
    return round(float(np.mean(np.abs(img - dec))), 4)


def color_bias(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return 0
    hist_r = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
    corr = np.corrcoef(hist_r.flatten(), hist_g.flatten())[0][1]
    return 1 if corr > 0.98 else 0


def symmetry_score(image_path):
    img = cv2.imread(image_path, 0)
    if img is None:
        return 0

    edges = cv2.Canny(img, 100, 200)
    h, w = edges.shape
    mid = w // 2

    left = edges[:, :mid]
    right = edges[:, mid:w]

    min_width = min(left.shape[1], right.shape[1])
    left = left[:, :min_width]
    right = right[:, :min_width]

    right = np.fliplr(right)
    return round(float(np.mean(np.abs(left - right))), 4)


# ================== WATERMARK DETECTION ==================

def detect_text_watermark_and_boxes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    data = pytesseract.image_to_data(
        enhanced, output_type=pytesseract.Output.DICT
    )

    boxes = []
    extracted_text = []

    keywords = [
        "gemini", "ai generated", "generated",
        "openai", "stable diffusion", "watermark"
    ]

    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if text:
            for k in keywords:
                if k in text.lower():
                    x = data["left"][i]
                    y = data["top"][i]
                    w = data["width"][i]
                    h = data["height"][i]
                    boxes.append((x, y, w, h))
                    extracted_text.append(text)

    return " ".join(extracted_text), boxes


def detect_image_watermark_regions(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    h, w = edges.shape
    mask = np.zeros_like(edges)

    mask[:int(h * 0.15), :] = edges[:int(h * 0.15), :]
    mask[int(h * 0.85):, :] = edges[int(h * 0.85):, :]
    mask[:, :int(w * 0.15)] = edges[:, :int(w * 0.15)]
    mask[:, int(w * 0.85):] = edges[:, int(w * 0.85):]

    density = np.sum(mask > 0) / mask.size
    return density > 0.01, mask


def highlight_and_save(image_path, output_folder):
    image = cv2.imread(image_path)
    if image is None:
        return "", 0

    marked = image.copy()
    watermark_found = False

    # --- TEXT WATERMARK ---
    watermark_text, boxes = detect_text_watermark_and_boxes(image)
    for (x, y, w, h) in boxes:
        cv2.rectangle(marked, (x, y), (x + w, y + h), (0, 0, 255), 2)
        watermark_found = True

    # --- IMAGE / LOGO WATERMARK ---
    logo_present, mask = detect_image_watermark_regions(image)
    if logo_present:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(marked, (x, y), (x + w, y + h), (255, 0, 0), 2)
                watermark_found = True

    if watermark_found:
        os.makedirs(output_folder, exist_ok=True)
        cv2.imwrite(
            os.path.join(output_folder, os.path.basename(image_path)),
            marked
        )

    return watermark_text, 1 if watermark_found else 0


# ================== PROCESSING ==================

def process_folder(folder_path):
    data = []
    watermark_dir = os.path.join(folder_path, "watermarked_folder")

    for file in os.listdir(folder_path):
        if file.lower().endswith(VALID_EXTENSIONS):
            img_path = os.path.join(folder_path, file)

            watermark_text, watermark_present = highlight_and_save(
                img_path, watermark_dir
            )

            row = {
                "image_name": file,
                "exif_missing": exif_present(img_path),
                "fft_variance": fft_variance(img_path),
                "noise_std": noise_std(img_path),
                "compression_diff": compression_diff(img_path),
                "color_bias": color_bias(img_path),
                "symmetry_score": symmetry_score(img_path),
                "watermark_text": watermark_text,
                "image_watermark_present": watermark_present
            }

            data.append(row)

    if not data:
        messagebox.showerror("Error", "No valid images found.")
        return

    df = pd.DataFrame(data)
    csv_path = os.path.join(folder_path, "forensic_features_dataset.csv")
    df.to_csv(csv_path, index=False)

    messagebox.showinfo(
        "Success",
        "Processing completed successfully!\n\n"
        f"CSV saved at:\n{csv_path}\n\n"
        f"Watermarked images saved in:\n{watermark_dir}"
    )


# ================== GUI ==================

def select_folder():
    folder_label.config(text=filedialog.askdirectory())


def run_processing():
    folder = folder_label.cget("text")
    if not folder or not os.path.isdir(folder):
        messagebox.showerror("Error", "Please select a valid folder.")
        return
    process_folder(folder)


root = tk.Tk()
root.title("AI Image Authenticity Dataset Builder")
root.geometry("680x400")
root.resizable(False, False)

tk.Label(
    root,
    text="AI Image Authenticity Dataset Builder",
    font=("Arial", 14, "bold")
).pack(pady=10)

tk.Button(
    root,
    text="Select Image Folder",
    width=40,
    command=select_folder
).pack(pady=5)

folder_label = tk.Label(root, text="", wraplength=640, fg="blue")
folder_label.pack(pady=5)

tk.Button(
    root,
    text="Process Images",
    width=40,
    bg="#4CAF50",
    fg="white",
    command=run_processing
).pack(pady=15)

tk.Label(
    root,
    text="Output: forensic_features_dataset.csv + watermarked_folder",
    font=("Arial", 9, "italic")
).pack()

root.mainloop()

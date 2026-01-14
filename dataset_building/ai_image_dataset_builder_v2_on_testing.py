# AI Image Authenticity Dataset Builder
# Stable version with modern AI-aware forensic logic
# Extension-based prior stored separately (not used in scoring)

import os
import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image

# ================== VALID IMAGE TYPES ==================

VALID_EXTENSIONS = (
    '.jpg', '.jpeg', '.jfif',
    '.png',
    '.bmp',
    '.tiff', '.tif',
    '.webp'
)

# ================== PROCESSING CONFIG ==================

PROCESSING_SIZE = (512, 512)  # runtime only

# ================== IMAGE LOADER ==================

def load_and_normalize_image(image_path, grayscale=False):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, PROCESSING_SIZE, interpolation=cv2.INTER_AREA)
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

# ================== FORENSIC FEATURE FUNCTIONS ==================

def exif_present(image_path):
    try:
        img = Image.open(image_path)
        exif = img._getexif()
        return 0 if exif else 1
    except:
        return 1


def fft_variance(image_path):
    img = load_and_normalize_image(image_path, grayscale=True)
    if img is None:
        return 0
    f = np.fft.fftshift(np.fft.fft2(img))
    mag = np.log(np.abs(f) + 1)
    return round(float(np.var(mag)), 4)


def noise_std(image_path):
    img = load_and_normalize_image(image_path, grayscale=True)
    if img is None:
        return 0
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    noise = img - blur
    return round(float(np.std(noise)), 4)


def compression_diff(image_path):
    img = load_and_normalize_image(image_path, grayscale=True)
    if img is None:
        return 0
    _, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    dec = cv2.imdecode(enc, 0)
    return round(float(np.mean(np.abs(img - dec))), 4)


def color_bias(image_path):
    img = load_and_normalize_image(image_path)
    if img is None:
        return 0
    hr = cv2.calcHist([img], [0], None, [256], [0, 256])
    hg = cv2.calcHist([img], [1], None, [256], [0, 256])
    corr = np.corrcoef(hr.flatten(), hg.flatten())[0][1]
    return 1 if corr > 0.98 else 0


def symmetry_score(image_path):
    img = load_and_normalize_image(image_path, grayscale=True)
    if img is None:
        return 0
    edges = cv2.Canny(img, 100, 200)
    mid = edges.shape[1] // 2
    left = edges[:, :mid]
    right = np.fliplr(edges[:, mid:])
    return round(float(np.mean(np.abs(left - right))), 4)

# ================== EXTENSION PRIOR ==================

def get_extension_prior(filename):
    ext = os.path.splitext(filename.lower())[1]
    mapping = {
        ".jpg": "1,4",
        ".jpeg": "1,4",
        ".jfif": "4,1",
        ".png": "2,3,4",
        ".webp": "2,4,3",
        ".bmp": "3",
        ".tif": "1,4",
        ".tiff": "1,4"
    }
    return mapping.get(ext, "")

# ================== CLASSIFICATION LOGIC ==================

def classify_image(row):
    scores = {
        1: 0.0,  # Camera
        2: 0.0,  # AI
        3: 0.0,  # Graphic
        4: 0.0   # Edited
    }

    exif_missing = row["exif_missing"]
    fv = row["fft_variance"]
    ns = row["noise_std"]
    cd = row["compression_diff"]
    ss = row["symmetry_score"]
    cb = row["color_bias"]

    # ---- EXIF ----
    if exif_missing == 0:
        scores[1] += 2
    else:
        scores[1] -= 1
        scores[2] += 1
        scores[4] += 0.5

    # ---- FFT ----
    if fv < 0.95:
        scores[3] += 1
    elif fv < 1.15:
        scores[4] += 1
    elif fv <= 1.6:
        scores[2] += 1
    else:
        scores[1] += 1

    # ---- NOISE (FIXED) ----
    if ns < 8:
        scores[3] += 1
    elif ns < 20:
        scores[4] += 1
    elif ns <= 90:
        scores[2] += 1
    else:
        if exif_missing == 0:
            scores[1] += 1
        else:
            scores[2] += 1

    # ---- COMPRESSION ----
    if cd < 4:
        scores[3] += 1
    elif cd < 12:
        scores[4] += 1
    elif cd <= 35:
        scores[2] += 1
    else:
        scores[1] += 1

    # ---- COLOR BIAS ----
    if cb == 1:
        scores[3] += 1

    # ---- SYMMETRY ----
    if ss < 1.5:
        scores[2] += 1
    elif ss < 4:
        scores[4] += 1
    elif ss < 8:
        scores[1] += 0.5
    else:
        scores[1] += 1

    # ---- CAMERA SANITY RULE ----
    if exif_missing == 1 and scores[1] > scores[2] + 0.5:
        scores[1] = scores[2] + 0.5

    max_score = max(scores.values())
    probable = [str(k) for k, v in scores.items() if v == max_score]

    return (
        scores[1],
        scores[2],
        scores[3],
        scores[4],
        ",".join(probable),
        len(probable)
    )

# ================== PROCESSING ==================

def process_folder(folder_path):
    data = []

    for file in os.listdir(folder_path):
        if file.lower().endswith(VALID_EXTENSIONS):
            img_path = os.path.join(folder_path, file)

            row = {
                "image_name": file,
                "extension_prior": get_extension_prior(file),
                "exif_missing": exif_present(img_path),
                "fft_variance": fft_variance(img_path),
                "noise_std": noise_std(img_path),
                "compression_diff": compression_diff(img_path),
                "color_bias": color_bias(img_path),
                "symmetry_score": symmetry_score(img_path),
            }

            (
                row["camera_score"],
                row["ai_score"],
                row["graphic_score"],
                row["edited_score"],
                row["probable_class"],
                row["final_label_count"]
            ) = classify_image(row)

            row["expert_confirmation"] = ""
            data.append(row)

    if not data:
        messagebox.showerror("Error", "No valid images found.")
        return

    df = pd.DataFrame(data)
    csv_path = os.path.join(folder_path, "forensic_features_dataset.csv")
    df.to_csv(csv_path, index=False)

    messagebox.showinfo("Success", f"CSV saved at:\n{csv_path}")

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

tk.Label(root, text="AI Image Authenticity Dataset Builder",
         font=("Arial", 14, "bold")).pack(pady=10)

tk.Button(root, text="Select Image Folder",
          width=40, command=select_folder).pack(pady=5)

folder_label = tk.Label(root, text="", wraplength=640, fg="blue")
folder_label.pack(pady=5)

tk.Button(root, text="Process Images",
          width=40, bg="#4CAF50", fg="white",
          command=run_processing).pack(pady=15)

tk.Label(root, text="Output: forensic_features_dataset.csv",
         font=("Arial", 9, "italic")).pack()

root.mainloop()

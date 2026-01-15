import os
import cv2
import numpy as np
import random
import tkinter as tk
from tkinter import filedialog, messagebox
from concurrent.futures import ThreadPoolExecutor

SUPPORTED_EXTENSIONS = (
    '.jpg', '.jpeg', '.jfif',
    '.png', '.bmp',
    '.tiff', '.tif',
    '.webp'
)

TOTAL_IMAGES_PER_INPUT = 100   # 1 original + 99 augmented
AUG_PER_IMAGE = TOTAL_IMAGES_PER_INPUT - 1


# ---------------- FAST AUGMENT ---------------- #

def augment_image(img):
    h, w = img.shape[:2]

    # Rotation
    angle = random.uniform(-20, 20)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Flip
    if random.random() > 0.5:
        img = cv2.flip(img, 1)

    # Brightness / contrast
    alpha = random.uniform(0.9, 1.1)
    beta = random.randint(-20, 20)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # Noise (light)
    noise = np.random.normal(0, 8, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


# ---------------- WORKER ---------------- #

def process_single_image(args):
    img_path, output_folder = args
    img = cv2.imread(img_path)
    if img is None:
        return 0

    base = os.path.basename(img_path)
    name, ext = os.path.splitext(base)

    count = 0

    # ✅ Save original
    orig_path = os.path.join(output_folder, f"{name}_orig{ext}")
    cv2.imwrite(orig_path, img)
    count += 1

    # Generate augmented
    for i in range(AUG_PER_IMAGE):
        aug = augment_image(img)
        out_name = f"{name}_aug_{i+1}{ext}"
        cv2.imwrite(os.path.join(output_folder, out_name), aug)
        count += 1

    return count


# ---------------- PROCESS ---------------- #

def process_folder(folder_path):
    images = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    ]

    if not images:
        messagebox.showerror("Error", "No supported images found!")
        return

    total_expected = len(images) * TOTAL_IMAGES_PER_INPUT
    output_folder = os.path.join(
        folder_path, f"Augmented_images_{total_expected}"
    )
    os.makedirs(output_folder, exist_ok=True)

    tasks = [(img, output_folder) for img in images]

    total_generated = 0

    # 🚀 PARALLEL EXECUTION
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        for result in executor.map(process_single_image, tasks):
            total_generated += result

    messagebox.showinfo(
        "Completed",
        f"Augmentation done!\nTotal images generated: {total_generated}"
    )


# ---------------- GUI ---------------- #

def select_folder():
    folder = filedialog.askdirectory()
    folder_var.set(folder)


def start_process():
    folder = folder_var.get()
    if not folder:
        messagebox.showwarning("Warning", "Select a folder first")
        return
    process_folder(folder)


root = tk.Tk()
root.title("Fast Image Augmentation Tool")
root.geometry("520x220")

folder_var = tk.StringVar()

tk.Label(root, text="Select Image Folder").pack(pady=10)
tk.Entry(root, textvariable=folder_var, width=65).pack()
tk.Button(root, text="Browse", command=select_folder).pack(pady=5)
tk.Button(root, text="Process", command=start_process).pack(pady=15)

root.mainloop()

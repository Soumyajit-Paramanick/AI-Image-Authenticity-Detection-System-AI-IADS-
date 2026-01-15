# AI Image Authenticity Dataset Builder
# Stable version with modern AI-aware forensic logic
# Extension-based prior stored separately (not used in scoring)
import os
import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image
import concurrent.futures
import threading

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
MAX_WORKERS = 4  # Number of parallel processes

# ================== IMAGE LOADER (MAINTAINS ORIGINAL ACCURACY) ==================

def load_and_normalize_image(image_path, grayscale=False):
    """Original accurate image loader"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, PROCESSING_SIZE, interpolation=cv2.INTER_AREA)
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

# ================== FORENSIC FEATURE FUNCTIONS (ORIGINAL ACCURATE VERSIONS) ==================

def exif_present(image_path):
    """Original EXIF check"""
    try:
        img = Image.open(image_path)
        exif = img._getexif()
        return 0 if exif else 1
    except:
        return 1

def fft_variance(image_path):
    """Original accurate FFT calculation"""
    img = load_and_normalize_image(image_path, grayscale=True)
    if img is None:
        return 0
    f = np.fft.fftshift(np.fft.fft2(img))
    mag = np.log(np.abs(f) + 1)
    return round(float(np.var(mag)), 4)

def noise_std(image_path):
    """Original noise calculation"""
    img = load_and_normalize_image(image_path, grayscale=True)
    if img is None:
        return 0
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    noise = img - blur
    return round(float(np.std(noise)), 4)

def compression_diff(image_path):
    """Original compression calculation"""
    img = load_and_normalize_image(image_path, grayscale=True)
    if img is None:
        return 0
    _, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    dec = cv2.imdecode(enc, 0)
    return round(float(np.mean(np.abs(img - dec))), 4)

def color_bias(image_path):
    """Original color bias calculation"""
    img = load_and_normalize_image(image_path)
    if img is None:
        return 0
    hr = cv2.calcHist([img], [0], None, [256], [0, 256])
    hg = cv2.calcHist([img], [1], None, [256], [0, 256])
    corr = np.corrcoef(hr.flatten(), hg.flatten())[0][1]
    return 1 if corr > 0.98 else 0

def symmetry_score(image_path):
    """Original symmetry calculation"""
    img = load_and_normalize_image(image_path, grayscale=True)
    if img is None:
        return 0
    edges = cv2.Canny(img, 100, 200)
    mid = edges.shape[1] // 2
    left = edges[:, :mid]
    right = np.fliplr(edges[:, mid:])
    return round(float(np.mean(np.abs(left - right))), 4)

# ================== PROCESS SINGLE IMAGE ==================

def process_single_image(file_path, filename):
    """Process a single image and return its features"""
    try:
        row = {
            "image_name": filename,
            "extension_prior": get_extension_prior(filename),
            "exif_missing": exif_present(file_path),
            "fft_variance": fft_variance(file_path),
            "noise_std": noise_std(file_path),
            "compression_diff": compression_diff(file_path),
            "color_bias": color_bias(file_path),
            "symmetry_score": symmetry_score(file_path),
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
        return row
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return None

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

# ================== PROCESSING WITH PROGRESS ==================

def process_folder_threaded(folder_path, progress_callback=None, completion_callback=None):
    """Process folder in a separate thread with parallel processing"""
    try:
        # Get all image files
        image_files = []
        for file in os.listdir(folder_path):
            if file.lower().endswith(VALID_EXTENSIONS):
                image_files.append((os.path.join(folder_path, file), file))
        
        if not image_files:
            if completion_callback:
                completion_callback("Error", "No valid images found.")
            return
        
        total_images = len(image_files)
        data = []
        processed = 0
        
        # Process images in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(process_single_image, file_path, filename): (file_path, filename)
                for file_path, filename in image_files
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_file):
                result = future.result()
                if result is not None:
                    data.append(result)
                
                processed += 1
                if progress_callback:
                    progress = (processed / total_images) * 100
                    progress_callback(progress, processed, total_images)
        
        # Save to CSV
        if data:
            df = pd.DataFrame(data)
            csv_path = os.path.join(folder_path, "forensic_features_dataset.csv")
            df.to_csv(csv_path, index=False)
            
            if completion_callback:
                completion_callback("Success", f"Processed {len(data)} images.\nCSV saved at:\n{csv_path}")
        else:
            if completion_callback:
                completion_callback("Error", "No images could be processed.")
                
    except Exception as e:
        if completion_callback:
            completion_callback("Error", f"Processing failed: {str(e)}")

# ================== GUI WITH PROGRESS BAR ==================

class ProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Image Authenticity Dataset Builder")
        self.root.geometry("680x480")
        self.root.resizable(False, False)
        
        self.setup_ui()
        self.processing = False
        self.stop_requested = False
        
    def setup_ui(self):
        # Title
        tk.Label(self.root, text="AI Image Authenticity Dataset Builder",
                font=("Arial", 14, "bold")).pack(pady=10)
        
        # Folder selection
        tk.Button(self.root, text="Select Image Folder",
                 width=40, command=self.select_folder).pack(pady=5)
        
        self.folder_label = tk.Label(self.root, text="", wraplength=640, fg="blue")
        self.folder_label.pack(pady=5)
        
        # Progress frame
        progress_frame = tk.Frame(self.root)
        progress_frame.pack(pady=10)
        
        self.progress_bar = ttk.Progressbar(progress_frame, length=400, mode='determinate')
        self.progress_bar.pack(side=tk.LEFT, padx=(0, 10))
        
        self.progress_label = tk.Label(progress_frame, text="0%")
        self.progress_label.pack(side=tk.LEFT)
        
        # Status label
        self.status_label = tk.Label(self.root, text="", fg="green")
        self.status_label.pack(pady=5)
        
        # Process button
        self.process_btn = tk.Button(self.root, text="Process Images",
                                    width=40, bg="#4CAF50", fg="white",
                                    command=self.start_processing)
        self.process_btn.pack(pady=15)
        
        # Output label
        tk.Label(self.root, text="Output: forensic_features_dataset.csv",
                font=("Arial", 9, "italic")).pack()
        
        # Stop button (initially disabled)
        self.stop_btn = tk.Button(self.root, text="Stop Processing",
                                 width=40, bg="#f44336", fg="white",
                                 command=self.stop_processing,
                                 state=tk.DISABLED)
        self.stop_btn.pack(pady=5)
        
    def select_folder(self):
        if not self.processing:
            folder = filedialog.askdirectory()
            if folder:
                self.folder_label.config(text=folder)
                self.status_label.config(text="Ready to process", fg="green")
    
    def update_progress(self, progress, processed, total):
        if self.stop_requested:
            return
        self.progress_bar['value'] = progress
        self.progress_label.config(text=f"{processed}/{total} ({progress:.1f}%)")
        self.status_label.config(text=f"Processing... {processed}/{total} images")
        self.root.update_idletasks()
    
    def on_complete(self, title, message):
        self.processing = False
        self.stop_requested = False
        self.process_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        
        if title == "Success":
            self.status_label.config(text="Processing complete!", fg="green")
            messagebox.showinfo(title, message)
        else:
            self.status_label.config(text="Processing failed", fg="red")
            messagebox.showerror(title, message)
        
        # Reset progress
        self.progress_bar['value'] = 0
        self.progress_label.config(text="0%")
    
    def start_processing(self):
        folder = self.folder_label.cget("text")
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("Error", "Please select a valid folder.")
            return
        
        self.processing = True
        self.stop_requested = False
        self.process_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Starting processing...", fg="blue")
        
        # Start processing in separate thread
        thread = threading.Thread(
            target=process_folder_threaded,
            args=(folder, self.update_progress, self.on_complete),
            daemon=True
        )
        thread.start()
    
    def stop_processing(self):
        if self.processing:
            self.stop_requested = True
            self.status_label.config(text="Stopping... Please wait", fg="orange")
            self.stop_btn.config(state=tk.DISABLED)

# ================== MAIN ==================

if __name__ == "__main__":
    root = tk.Tk()
    app = ProcessingApp(root)
    root.mainloop()
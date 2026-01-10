# 📊 Dataset Building Research & Analysis  
## AI Image Authenticity Detection System (AI-IADS)

This document presents the **complete research study, reasoning, and numeric classification rules** behind the dataset creation process for the **AI Image Authenticity Detection System (AI-IADS)**.

The objective of this research is to design an **explainable, forensic, and machine-learning-ready dataset** that helps distinguish between:

- 📸 Camera-captured (real) images  
- 🤖 AI-generated images  
- 🖼️ Digitally created graphics / diagrams  
- ✂️ Edited or screenshot-based images  

This document explains **what each dataset field denotes**, **why it is important**, **what numeric ranges mean**, and **how the dataset is later used for analysis and model training**.

---

## 1. Problem Statement

With the rapid growth of AI image generation tools, it has become increasingly difficult to determine whether an image is real or AI-generated. Traditional approaches relying only on metadata inspection are unreliable because metadata can be removed, altered, or spoofed.

Therefore, a **signal-based forensic approach** is required—one that relies on measurable properties of images rather than trust-based indicators.

---

## 2. Study Design & Methodology

After studying existing work in:
- Digital image forensics
- AI-generated image artifacts
- Compression behavior
- Noise modeling
- Frequency-domain analysis

we designed a **multi-feature forensic pipeline** based on the following principles:

- Do not rely on a single indicator  
- Combine metadata, spatial, frequency, and statistical signals  
- Preserve explainability  
- Support human verification  
- Enable reliable machine learning training  

Each image is processed independently, and all extracted forensic values are stored in a structured **CSV (Excel-compatible) dataset**.

---

## 3. Dataset Structure


Each image produces **one row** in the dataset.  
Each forensic signal is stored as a **separate column**.

### Dataset Columns

- image_name
- exif_missing
- fft_variance
- noise_std
- compression_diff
- color_bias
- symmetry_score
- watermark_text
- image_watermark_present

The dataset is saved as: 
forensic_features_dataset.csv


---

## 4. Feature-by-Feature Explanation with Numeric Thresholds

This section defines **what each field denotes**, **why it is important**, and **how numeric values are interpreted**.

---

### 4.1 image_name

**Meaning:**  
The original filename of the image.

**Importance:**  
- Acts as a unique identifier  
- Maintains traceability between dataset and actual image  
- Enables auditing, manual review, and visualization  

---

### 4.2 exif_missing

**Meaning:**  
Indicates whether camera EXIF metadata is present.

**Values:**
| Value | Meaning |
|-----|--------|
| 0 | EXIF present |
| 1 | EXIF missing |

**Observation:**  
Camera images usually contain EXIF data (ISO, shutter speed, lens). AI-generated images, graphics, and screenshots typically do not.

**Importance:**  
A strong indicator of camera origin, but **not decisive alone**.

---

### 4.3 fft_variance (Frequency Domain Variance)

**Meaning:**  
Measures complexity and randomness of frequency components using FFT.

**Observed Numeric Ranges:**
| Range | Interpretation |
|-----|---------------|
| < 0.95 | Digital graphic / diagram |
| 0.95 – 1.15 | Edited / screenshot |
| 1.15 – 1.45 | AI-generated image |
| > 1.45 | Camera photo |

**Importance:**  
Separates **synthetic smoothness** from **natural real-world textures**.

---

### 4.4 noise_std (Sensor Noise Standard Deviation)

**Meaning:**  
Measures residual noise after Gaussian smoothing.

**Observed Numeric Ranges:**
| Range | Interpretation |
|-----|---------------|
| < 8 | Digital graphic / diagram |
| 8 – 20 | Edited / screenshot |
| 20 – 50 | AI-generated image |
| > 50 | Camera photo |

**Importance:**  
One of the **strongest forensic indicators** of camera-captured images, as real sensor noise is difficult to replicate.

---

### 4.5 compression_diff (Recompression Loss)

**Meaning:**  
Measures information loss after JPEG recompression.

**Observed Numeric Ranges:**
| Range | Interpretation |
|-----|---------------|
| < 4 | Digital graphic |
| 4 – 12 | Edited / screenshot |
| 12 – 35 | AI-generated image |
| > 35 | Camera photo |

**Importance:**  
Reflects hidden entropy and natural image complexity.

---

### 4.6 color_bias

**Meaning:**  
Measures correlation between RGB color channels.

**Values:**
| Value | Interpretation |
|-----|---------------|
| 1 | Strong color bias (diagram / UI) |
| 0 | Natural color variation |

**Importance:**  
Highly effective for separating **digital graphics** from photos and AI images.

---

### 4.7 symmetry_score

**Meaning:**  
Measures structural symmetry between left and right halves of the image.

**Observed Numeric Ranges:**
| Range | Interpretation |
|-----|---------------|
| < 1.5 | AI-generated / graphic |
| 1.5 – 4.0 | Edited / mixed |
| 4.0 – 8.0 | Possible camera image |
| > 8.0 | Strong camera evidence |

**Importance:**  
AI-generated images often introduce symmetric artifacts; real-world scenes are asymmetric.

---

### 4.8 watermark_text

**Meaning:**  
Extracted visible watermark text using OCR.

**Values:**
| Value | Interpretation |
|-----|---------------|
| Empty | No visible text watermark |
| Non-empty | Direct AI evidence |

**Examples:**
- Generated by Gemini
- AI Generated

**Importance:**  
Provides **explicit, explainable evidence** of AI generation.

---

### 4.9 image_watermark_present

**Meaning:**  
Binary detection of non-text (logo/image) watermark.

**Values:**
| Value | Interpretation |
|-----|---------------|
| 0 | No image watermark detected |
| 1 | Image/logo watermark present |

**Importance:**  
Captures watermark cases where OCR cannot detect text.

---

## 5. Final Image Classification Rules (Numeric)

Based on observed feature behavior, images are classified into **four primary categories**.


---

### 📸 Camera-Captured Image
```
exif_missing = 0
noise_std > 50
fft_variance > 1.45
compression_diff > 35
symmetry_score > 8
image_watermark_present = 0
```
---

### 🤖 AI-Generated Image
```
exif_missing = 1
noise_std BETWEEN 20 AND 50
fft_variance BETWEEN 1.15 AND 1.45
symmetry_score < 1.5
image_watermark_present = 1 (optional but strong)
```

---

### 🖼️ Digital Graphic / Diagram


```
exif_missing = 1
noise_std < 8
fft_variance < 0.95
compression_diff < 4
color_bias = 1
```

---

### ✂️ Edited / Screenshot Image
```
exif_missing = 1
noise_std BETWEEN 8 AND 20
fft_variance BETWEEN 0.95 AND 1.15
compression_diff BETWEEN 4 AND 12
image_watermark_present = 0
```

---

## 6. Role of Watermark-Based Classification

From analysis and observation:

- Watermarks provide **direct visual evidence** of AI generation
- Even if other features are ambiguous, watermark presence is decisive
- Watermark highlighting enables human verification

Therefore:
- Watermarked images are explicitly marked
- Saved separately for audit
- Used as high-confidence AI samples during training

---

## 7. Why CSV / Excel Is Used Before Training

Before training ML or DL models, it is essential to:

1. Perform exploratory data analysis (EDA)
2. Validate numeric thresholds
3. Remove mislabeled samples
4. Balance class distribution
5. Analyze feature correlations

CSV format allows:
- Manual inspection
- Filtering
- Label correction
- Collaboration with reviewers

---

## 8. Usage for Machine Learning Training

The dataset is later used to:

1. Normalize features
2. Assign labels using rules + human review
3. Split data into train / test / validation
4. Train ML models (Random Forest, XGBoost)
5. Analyze feature importance
6. Iteratively refine thresholds

This ensures models learn from **forensic evidence**, not guesses.

---

## 9. Conclusion

This research demonstrates that image authenticity can be assessed using a combination of metadata analysis, frequency-domain signals, noise statistics, compression behavior, symmetry analysis, and watermark detection.

By defining **explicit numeric thresholds**, ambiguity is removed and the dataset becomes reproducible, explainable, and ML-ready.

---

### Final Summary

> This study defines a forensic dataset design using measurable image signals and numeric thresholds to reliably classify camera, AI-generated, digital, and edited images, while preparing high-quality data for machine learning training.

⭐ This document forms the theoretical foundation of the AI-IADS dataset-building pipeline.
  

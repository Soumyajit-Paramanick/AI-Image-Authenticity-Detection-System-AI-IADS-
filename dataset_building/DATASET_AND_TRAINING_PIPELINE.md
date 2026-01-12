# 📊 Dataset and Training Pipeline

## AI Image Authenticity Detection System (AI-IADS)

This document describes the dataset sources, dataset construction process, extracted forensic features, feature priority order, and the objective of model training for the **AI Image Authenticity Detection System (AI-IADS)**.

---

## 1. Dataset Sources

To build a large-scale, openly usable dataset, images were collected from publicly available and free sources on Kaggle.

The datasets were selected based on:
- Open availability
- Free usage for research and projects
- Clear separation between real and AI-generated images
- Large image volume suitable for machine learning

### Selected Datasets
- CIFAKE: Real and AI-generated synthetic images  
- AI-generated images vs real images dataset  

📌 **Total images collected:** ~180,000

These datasets provide a diverse mix of:
- Camera-captured photographs
- AI-generated images
- Edited, resized, and recompressed images

---

## 2. Motivation for Dataset Construction

Raw images are **not directly used** for training.

Instead, a **forensic, signal-based approach** is adopted to:
- Avoid black-box learning
- Improve explainability
- Reduce overfitting to dataset artifacts
- Enable human-verifiable decisions

Each image is processed independently to extract measurable forensic signals.

---

## 3. CSV Dataset Generation

A custom pipeline implemented in:

ai_image_dataset_builder.py

is used to analyze each image and generate a structured CSV file:


Each row in the CSV corresponds to **one image**.

---

## 4. Dataset Columns

The following columns are generated:

| Column Name | Description |
|------------|-------------|
| image_name | Original image filename (for traceability only) |
| exif_missing | Whether EXIF metadata is missing (0 = present, 1 = missing) |
| fft_variance | Frequency-domain complexity using FFT |
| noise_std | Residual sensor noise standard deviation |
| compression_diff | Information loss after JPEG recompression |
| color_bias | RGB channel correlation indicating digital graphics |
| symmetry_score | Structural symmetry between left and right halves |
| watermark_text | OCR-extracted visible watermark text (if any) |
| image_watermark_present | Binary detection of image/logo watermark |

📌 **Note:**  
`image_name` is **not used** for model training.  
It is retained only for auditing and manual verification.

---

## 5. Feature Priority Order

Not all features have equal forensic strength.  
Based on empirical observation and image forensics principles, features are prioritized as follows:

### 🔐 Feature Priority (Highest → Lowest)

1. **noise_std**  
   Strong indicator of real camera sensor behavior.

2. **symmetry_score**  
   Detects unnatural structural symmetry common in AI-generated images.

3. **fft_variance**  
   Measures texture richness and frequency randomness.

4. **compression_diff**  
   Reflects hidden entropy and recompression artifacts.

5. **exif_missing**  
   Weak indicator; metadata can be removed or altered.

6. **color_bias**  
   Effective mainly for detecting digital graphics or diagrams.

7. **watermark_text / image_watermark_present**  
   Contextual evidence only.  
   Watermarks do **not** imply AI generation, as many real photos also contain watermarks.

📌 Watermark-related features are **never used as decisive indicators**.

---

## 6. Training Objective

The **primary objective** of AI-IADS is to classify an input image into **one of four categories**:

- 📸 **Camera-captured (real) images**
- 🤖 **AI-generated images**
- 🖼️ **Digitally created graphics / diagrams**
- ✂️ **Edited or screenshot-based images**

This classification is based on **forensic signal behavior**, not on metadata or filenames.

---

## 7. Model Training Philosophy

- Training uses **forensic numeric features**, not raw pixels
- Priority features influence decisions more strongly
- No single feature is treated as decisive
- Ambiguous cases are expected and supported
- The system favors **explainability over blind accuracy**

This ensures robustness against:
- Dataset bias
- Metadata stripping
- Watermark misuse
- Platform recompression

---

## 8. Summary

This dataset and training pipeline enable:

- Explainable AI image authenticity detection
- Multi-class classification grounded in forensic evidence
- Scalable machine learning on large, open datasets
- Transparent and auditable decision-making

The generated CSV dataset forms the foundation for both rule-based analysis and supervised machine learning models in AI-IADS.

---

⭐ This document defines the dataset and training philosophy behind AI-IADS.

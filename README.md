# 📘 AI Image Authenticity Detection System (AI-IADS)

A DevOps-driven, microservice-based system that detects whether an image is real (camera-captured) or AI-generated, using forensic analysis, machine learning, deep learning, and human-in-the-loop verification.

Designed with scalability, explainability, microservices architecture, and CI/CD best practices.

---

## 📌 Table of Contents

1. Introduction  
2. Project Objectives  
3. Overall System Description  
4. High-Level Architecture  
5. Low-Level Architecture (Service-wise)  
6. Database Design  
7. Data Collection Process  
8. Machine Learning & Deep Learning Flow  
9. Non-Functional Requirements  
10. DevOps & Collaboration  
11. Complete Project Flow (End-to-End)  
12. Conclusion  

---

## 1. INTRODUCTION

### 1.1 Purpose

The purpose of this project is to design and implement a scalable, explainable, microservice-based AI system that determines whether an image is real or AI-generated.

The system integrates:
- Image forensic analysis  
- Machine learning and deep learning inference  
- Human-in-the-loop validation  
- Production-ready DevOps practices  

---

### 1.2 Scope

The system:
- Accepts image uploads from users  
- Performs forensic analysis using multiple test cases  
- Uses ML/DL models trained on curated datasets  
- Allows manual and forensic expert verification  
- Produces a final explainable authenticity verdict  
- Is built using microservice architecture  
- Is deployable using modern DevOps practices  

---

## 2. OBJECTIVES OF THE PROJECT

### 2.1 Technical Objectives

- Learn and implement Microservice Architecture
- Work with Spring Boot, Django, and React
- Design the complete ML/DL lifecycle (data → training → deployment)
- Implement human-in-the-loop AI
- Apply DevOps practices (Docker, CI/CD, Kubernetes)
- Build scalable, maintainable, and secure systems

---

### 2.2 Learning Objectives

- Understand distributed system design  
- Learn service-to-service communication  
- Practice REST API design and versioning  
- Implement explainable AI systems  
- Gain experience in collaborative development  
- Apply software engineering best practices  

---

### 2.3 Industry Relevance

This project reflects real-world systems used in:
- Digital forensics  
- Media authenticity verification  
- Content moderation platforms  
- AI governance and compliance tools  

---

## 3. OVERALL SYSTEM DESCRIPTION

### 3.1 User Roles

- End User — Uploads images  
- Reviewer — Performs blind manual labeling  
- Forensic Expert — Final expert verification  
- Admin — Manages system configuration and ML models  

---

### 3.2 Operating Environment

- Web application (browser-based UI)
- Backend microservices (Dockerized)
- Database server (PostgreSQL / MySQL)
- Object storage (Amazon S3 / MinIO)
- Kubernetes cluster (optional)

---

## 4. HIGH-LEVEL ARCHITECTURE

### 4.1 High-Level Architecture Diagram (Logical)

<p align="center">
  <img src="README_images/architecture_diagram.png" alt="Architecture Diagram" width="800"/>
</p>

*Figure: High-level microservice architecture of AI Image Authenticity Detection System (AI-IADS)*

---

## 5. LOW-LEVEL ARCHITECTURE (SERVICE-WISE)

### 5.1 Image Upload Service (Spring Boot)

Responsibilities:
- Accept image uploads
- Validate file format and size
- Store images in object storage
- Generate and manage unique image IDs

Endpoints:
- POST /images/upload  
- GET  /images/{id}  

---

### 5.2 Forensic Feature Extraction Service (Django)

Responsibilities:
- Metadata analysis  
- FFT (frequency domain) analysis  
- Noise pattern analysis  
- Color distribution analysis  
- Compression artifact detection  
- Symmetry and structural consistency checks  

Sample Output:
{
  image_id: img123  
  fft_variance: 0.83  
  noise_std: 1.2  
  color_bias: true  
}

---

### 5.3 ML Inference Service (Django)

Responsibilities:
- Load approved ML/DL models
- Perform inference on extracted features
- Return prediction with confidence score
- Support model versioning

Important:
- No training occurs in this service  
- Read-only inference for production safety  

---

### 5.4 Manual Review Service (Spring Boot)

Responsibilities:
- Blind human labeling
- Forensic expert validation
- Conflict resolution
- Final verdict approval

---

### 5.5 API Gateway (Spring Boot)

Responsibilities:
- JWT-based authentication
- Request routing to microservices
- Rate limiting
- Centralized logging and monitoring

---

## 6. DATABASE DESIGN

### 6.1 Database Choice

PostgreSQL (Recommended)

Reasons:
- Strong consistency guarantees  
- Excellent JSON support  
- Widely adopted in microservice architectures  

(MySQL is supported but PostgreSQL is preferred.)

---

### 6.2 Core Tables

- images  
- features  
- ml_predictions  
- human_labels  
- final_decisions  
- audit_logs  
- model_versions  

---

## 7. DATA COLLECTION PROCESS

### 7.1 Data Sources

- Kaggle — AI vs Real datasets  
- Research Institutes — GAN detection datasets  
- Public Benchmarks — FaceForensics++, CelebDF  
- Raw Camera Data — Mobile and DSLR images  
- Generated Data — Stable Diffusion, DALL·E  

---

### 7.2 Dataset Handling Strategy

- Store raw images in object storage  
- Store metadata and labels in the database  
- Maintain dataset versioning  
- Freeze dataset snapshots before training  

---

## 8. MACHINE LEARNING & DEEP LEARNING FLOW

### 8.1 Training (Offline)

- Random Forest  
- XGBoost  
- ResNet (future phase)  
- EfficientNet (future phase)  

Training is performed offline on local or GPU-enabled systems.

---

### 8.2 Inference (Online)

- Deployed as a Django microservice  
- Loads approved model versions  
- Produces prediction with confidence score  

---

## 9. NON-FUNCTIONAL REQUIREMENTS

Performance:
- Inference latency < 500 ms  
- Asynchronous processing for heavy tasks  

Scalability:
- Horizontal scaling via Kubernetes  
- Stateless microservices  

Security:
- JWT authentication  
- Secure file uploads  
- Audit logging  

Maintainability:
- Clean service boundaries  
- Versioned APIs  
- CI/CD pipelines  

---

## 10. DEVOPS & COLLABORATION

DevOps Practices:
- Dockerized microservices  
- GitHub Actions CI/CD pipelines  
- Kubernetes-ready deployment  
- Environment-based configurations  

Collaboration:
- Git branching strategy  
- Code reviews  
- Shared API contracts  
- Centralized documentation  

---

## 11. COMPLETE PROJECT FLOW (END-TO-END)

<p align="center">
  <img src="README_images/flow_diagram.png" alt="Project Flow Diagram" width="800"/>
</p>

*Figure: End-to-end flow from image upload to final authenticity verdict*

---

## 12. CONCLUSION

This project demonstrates:
- Strong understanding of microservice architecture  
- End-to-end AI system lifecycle implementation  
- Integration of Spring Boot, Django, ML/DL, and DevOps  
- Explainable and responsible AI design  
- Production-ready engineering mindset  

If you find this project useful, consider starring the repository ⭐

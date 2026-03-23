# 🛍️ Product Image Classification

An end-to-end Computer Vision pipeline to classify over **100,000 marketplace images** into 42 distinct categories. This project transitions from raw data exploration to a live-deployed web application.

### 🛠️ Tech Stack
* **Language:** Python
* **Deep Learning:** TensorFlow, Keras
* **Architecture:** MobileNetV2 (Transfer Learning)
* **Deployment:** Streamlit, GitHub
* **Tools:** Google Colab (GPU), KaggleHub, Matplotlib, Pandas

---

### 📊 Key Performance Metrics
| Metric | Value |
| :--- | :--- |
| **Initial Baseline Acc** | 67.3% (Frozen Base) |
| **Final Validation Acc** | **72.4%** (Fine-tuned) |
| **Training Time** | ~10 mins/epoch (T4 GPU) |
| **Classes** | 42 Anonymized Categories |

---

### 🧩 The Problem & Challenges
* **Data Scale:** Managing a **10GB dataset** with nested directory structures within a cloud environment.
* **Class Imbalance:** Identifying that specific categories (like ID 33) had **5x less data** than others, causing biased predictions.
* **Desensitized Labels:** Working with numeric IDs instead of human-readable names, requiring visual forensic analysis to map categories.

---

### 💡 Outcomes & Solutions
1.  **Transfer Learning:** Implemented MobileNetV2 to leverage pre-trained ImageNet weights, drastically reducing training time.
2.  **Strategic Fine-Tuning:** Unfroze the base model with a **$10^{-5}$ learning rate** to capture fine-grained product details (e.g., distinguishing between electronics and stationery).
3.  **Handling Imbalance:** Applied **Class Weighting** and analyzed distribution graphs to ensure the model didn't just guess the "majority" classes.
4.  **Live Deployment:** Built a **Streamlit Dashboard** allowing users to upload real-time images for instant classification.

---

### 🚀 How to Run
1. Clone the repo.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the app: `streamlit run app.py`.

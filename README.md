# 🛍️ Product Image Classification

An end-to-end Computer Vision pipeline to classify over **100,000 marketplace images** into 42 distinct categories. This project transitions from raw data exploration to a live-deployed web application.
### Live demo: https://jiya-image-classifier.streamlit.app/

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
graph TD
    %% Define colors and styles for clarity
    classDef data fill:#E1F5FE,stroke:#0277BD,stroke-width:2px;
    classDef preprocess fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px;
    classDef model_base fill:#FFFDE7,stroke:#FBC02D,stroke-width:2px;
    classDef model_head fill:#FFF3E0,stroke:#EF6C00,stroke-width:2px;
    classDef output fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px;
    classDef deploy fill:#FFEBEE,stroke:#C62828,stroke-width:2px;

    %% 1. Data Input & Exploration (Light Blue)
    subgraph Data_Pipeline [1. Data Pipeline]
        Raw_Images(100k+ Raw Shopee Images) --> KaggleHub_Paths(KaggleHub API Path Parsing);
        KaggleHub_Paths --> Nested_Folders(Nested Directory Structuring);
        Nested_Folders --> Final_Paths(Final Cleaned Image Paths);
        class Raw_Images,KaggleHub_Paths,Nested_Folders,Final_Paths data;
    end

    %% 2. Preprocessing (Light Green)
    subgraph Preprocessing_Pipeline [2. Preprocessing]
        Final_Paths --> ImageDataGen(Keras ImageDataGenerator);
        ImageDataGen --> Resize(Resize to 224x224x3);
        Resize --> Rescale(Rescale Pixels 1/255);
        Rescale --> Augment(Data Augmentation - Zoom/Flip);
        class ImageDataGen,Resize,Rescale,Augment preprocess;
    end

    %% 3. Model Architecture: Feature Extraction (Yellow)
    subgraph Feature_Extraction [3. Feature Extraction - Pre-trained Base]
        Augment --> Input_Layer(Input Layer: 224x224x3);
        Input_Layer --> MobileNetV2(MobileNetV2 Backbone - ImageNet Weights);
        MobileNetV2 --> Frozen_Base(Frozen Initial Layers);
        Frozen_Base --> Depthwise_Conv(Depthwise Separable Convolutions);
        class Input_Layer,MobileNetV2,Frozen_Base,Depthwise_Conv model_base;
    end

    %% 4. Model Architecture: Classification Head (Orange)
    subgraph Classification_Head [4. Classification Head - Custom Logic]
        Depthwise_Conv --> GAP_Layer(Global Average Pooling GAP);
        GAP_Layer --> Dropout(Dropout Layer 0.2);
        Dropout --> Dense_Out(Dense Output Layer 42 Units);
        Dense_Out --> Softmax(Softmax Activation);
        class GAP_Layer,Dropout,Dense_Out,Softmax model_head;
    end

    %% 5. Training, Optimization & Outcome (Purple)
    subgraph Training_Outcome [5. Training & Optimization]
        Softmax --> Categorical_CE(Categorical Crossentropy Loss);
        Categorical_CE --> Adam_Opt(Adam Optimizer);
        Adam_Opt --> Fine_Tuning(Fine-Tuning LR: 1e-5);
        Fine_Tuning --> History(Training History / Epochs);
        History --> Final_Acc(Final Validation Acc: 72.4%);
        class Categorical_CE,Adam_Opt,Fine_Tuning,History,Final_Acc output;
    end

    %% 6. Deployment (Light Red)
    subgraph Deployment_Pipeline [6. Deployment]
        Final_Acc --> Save_Model(Save Model as .keras);
        Save_Model --> GitHub_Repo(GitHub Repository);
        GitHub_Repo --> requirements(requirements.txt);
        requirements --> app_py(app.py - Streamlit);
        app_py --> Streamlit_Cloud(Streamlit Community Cloud Deployment);
        Streamlit_Cloud --> User_Interface(Live Live UI App);
        class Save_Model,GitHub_Repo,requirements,app_py,Streamlit_Cloud,User_Interface deploy;
    end

---

### 🚀 How to Run
1. Clone the repo.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the app: `streamlit run app.py`.

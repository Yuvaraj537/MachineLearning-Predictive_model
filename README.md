# 🤖 Machine Learning & Deep Learning Overview

Welcome to this repository!  
This document provides an overview of **Machine Learning (ML)**, **Deep Learning (DL)**, and **Natural Language Processing (NLP)** concepts.

---

## 📌 1. Machine Learning Types

Machine Learning is broadly divided into:

1. **🎯 Supervised Learning**  
   - Training data has input features **X** and output labels **Y**.  
   - The model learns a mapping function `f(X) → Y`.  
   - ✅ Example: Predicting house price, disease detection.  

2. **🌀 Unsupervised Learning**  
   - Data has **no labels**.  
   - The model finds hidden patterns or groups.  
   - ✅ Example: Customer segmentation, anomaly detection.  

3. **🎮 Reinforcement Learning (RL)** *(optional extension)*  
   - Agent learns by interacting with an environment.  
   - ✅ Example: Self-driving cars, game playing bots.  

---

## 🎯 2. Supervised Learning

### 🔹 📂 Classification Models  
**Definition:** Classification is the task of predicting a **category or class label** from input data.  
- Output → **Discrete values** (e.g., "Yes/No", "Spam/Not Spam").  
- ✅ Examples: Spam detection, disease classification.  
- 📊 Popular Algorithms:
  - Logistic Regression  
  - Decision Trees  
  - Random Forest  
  - Support Vector Machines (SVM)  
  - K-Nearest Neighbors (KNN)  
  - Naïve Bayes  
  - Gradient Boosting (XGBoost, LightGBM)  

### 🔹 📈 Regression Models  
**Definition:** Regression is the task of predicting a **continuous numeric value** from input data.  
- Output → **Real numbers** (e.g., price, temperature, age).  
- ✅ Examples: Predicting house price, stock value.  
- 📊 Popular Algorithms:
  - Linear Regression  
  - Polynomial Regression  
  - Decision Tree Regressor  
  - Random Forest Regressor  
  - Support Vector Regressor (SVR)  
  - Ridge & Lasso Regression  
  - Gradient Boosting Regressor  

---

## 🧩 3. Unsupervised Learning

### 🔹 🔗 Clustering  
**Definition:** Clustering is the process of **grouping similar data points** into clusters without predefined labels.  
- ✅ Example: Customer segmentation in marketing.  
- 📊 Types of Clustering:
  - **K-Means** – partitions data into *k* clusters.  
  - **Hierarchical Clustering** – builds a tree of clusters.  
  - **DBSCAN** – density-based clustering that detects arbitrary shaped clusters.  

### 🔹 🔻 Dimensionality Reduction  
**Definition:** Dimensionality Reduction is the process of **reducing the number of features** in a dataset while preserving important information.  
- ✅ Example: Reducing image pixels/features for faster training.  
- 📊 Types of Dimensionality Reduction:
  - **Principal Component Analysis (PCA)** – transforms features into principal components.  
  - **t-SNE** – useful for visualization in 2D/3D.  
  - **Autoencoders** – neural network-based feature compression.  

---

## 🤖 4. Deep Learning

Deep Learning is a subset of ML using **neural networks**.  

### 🔹 🧠 Artificial Neural Network (ANN)  
- Basic form of deep learning model.  
- Layers: **Input → Hidden → Output**.  
- ✅ Used for: General classification/prediction tasks.  

### 🔹 🖼️ Convolutional Neural Network (CNN)  
- Specialized for **image & spatial data**.  
- Uses **convolutional filters** to detect patterns.  
- ✅ Applications: Image recognition, object detection.  

### 🔹 ⏳ Recurrent Neural Network (RNN) & LSTM  
- Specialized for **sequence data**.  
- **RNN:** Handles short-term memory but struggles with long dependencies.  
- **LSTM (Long Short-Term Memory):** Solves long-term dependency issues.  
- ✅ Applications: Time-series forecasting, speech recognition, text prediction.  

---

## 📝 5. Natural Language Processing (NLP)

NLP bridges **computers and human language**.  

### 🔹 ⚙️ Text Processing Steps
1. ✂️ Tokenization – splitting text into words/sentences.  
2. 🗑️ Stopword Removal – removing common words (is, the, a).  
3. 🔄 Stemming/Lemmatization – reducing words to root form.  
4. 🔢 Vectorization – converting text into numbers (Bag-of-Words, TF-IDF, Word2Vec, BERT).  

### 🔹 📂 NLP Tasks
- 😀 Sentiment Analysis  
- 📨 Text Classification (spam detection)  
- 🏷️ Named Entity Recognition (NER)  
- 🌍 Machine Translation  
- ✍️ Text Summarization  
- 🤖 Chatbots & Question Answering  

### 🔹 🔣 Common Symbols in NLP
- `w` → word  
- `t` → token  
- `d` → document  
- `V` → vocabulary  
- `|D|` → total number of documents  

---

## ✅ Summary

- **ML** → Supervised & Unsupervised.  
- **Supervised** → Classification & Regression (📂 categories vs 📈 numbers).  
- **Unsupervised** → Clustering (🔗 groups) & Dimensionality Reduction (🔻 feature reduction).  
- **DL** → ANN 🧠, CNN 🖼️, LSTM ⏳.  
- **NLP** → Text processing ⚙️ & applications 📂.  

---

📌 *This README is designed for learners, interview prep, and quick revision.*

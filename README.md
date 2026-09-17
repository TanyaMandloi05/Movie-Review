# 🎬 Movie Review Sentiment Analysis

A Machine Learning project that predicts whether a movie review is **Positive** or **Negative** using an **RNN (Recurrent Neural Network)** and **TF-IDF**.

The trained model is deployed as an interactive web application using **Streamlit**.

---

## 📌 Project Overview

This project takes a movie review as input and predicts its sentiment.

### Example

**Input:**
> This movie was absolutely fantastic. The acting was brilliant and the story was amazing.

**Output:**
> 😊 Positive Review

The application allows users to enter their own movie reviews and get a sentiment prediction instantly.

---

## 🚀 Features

- 📝 Accepts movie reviews as user input
- 🧹 Performs text preprocessing
- 🔤 Converts text into TF-IDF features
- 🧠 Uses an RNN for sentiment classification
- 📊 Predicts Positive or Negative sentiment
- 🌐 Interactive Streamlit web interface
- 💾 Uses saved trained model and TF-IDF vectorizer

---

## 🛠️ Technologies Used

- Python
- PyTorch
- Scikit-learn
- NLTK
- Pandas
- NumPy
- Joblib
- Streamlit

---

## 🧠 Machine Learning Pipeline

The project follows this pipeline:

```text
Movie Review
     ↓
Text Preprocessing
     ↓
TF-IDF Vectorization
     ↓
RNN Model
     ↓
Sigmoid
     ↓
Positive / Negative

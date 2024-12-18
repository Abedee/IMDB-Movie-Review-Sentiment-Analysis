# IMDB Movie Review Sentiment Analysis

This repository contains a **Sentiment Analysis** model for classifying IMDB movie reviews as either **positive** or **negative**. The project uses a **Simple RNN (Recurrent Neural Network)** trained on the IMDB dataset, and provides a user-friendly interface built with **Streamlit**.

---

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Installation and Setup](#installation-and-setup)
5. [How It Works](#how-it-works)
6. [Model Details](#model-details)
7. [Streamlit App](#streamlit-app)
8. [Usage](#usage)
9. [Acknowledgments](#acknowledgments)

---

## Overview

This project demonstrates how to perform **sentiment analysis** on movie reviews from the IMDB dataset using a pre-trained RNN model. Users can input a movie review, and the model predicts whether the sentiment is **positive** or **negative**. The application also provides the prediction score for better insight into the model's confidence.

---

## Features
- **Pre-trained Sentiment Analysis Model**: Uses a simple RNN trained on the IMDB dataset.
- **Real-Time Predictions**: Classify sentiments of user-provided text inputs.
- **Streamlit Interface**: An easy-to-use GUI for interacting with the model.
- **Review Decoding**: Converts encoded reviews back into human-readable text.

---

## Technologies Used
- **Python**: Programming language.
- **TensorFlow/Keras**: For building and loading the sentiment analysis model.
- **Streamlit**: To create the interactive web app.
- **NumPy**: For numerical computations.

---

## Installation and Setup

### Prerequisites
Ensure you have the following installed on your system:
- Python (>=3.8)
- pip (Python package manager)

### Steps to Run the Project
1. Clone the repository:
    ```bash
    git clone https://github.com/your_username/imdb_sentiment_analysis.git
    cd imdb_sentiment_analysis
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the pre-trained model (`simple_rnn_imdb.h5`) and place it in the project directory.

4. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

---

## How It Works

### Data Preparation
- The **IMDB dataset** contains movie reviews with labels indicating whether they are positive or negative.
- Each review is preprocessed to be tokenized into integers using the dataset's built-in word index.
- Reviews are padded or truncated to a fixed length of 500 words to ensure uniform input size.

### Model Details
- The model is a **Simple RNN** trained to classify the sentiment of reviews.
- It was saved as a pre-trained Keras model (`simple_rnn_imdb.h5`) for reuse in this project.

### Streamlit Integration
- The app provides a text box for user input and a button to classify the sentiment of the review.
- The backend pre-processes the input, feeds it to the model, and returns the sentiment and prediction score.

---

## Streamlit App

### Interface
- **Title**: "IMDB Movie Review Sentiment Analysis"
- **User Input**: A text area where users can type their movie reviews.
- **Classification Button**: When clicked, the app predicts the sentiment of the input review.
- **Output**:
  - Sentiment (Positive/Negative)
  - Prediction Score (e.g., confidence level)

### Helper Functions
#### 1. Decode Reviews
Converts encoded reviews back into human-readable text using the IMDB word index.
```python
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])
```

#### 2. Preprocess User Input
Converts user-provided text into a padded sequence suitable for the model.
```python
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review
```

---

## Usage
1. Launch the Streamlit app using `streamlit run app.py`.
2. Enter a movie review in the provided text box.
3. Click the **Classify** button to get the sentiment and prediction score.

Example:
- Input: *"This movie was absolutely amazing! Loved the story and the acting."
- Output:
  - Sentiment: Positive
  - Prediction Score: 0.89

---

## Acknowledgments
- **IMDB Dataset**: Provided by TensorFlow/Keras.
- **Streamlit**: For creating an intuitive and user-friendly GUI.
- **TensorFlow/Keras Documentation**: For guidance on model loading and implementation.

Feel free to fork, modify, and contribute to this project! :smile:

---

### License
This project is licensed under the [MIT License](LICENSE).

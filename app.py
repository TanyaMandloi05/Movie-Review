import streamlit as st
import torch
import torch.nn as nn
import joblib
import re
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# --------------------- NLTK SETUP ---------------------

# Download the NLTK resources required for preprocessing
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")


# --------------------- TEXT PREPROCESSING ---------------------

def remove_url(text):
    # Remove URLs starting with http
    text = re.sub(r"http\S+", "", text)
    return text


def remove_html(text):
    # Remove HTML tags such as <br>
    text = re.sub(r"<.*?>", "", text)
    return text


def remove_punctutation(text):
    # Keep only letters, numbers and spaces
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    return text


def remove_stopwords(text):
    # Split the text into individual words
    tokens = word_tokenize(text)

    # Get the English stopword list
    stop_words = stopwords.words("english")

    # Same stopword-removal logic used in the notebook
    for word in tokens:
        if word in stop_words:
            text = text.replace(word, "")

    return text


def stemming(text):
    # Create a Porter stemmer
    ps = PorterStemmer()

    # Store the stemmed words
    stemmed_words = []

    # Split text into individual words
    tokens = word_tokenize(text)

    # Stem each word
    for token in tokens:
        stemmed_token = ps.stem(token)
        stemmed_words.append(stemmed_token)

    # Join the stemmed words back into one sentence
    return " ".join(stemmed_words)


def preprocess_text(text):

    # Convert text to lowercase
    text = text.lower()

    # Remove URLs
    text = remove_url(text)

    # Remove HTML tags
    text = remove_html(text)

    # Remove punctuation
    text = remove_punctutation(text)

    # Remove stopwords
    text = remove_stopwords(text)

    # Apply stemming
    text = stemming(text)

    return text


# --------------------- RNN MODEL ---------------------

class RNN(nn.Module):

    def __init__(self, input_size, hidden_size=128, num_layers=1):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Same RNN architecture used during training
        self.rnn = nn.RNN(
            input_size,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )

        # Convert 128 hidden features into one output
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):

        # Pass input through the RNN
        out, _ = self.rnn(x)

        # Take the output from the last timestep
        out = self.fc(out[:, -1, :])

        return out


# --------------------- LOAD TF-IDF ---------------------

# Load the exact TF-IDF vectorizer fitted during training
tf = joblib.load("tfidf.pkl")


# --------------------- LOAD TRAINED MODEL ---------------------

# Your TF-IDF has 5000 features
model = RNN(input_size=5000)

# Load the trained model weights
model.load_state_dict(
    torch.load("rnn_model.pth", map_location="cpu")
)

# Put the model into evaluation mode
model.eval()


# --------------------- STREAMLIT UI ---------------------

st.title("🎬 Movie Review Sentiment Analysis")

st.write(
    "Enter a movie review and the model will predict "
    "whether it is Positive or Negative."
)

# Text box for the user
review = st.text_area("Enter your review:")


# --------------------- PREDICTION ---------------------

if st.button("Predict Sentiment"):

    # Check whether the user entered anything
    if review.strip() == "":
        st.warning("Please enter a review.")

    else:

        # Apply EXACTLY the same preprocessing used during training
        processed_review = preprocess_text(review)

        # Convert the processed review into TF-IDF features
        review_tfidf = tf.transform([processed_review])

        # Convert sparse matrix into NumPy array
        review_array = review_tfidf.toarray()

        # Convert NumPy array into PyTorch tensor
        review_tensor = torch.tensor(
            review_array,
            dtype=torch.float32
        )

        # RNN expects:
        # (batch_size, sequence_length, input_size)
        #
        # Current shape:
        # (1, 5000)
        #
        # After unsqueeze:
        # (1, 1, 5000)
        review_tensor = review_tensor.unsqueeze(1)

        # We are predicting, so gradients are not required
        with torch.no_grad():

            # Get raw output from the RNN
            output = model(review_tensor)

            # Apply sigmoid to convert output into probability
            probability = torch.sigmoid(
                output.squeeze()
            ).item()

        # Classify using 0.5 threshold
        if probability >= 0.5:

            st.success("😊 Positive Review")

        else:

            st.error("😞 Negative Review")

        # Display probability
        st.write(
            f"Prediction probability: {probability:.2f}"
        )
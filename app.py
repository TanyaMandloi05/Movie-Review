import streamlit as st
import torch
import torch.nn as nn
import joblib
import re
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# --------------------- PAGE CONFIG ---------------------

st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="expanded"
)


# --------------------- CUSTOM STYLING ---------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Overall app background */
.stApp {
    background: #14121f;
}

/* Hide default streamlit branding clutter */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main title */
.hero {
    text-align: center;
    padding: 1.4rem 0 0.4rem 0;
}

.big-title {
    font-family: 'Poppins', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    line-height: 1.15;
    color: #f2f2f7;
    margin-bottom: 0.5rem;
}

.subtitle {
    color: #b9bde0;
    font-size: 1.05rem;
    max-width: 520px;
    margin: 0 auto 1.6rem auto;
    line-height: 1.5;
}

/* Card container (native Streamlit bordered container) */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255, 255, 255, 0.045);
    border-radius: 18px !important;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
    margin-bottom: 1.2rem;
}

div[data-testid="stVerticalBlockBorderWrapper"] > div {
    border-radius: 18px !important;
}

.card-heading {
    font-family: 'Poppins', sans-serif;
    color: #ffffff;
    font-size: 1.15rem;
    font-weight: 700;
    margin-bottom: 0.9rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Text area */
.stTextArea textarea {
    background-color: #1e1b2e !important;
    color: #ffffff !important;
    border-radius: 14px !important;
    border: 1px solid rgba(255, 255, 255, 0.16) !important;
    font-size: 1rem !important;
    line-height: 1.55 !important;
    padding: 1rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.stTextArea textarea:focus {
    border-color: #ff8c6a !important;
    box-shadow: 0 0 0 3px rgba(255, 138, 106, 0.18) !important;
}

.stTextArea textarea::placeholder {
    color: rgba(255, 255, 255, 0.4) !important;
}

/* Buttons */
.stButton > button {
    border-radius: 12px;
    border: none;
    font-weight: 700;
    font-size: 0.95rem;
    padding: 0.65rem 1rem;
    transition: all 0.2s ease-in-out;
    letter-spacing: 0.01em;
}

.stButton > button:first-child {
    background: #4f46e5;
    color: white;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(79, 70, 229, 0.35);
}

.stButton > button:active {
    transform: translateY(0px);
}

/* Secondary (clear) button look */
div[data-testid="column"]:nth-of-type(2) .stButton > button {
    background: rgba(255, 255, 255, 0.08);
    color: #e6e6fa;
    border: 1px solid rgba(255, 255, 255, 0.18);
    box-shadow: none;
}

div[data-testid="column"]:nth-of-type(2) .stButton > button:hover {
    background: rgba(255, 255, 255, 0.14);
    box-shadow: none;
}

/* Metric styling */
[data-testid="stMetricValue"] {
    color: #a5b4fc;
    font-weight: 800;
    font-family: 'Poppins', sans-serif;
}

[data-testid="stMetricLabel"] {
    color: #b9bde0;
}

/* Success / error boxes */
div[data-testid="stAlert"] {
    border-radius: 14px;
    font-size: 1.1rem;
    font-weight: 600;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

/* Progress bar */
.stProgress > div > div > div > div {
    background: #6366f1;
    border-radius: 999px;
}

.stProgress > div > div > div {
    background-color: rgba(255, 255, 255, 0.1);
    border-radius: 999px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

[data-testid="stSidebar"] * {
    color: #e6e6fa !important;
}

[data-testid="stSidebar"] h3 {
    font-family: 'Poppins', sans-serif;
    font-weight: 700 !important;
}

/* Expander */
.streamlit-expanderHeader {
    background-color: rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    color: #ffffff !important;
    font-weight: 600;
}

/* Divider */
hr {
    border-color: rgba(255, 255, 255, 0.15) !important;
}

/* Code block inside expander */
.stCodeBlock {
    border-radius: 12px !important;
}

</style>
""", unsafe_allow_html=True)


# --------------------- NLTK SETUP ---------------------

@st.cache_resource
def setup_nltk():
    nltk.download("punkt")
    nltk.download("punkt_tab")
    nltk.download("stopwords")

setup_nltk()


# --------------------- TEXT PREPROCESSING ---------------------

def remove_url(text):
    text = re.sub(r"http\S+", "", text)
    return text


def remove_html(text):
    text = re.sub(r"<.*?>", "", text)
    return text


def remove_punctutation(text):
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    return text


def remove_stopwords(text):
    tokens = word_tokenize(text)
    stop_words = stopwords.words("english")

    for word in tokens:
        if word in stop_words:
            text = text.replace(word, "")

    return text


def stemming(text):
    ps = PorterStemmer()
    stemmed_words = []
    tokens = word_tokenize(text)

    for token in tokens:
        stemmed_token = ps.stem(token)
        stemmed_words.append(stemmed_token)

    return " ".join(stemmed_words)


def preprocess_text(text):
    text = text.lower()
    text = remove_url(text)
    text = remove_html(text)
    text = remove_punctutation(text)
    text = remove_stopwords(text)
    text = stemming(text)
    return text


# --------------------- RNN MODEL ---------------------

class RNN(nn.Module):

    def __init__(self, input_size, hidden_size=128, num_layers=1):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out


# --------------------- LOAD MODEL + VECTORIZER (cached) ---------------------

@st.cache_resource
def load_model_and_vectorizer():
    tf = joblib.load("tfidf.pkl")

    model = RNN(input_size=5000)
    model.load_state_dict(
        torch.load("rnn_model.pth", map_location="cpu")
    )
    model.eval()

    return tf, model

tf, model = load_model_and_vectorizer()


# --------------------- SIDEBAR ---------------------

with st.sidebar:
    st.markdown("### 🎬 About")
    st.write(
        "This app uses a simple RNN trained on movie reviews "
        "to predict whether a review is positive or negative."
    )
    st.markdown("**How it works:**")
    st.markdown(
        "- Text is cleaned and stemmed\n"
        "- Converted into TF-IDF features\n"
        "- Fed into an RNN for classification"
    )
    st.divider()
    st.caption("Built with Streamlit + PyTorch")


# --------------------- MAIN UI ---------------------

st.markdown(
    '''
    <div class="hero">
        <div class="big-title">🎬 Movie Review Sentiment Analysis</div>
        <div class="subtitle">Enter a movie review below and the model will predict
        whether it's <b>Positive</b> or <b>Negative</b>, along with a confidence score.</div>
    </div>
    ''',
    unsafe_allow_html=True
)

with st.container(border=True):
    st.markdown('<div class="card-heading">✍️ Your Review</div>', unsafe_allow_html=True)

    review = st.text_area(
        "Your review",
        placeholder="Type or paste a movie review here...",
        height=160,
        label_visibility="collapsed"
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        predict_clicked = st.button("🔍 Predict Sentiment", use_container_width=True)
    with col2:
        clear_clicked = st.button("🗑️ Clear", use_container_width=True)

if clear_clicked:
    st.rerun()


# --------------------- PREDICTION ---------------------

if predict_clicked:

    if review.strip() == "":
        st.warning("Please enter a review before predicting.")

    else:
        with st.spinner("Analyzing review..."):

            processed_review = preprocess_text(review)
            review_tfidf = tf.transform([processed_review])
            review_array = review_tfidf.toarray()

            review_tensor = torch.tensor(
                review_array,
                dtype=torch.float32
            )

            # RNN expects (batch_size, sequence_length, input_size)
            review_tensor = review_tensor.unsqueeze(1)

            with torch.no_grad():
                output = model(review_tensor)
                probability = torch.sigmoid(output.squeeze()).item()

        with st.container(border=True):
            st.markdown('<div class="card-heading">📊 Result</div>', unsafe_allow_html=True)

            result_col, score_col = st.columns([1, 1])

            with result_col:
                if probability >= 0.5:
                    st.success("😊 Positive Review")
                else:
                    st.error("😞 Negative Review")

            with score_col:
                st.metric("Confidence", f"{probability * 100:.1f}%")

            st.progress(probability)

            with st.expander("See processed text"):
                st.code(processed_review, language="text")
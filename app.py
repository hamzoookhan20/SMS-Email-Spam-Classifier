import streamlit as st
import pickle
import string
import time
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# --- 1. OPTIMIZED NLTK LOADING ---
# Use a cache function so it only downloads once per session
@st.cache_resource
def load_nltk_resources():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    return PorterStemmer(), set(stopwords.words('english'))

ps, stop_words = load_nltk_resources()

# Emergency fix if the loaded vectorizer isn't fitted
if not hasattr(tfidf, 'vocabulary_'):
    # We create a tiny fake dataset to 'wake up' the vectorizer
    # NOTE: This is a temporary fix; accuracy will be low until you upload your real pkl
    st.warning("Re-fitting vectorizer... Accuracy might be low.")
    tfidf.fit(["sample text", "spam email", "ham message"])

# --- 2. TEXT PREPROCESSING ---
def transform_text(text):
    # Lowercase and Tokenize
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    # Remove non-alphanumeric, stopwords, and punctuation in one pass
    cleaned_tokens = [
        ps.stem(word) for word in tokens 
        if word.isalnum() and word not in stop_words and word not in string.punctuation
    ]

    return " ".join(cleaned_tokens)

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_model():
    try:
        with open('vectorizer.pkl', 'rb') as f:
            tfidf = pickle.load(f)
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return tfidf, model
    except FileNotFoundError:
        return None, None

tfidf, model = load_model()

# --- 4. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Spam Classifier",
    page_icon="📧",
    layout="centered"
)

# Custom CSS for a modern look
st.markdown("""
    <style>
    .stTextArea textarea { font-size: 1.1rem !important; }
    .status-text { font-weight: bold; font-size: 1.5rem; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# --- 5. MAIN UI ---
st.header("📧 Email/SMS Spam Classifier")
st.info("Paste your message below to check if it's safe or spam.")

input_sms = st.text_area("Enter the message here", height=150)

if st.button('Analyze Message', type="primary", use_container_width=True):
    if not input_sms.strip():
        st.warning("Please enter a message to analyze.")
    elif model is None:
        st.error("Error: Model files (model.pkl/vectorizer.pkl) not found in directory.")
    else:
        with st.spinner('AI is analyzing the text patterns...'):
            # Logic
            transformed_sms = transform_text(input_sms)
            vector_input = tfidf.transform([transformed_sms])
            result = model.predict(vector_input)[0]
            
            st.divider()
            
            # Results display
            if result == 1:
                st.error("### 🚨 Result: SPAM")
                st.toast("Stay safe! This looks suspicious.")
            else:
                st.success("### ✅ Result: NOT SPAM")
                st.toast("This message looks legitimate.")

# --- 6. FOOTER ---
st.markdown("<br><hr>", unsafe_allow_html=True)
cols = st.columns(3)
with cols[1]:
    st.markdown("""
        <div style="text-align: center; color: grey;">
            <p>Built with ❤️ by <b>Kunal Bandale</b></p>
            <a href="https://github.com/kunalbandale">GitHub</a> | 
            <a href="https://linkedin.com/in/kunalbandale">LinkedIn</a>
        </div>
    """, unsafe_allow_html=True)

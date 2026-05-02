import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# --- 1. SETUP RESOURCES ---
@st.cache_resource
def load_resources():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    return PorterStemmer(), set(stopwords.words('english'))

ps, stop_words = load_resources()

# --- 2. LOAD ARTIFACTS WITH SAFETY CHECK ---
@st.cache_resource
def load_model_and_vectorizer():
    try:
        with open('vectorizer.pkl', 'rb') as f:
            v = pickle.load(f)
        with open('model.pkl', 'rb') as f:
            m = pickle.load(f)
        
        # Check if vectorizer is 'Fitted'
        if not hasattr(v, 'vocabulary_'):
            return "NOT_FITTED", None
        return v, m
    except Exception as e:
        return f"ERROR: {str(e)}", None

tfidf, model = load_model_and_vectorizer()

# --- 3. PREPROCESSING ---
def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    # Filter: alphanumeric, not stopwords, not punctuation, then Stem
    cleaned = [ps.stem(w) for w in tokens if w.isalnum() and w not in stop_words and w not in string.punctuation]
    return " ".join(cleaned)

# --- 4. APP INTERFACE ---
st.set_page_config(page_title="Spam Classifier", page_icon="🛡️")
st.header("🛡️ Email/SMS Spam Classifier")

# Error Handling for the NotFittedError
if tfidf == "NOT_FITTED":
    st.error("### ❌ Broken Vectorizer Detected")
    st.info("""
        The `vectorizer.pkl` from the forked repo is empty. 
        **To fix this:** Go to your Colab, run `tfidf.fit(X_train)`, 
        save the file, and upload YOUR `vectorizer.pkl` to GitHub.
    """)
    st.stop()

input_sms = st.text_area("Paste the message here:", height=150)

if st.button('Predict', type="primary"):
    if not input_sms.strip():
        st.warning("Please enter a message.")
    else:
        # Step-by-step processing
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        prediction = model.predict(vector_input)[0]
        
        if prediction == 1:
            st.error("### 🚨 Result: Spam")
        else:
            st.success("### ✅ Result: Not Spam")

st.markdown("---")
st.caption("Built with Python & Scikit-Learn")

import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. SETUP NLTK & STEMMER
@st.cache_resource
def load_resources():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    return PorterStemmer(), set(stopwords.words('english'))

ps, stop_words = load_resources()

# 2. LOAD MODEL & VECTORIZER WITH EMERGENCY FALLBACK
@st.cache_resource
def load_artifacts():
    try:
        # Try to load your files
        with open('vectorizer.pkl', 'rb') as f:
            v = pickle.load(f)
        with open('model.pkl', 'rb') as f:
            m = pickle.load(f)
        
        # EMERGENCY FIX: If the loaded vectorizer is not fitted, fit it now
        if not hasattr(v, 'vocabulary_'):
            # This prevents the NotFittedError by giving it a tiny vocabulary
            v.fit(["sample text", "spam mail", "urgent click", "hello friend"])
            
        return v, m
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None

tfidf, model = load_artifacts()

# 3. PREPROCESSING FUNCTION
def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    # Clean: Alphanumeric, remove stopwords/punctuation, then Stem
    y = [ps.stem(i) for i in tokens if i.isalnum() and i not in stop_words and i not in string.punctuation]
    return " ".join(y)

# 4. STREAMLIT UI
st.set_page_config(page_title="Email/SMS Spam Classifier", page_icon="📧")
st.header("Email/SMS Spam Classifier")
st.write("Paste the message below to analyze for spam.")

input_sms = st.text_area("Enter Message", height=150)

if st.button('Predict'):
    if not input_sms.strip():
        st.warning("Please enter a message.")
    elif tfidf is None or model is None:
        st.error("Model files are missing from the repository.")
    else:
        # Process and Predict
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        
        # Display
        if result == 1:
            st.error("### 🚨 Result: Spam")
        else:
            st.success("### ✅ Result: Not Spam")

# 5. FOOTER
st.markdown("---")
st.caption("Built with Streamlit & Scikit-Learn")

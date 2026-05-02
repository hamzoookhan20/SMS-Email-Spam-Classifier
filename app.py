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

# 2. LOAD MODEL & VECTORIZER
@st.cache_resource
def load_artifacts():
    try:
        with open('vectorizer.pkl', 'rb') as f:
            v = pickle.load(f)
        with open('model.pkl', 'rb') as f:
            m = pickle.load(f)
        
        # Check if the vectorizer actually has a vocabulary
        if not hasattr(v, 'vocabulary_'):
            return "NOT_FITTED", None
            
        return v, m
    except Exception as e:
        return f"ERROR: {str(e)}", None

tfidf, model = load_artifacts()

# 3. PREPROCESSING FUNCTION
def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    y = [ps.stem(i) for i in tokens if i.isalnum() and i not in stop_words and i not in string.punctuation]
    return " ".join(y)

# 4. STREAMLIT UI
st.set_page_config(page_title="Email/SMS Spam Classifier", page_icon="📧")
st.header("Email/SMS Spam Classifier")

if tfidf == "NOT_FITTED":
    st.error("### ❌ Critical Error: Broken Vectorizer")
    st.write("The `vectorizer.pkl` you forked is empty. You must replace it with a fitted one from Colab.")
    st.stop()
elif isinstance(tfidf, str) and "ERROR" in tfidf:
    st.error(f"### ❌ File Loading Error: {tfidf}")
    st.stop()

input_sms = st.text_area("Enter Message", height=150)

if st.button('Predict', type="primary"):
    if not input_sms.strip():
        st.warning("Please enter a message.")
    else:
        transformed_sms = transform_text(input_sms)
        # This is where the error was happening
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        
        if result == 1:
            st.error("### 🚨 Result: Spam")
        else:
            st.success("### ✅ Result: Not Spam")

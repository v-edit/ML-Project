import re
from functools import lru_cache

import joblib
import nltk
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

MIN_WORDS = 20

# ---- Cleaning: must match preprocess_v2.py exactly ----
SITE_NAMES = [
    "reuters", "abovetopsecret", "fellowshipoftheminds", "zerohedge",
    "beforeitsnews", "thedailysheeple", "infiniteunknown", "nytimes",
    "new york times", "theinternetpost", "shoebat", "newswithviews",
    "21st century wire", "breitbart", "infowars",
    "infinite unknown", "daily sheeple", "before it's news", "above top secret",
    "theodore shoebat", "walid shoebat", "christian post",
]
site_pattern = re.compile("|".join(SITE_NAMES), flags=re.I)


def strip_source_tags(text):
    text = re.sub(r"^.{0,100}?\(Reuters\)\s*-?\s*", "", text)
    text = re.sub(r"featured image via [^.]*", "", text, flags=re.I)
    text = re.sub(r"(photo|image) (by|via|credit)[^.]*", "", text, flags=re.I)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"pic\.twitter\.com/\S*|\S*twitter\.com/\S*", "", text, flags=re.I)
    text = re.sub(r"continue reading the main story|reading the main story", "", text, flags=re.I)
    text = re.sub(r"\badvertisement\b|\bsupported by\b|expand\s*/?\s*collapse", "", text, flags=re.I)
    text = re.sub(r"\bDCG\b", "", text)
    text = re.sub(r"follow [^.]{0,100}(twitter|facebook)[^.]*", "", text, flags=re.I)
    text = site_pattern.sub("", text)
    return text


@st.cache_resource
def load_everything():
    bundle = joblib.load("baseline_lr.joblib")
    return (bundle["vectorizer"], bundle["model"],
            set(stopwords.words("english")), WordNetLemmatizer())


vectorizer, model, stop_words, lemmatizer = load_everything()


@lru_cache(maxsize=None)
def lemma(word):
    return lemmatizer.lemmatize(word)


def clean_for_model(text):
    text = strip_source_tags(text[:5000]).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return " ".join(lemma(w) for w in text.split() if w not in stop_words and len(w) > 2)


# ---- App ----
st.title("📰 Fake News Detector")
st.write("Paste a news article below. The model compares its writing with thousands "
         "of articles already labelled fake or real.")

if "history" not in st.session_state:
    st.session_state.history = []

article = st.text_area("News article", height=250,
                       placeholder="Paste the full article text here...")

if st.button("Predict", type="primary"):
    if len(article.split()) < MIN_WORDS:
        st.warning(f"Please paste at least {MIN_WORDS} words. Short text gives unreliable results.")
    else:
        cleaned = clean_for_model(article)
        X = vectorizer.transform([cleaned])
        if X.nnz == 0:
            st.warning("None of the words in this text are known to the model.")
        else:
            p_fake, p_real = model.predict_proba(X)[0]
            is_real = p_real >= 0.5
            confidence = max(p_fake, p_real) * 100

            if is_real:
                st.success(f"Looks REAL  ({confidence:.1f}% confidence)")
            else:
                st.error(f"Looks FAKE  ({confidence:.1f}% confidence)")
            if confidence < 65:
                st.info("Low confidence: the writing style is not clearly one or the other.")

            st.subheader("Probability")
            st.bar_chart(pd.DataFrame({"Probability (%)": [p_fake * 100, p_real * 100]},
                                      index=["Fake", "Real"]))

            # Which words in THIS article pushed the prediction?
            names = vectorizer.get_feature_names_out()
            idx = X.nonzero()[1]
            contrib = X[0, idx].toarray().ravel() * model.coef_[0][idx]
            words = pd.DataFrame({"word": names[idx], "push": contrib})
            words["toward"] = words["push"].apply(lambda v: "Real" if v > 0 else "Fake")
            words["strength"] = words["push"].abs().round(3)
            top = words.sort_values("strength", ascending=False).head(10)
            st.subheader("Words that influenced this result")
            st.dataframe(top[["word", "toward", "strength"]], hide_index=True)

            st.session_state.history.insert(0, {
                "Article (start)": article.strip()[:80] + "...",
                "Prediction": "Real" if is_real else "Fake",
                "Confidence (%)": round(confidence, 1),
            })

if st.session_state.history:
    st.subheader("Prediction history")
    st.dataframe(pd.DataFrame(st.session_state.history), hide_index=True)

st.caption("This tool judges writing style, not facts. It cannot check whether the "
           "claims in an article are true, and it works best on news similar to its "
           "training data. Treat results as a hint, not a verdict.")

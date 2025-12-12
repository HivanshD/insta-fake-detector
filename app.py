import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Instagram Fake Profile Detector", layout="centered")

@st.cache_resource
def load_artifacts():
    pipe = joblib.load("artifacts/final_pipe.joblib")
    with open("artifacts/threshold.txt", "r") as f:
        threshold = float(f.read().strip())
    return pipe, threshold

pipe, THRESHOLD = load_artifacts()

st.title("Instagram Fake Profile Detector")
st.write("Predict whether an Instagram profile is **Fake (1)** or **Real (0)** using metadata.")

st.markdown("### Input account metadata")

col1, col2 = st.columns(2)

with col1:
    profile_pic = st.selectbox("Profile picture present?", [0, 1], index=1)
    private = st.selectbox("Private account?", [0, 1], index=0)
    external_url = st.selectbox("External URL present?", [0, 1], index=0)
    name_eq_username = st.selectbox("Name == Username?", [0, 1], index=0)

with col2:
    followers = st.number_input("#followers", min_value=0, value=100)
    follows = st.number_input("#follows", min_value=0, value=200)
    posts = st.number_input("#posts", min_value=0, value=10)
    description_length = st.number_input("Description length", min_value=0, value=20)

nums_len_username = st.slider("nums/length username", 0.0, 1.0, 0.2, 0.01)
fullname_words = st.number_input("fullname words", min_value=0, value=1)
nums_len_fullname = st.slider("nums/length fullname", 0.0, 1.0, 0.0, 0.01)

input_df = pd.DataFrame([{
    "profile pic": profile_pic,
    "nums/length username": nums_len_username,
    "fullname words": fullname_words,
    "nums/length fullname": nums_len_fullname,
    "name==username": name_eq_username,
    "description length": description_length,
    "external URL": external_url,
    "private": private,
    "#posts": posts,
    "#followers": followers,
    "#follows": follows
}])

st.divider()

if st.button("Predict"):
    proba = pipe.predict_proba(input_df)[:, 1][0]
    pred = int(proba >= THRESHOLD)

    if pred == 1:
        st.error(f"Prediction: **Fake (1)**\n\nProbability(fake) = **{proba:.3f}**")
    else:
        st.success(f"Prediction: **Real (0)**\n\nProbability(fake) = **{proba:.3f}**")

    st.caption(f"Threshold used: {THRESHOLD:.2f}")
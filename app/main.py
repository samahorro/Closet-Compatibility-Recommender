"""Starter Streamlit app (no model wired yet)."""
import streamlit as st

st.set_page_config(page_title="Closet Compatibility Recommender", page_icon="👕")
st.title("Closet Compatibility Recommender")
st.caption("AI outfit recommendations using items you already own")
mode = st.sidebar.radio("Mode", ["My Closet", "Should I Buy This?"])
if mode == "My Closet":
    st.info("Coming next: Upload your items and build compatible outfits.")
else:
    st.info("Coming next: Add a shopping item to estimate versatility and cost per outfit.")

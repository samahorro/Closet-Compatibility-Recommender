"""Starter Streamlit app (no model wired yet)."""
import streamlit as st

st.set_page_config(page_title="Closet Compatibility Recommender", page_icon="👕")
st.title("Closet Compatibility Recommender")
st.caption("AI outfit recommendations using items you already own")
mode = st.sidebar.radio("Mode", ["My Closet", "Should I Buy This?"])

# __________________________________________________Temporary Code________________________________________________________
if mode == "My Closet":
    # st.info("Coming next: Upload your items and build compatible outfits.")
    st.header("My Closet")
    st.write("Upload a photo of a clothing item you own")

    # User upload clothing image
    user_uploaded_file = st.file_uploader("Choose a clothing image", type=["jpg", "jpeg", "png"])

    # Display preview of user's uploaded image
    if user_uploaded_file is not None:
        st.image(user_uploaded_file, caption="Uploaded clothing item", width=300)
        
        # Add dropdown menu with selections Top, Bottom, Shoes, Outerwear, and Accessory (for now)
        category = st.selectbox("Select a clothing category", ["Top", "Bottom", "Shoes", "Outerwear", "Accessory"])
        st.write(f"Selected category: {category}")
# ________________________________________________________________________________________________________________________

else:
    st.info("Coming next: Add a shopping item to estimate versatility and cost per outfit.")

"""Starter Streamlit app (no model wired yet)."""
import streamlit as st
from rembg import remove
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="Closet Compatibility Recommender", page_icon="👕")
st.title("Closet Compatibility Recommender")
st.caption("AI outfit recommendations using items you already own")
mode = st.sidebar.radio("Mode", ["My Closet", "Should I Buy This?"])
@st.cache_resource
def load_bg_remover():
    from rembg import new_session
    return new_session("u2net")

def remove_bg(image_bytes):
    session = load_bg_remover()
    processed_bytes = remove(image_bytes, session=session)

    return Image.open(BytesIO(processed_bytes)).convert("RGBA")


# __________________________________________________Temporary Code________________________________________________________
if mode == "My Closet":
    # st.info("Coming next: Upload your items and build compatible outfits.")
    if "closet" not in st.session_state:
        st.session_state.closet = []
    st.header("My Closet")
    st.write("Upload a photo of a clothing item you own")

    # User upload clothing image
    user_uploaded_file = st.file_uploader("Choose a clothing image", type=["jpg", "jpeg", "png"])

    # Display preview of user's uploaded image
    if user_uploaded_file is not None:
        st.image(user_uploaded_file, caption="Uploaded clothing item", width=300)

        if st.button("Remove Background"):
            with st.spinner("Removing background. Please wait"):
                processed_image = remove_bg(user_uploaded_file.getvalue())
                #st.session_state.processed_image = processed_image
                buffer = BytesIO()
                processed_image.save(buffer, format ="PNG")
                st.session_state.processed_image = buffer.getvalue()
                st.session_state.processed_file_id = (user_uploaded_file.file_id)

        if ("processed_image" in st.session_state and st.session_state.get("processed_file_id") == user_uploaded_file.file_id):
            st.image(st.session_state.processed_image, caption="Background successfully removed!", width=300)
        
        # Add dropdown menu with selections Top, Bottom, Shoes, Outerwear, and Accessory (for now)
        category = st.selectbox("Select a clothing category", ["Top", "Bottom", "Shoes", "Outerwear", "Accessory"])
        st.write(f"Selected category: {category}")

        if st.button("Add to Closet"):
            clothing_item = {
                "name": user_uploaded_file.name,
                "category": category,
                "image": (
                    st.session_state.processed_image
                    if st.session_state.get("processed_file_id") == user_uploaded_file.file_id
                    else user_uploaded_file.getvalue())
            }

            st.session_state.closet.append(clothing_item)
            st.success("Clothing item added to your closet!")
        
    st.subheader("Your Wardrobe")
    if len(st.session_state.closet) == 0:
        st.info("Your closet is empty!")
    else:
        columns = st.columns(3)

        for index, item in enumerate(st.session_state.closet):
            with columns[index % 3]:
                st.image(item["image"], width= "stretch")
                st.write(item["name"])
                st.caption(item["category"])

                if st.button("Remove", key=f"remove_{index}"):
                    st.session_state.closet.pop(index)
                    st.rerun()

# ________________________________________________________________________________________________________________________

else:
    st.info("Coming next: Add a shopping item to estimate versatility and cost per outfit.")

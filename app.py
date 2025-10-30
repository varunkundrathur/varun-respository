import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from sklearn.neighbors import NearestNeighbors
import os
from numpy.linalg import norm
import streamlit as st
import base64 # <-- NEW IMPORT

# ============================
# 🌈 PAGE CONFIGURATION
# ============================
st.set_page_config(
    page_title="Fashion AI Recommender",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================
# 🎨 BACKGROUND & CONTAINER STYLE
# ============================
# --- TEMPORARY DEBUGGING STEP: USE RELATIVE PATH ---
# Please ensure the image file is in the SAME folder as this script before running!
background_path = "D:/VARUN/programming files/vs code/python/new fashion item generator/background.png"

# --- START FIX: Base64 Encoding ---
@st.cache_data
def get_base64_of_bin_file(bin_file):
    """Encodes a file to a base64 string for CSS embedding."""
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception as e:
        # Important: Handle case where file is not found
        st.error(f"Error loading background image: {e}. Please check the path and permissions.")
        return None

# Get the Base64 string for your image
img_base64 = get_base64_of_bin_file(background_path)

if img_base64:
    # Use the Base64 string in the CSS data URI, removing the .main-container styles
    page_bg = f"""
    <style>
    /* Full-page background image */
    .stApp {{
        background-image: url("data:image/png;base64,{img_base64}");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        /* Ensure content doesn't block the background */
        transition: background-image 0.5s;
    }}

    /*
    * .main-container CSS removed as requested.
    * Content will now float directly on the background image.
    */
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)
else:
    # Fallback style if image couldn't be loaded (e.g., file path error)
    st.warning("Could not load custom background image. Using default Streamlit background.")
    
# --- END FIX ---


# Container (div) opening tag removed


# ============================
# 🧠 MODEL LOADING
# ============================
@st.cache_resource
def load_model_and_data():
    Image_features = pkl.load(open(r'D:/VARUN/programming files/vs code/python/new fashion item generator/Dataset/Images_features.pkl', 'rb'))
    filenames = pkl.load(open(r'D:/VARUN/programming files/vs code/python/new fashion item generator/Dataset/filenames.pkl', 'rb'))
    
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
    base_model.trainable = False
    model = tf.keras.models.Sequential([base_model, GlobalMaxPool2D()])
    
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(Image_features)
    
    return Image_features, filenames, model, neighbors

dataset_path = r'D:/VARUN/programming files/vs code/python/new fashion item generator/Dataset'

# ============================
# ⚙️ FEATURE EXTRACTION FUNCTION
# ============================
def extract_features_from_images(image_path, model):
    img = image.load_img(image_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    img_preprocess = preprocess_input(img_expand_dim)
    result = model.predict(img_preprocess).flatten()
    norm_result = result / norm(result)
    return norm_result

# ============================
# 💄 UI ELEMENTS
# ============================
st.title('👗 Fashion AI Recommender 🤖')
st.subheader('✨ Discover Your Perfect Style Match with AI-Powered Recommendations ✨')

col_badge1, col_badge2, col_badge3, col_badge4 = st.columns([1,1,1,2])
with col_badge1:
    st.info('🎯 ResNet50', icon="🎯")
with col_badge2:
    st.info('🚀 Deep Learning', icon="🚀")
with col_badge3:
    st.info('💎 Smart Fashion', icon="💎")

st.divider()

with st.spinner('🔮 Loading AI Models...'):
    Image_features, filenames, model, neighbors = load_model_and_data()
st.success('✅ Models loaded successfully!')

st.divider()
st.info('📸 Upload a fashion item image and let our AI find similar styles for you!')
st.divider()

# ============================
# 📤 IMAGE UPLOAD SECTION
# ============================
upload_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
st.divider()

if upload_file is not None:
    if not os.path.exists('upload'):
        os.makedirs('upload')
    
    file_path = os.path.join('upload', upload_file.name)
    with open(file_path, 'wb') as f:
        f.write(upload_file.getbuffer())
    
    st.subheader('🎨 Your Uploaded Fashion Item')
    col_left, col_center, col_right = st.columns([0.5, 0.3, 0.5])
    with col_center:
        st.image(file_path, use_container_width=True)
    
    st.divider()
    with st.spinner('🔍 Analyzing your fashion item and finding perfect matches...'):
        input_img_features = extract_features_from_images(file_path, model)
        distance, indices = neighbors.kneighbors([input_img_features])

    st.subheader('✨ Top 5 Similar Fashion Recommendations ✨')
    st.write('')
    
    col1, col2, col3, col4, col5 = st.columns(5)
    cols = [col1, col2, col3, col4, col5]
    
    for idx, col in enumerate(cols):
        with col:
            st.markdown(f"**Match #{idx+1}**")
            # Ensure the index used here is safe: indices[0] will have n_neighbors + 1 elements, 
            # where the first is the image itself. If you want 5 *different* recommendations, 
            # start at index 1 and go up to index 5 (for a total of 6 neighbors fetched).
            # The original code uses idx+1, which is correct to skip the 0th index (the input image itself).
            img_path = os.path.join(dataset_path, filenames[indices[0][idx+1]]).replace("\\","/")
            st.image(img_path, width=160)  # Adjust size here
            
            similarity = max(0, (1 - distance[0][idx+1]) * 100)
            st.caption(f"✓ {similarity:.1f}% Similar")

    st.divider()
    st.success('🎉 Found amazing fashion matches for you!')
    st.divider()
else:
    st.warning('👆 Get Started: Upload a fashion item image to discover similar styles!')
    st.caption('Supported formats: JPG, JPEG, PNG')

st.divider()
st.info('💜 Powered by Deep Learning & Computer Vision 💜')
st.caption('Using ResNet50 Architecture for Advanced Fashion Recognition')

# Container (div) closing tag removed

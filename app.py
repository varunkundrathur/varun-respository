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
import base64

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
# 🎨 BACKGROUND STYLE
# ============================
background_path = os.path.join("assets", "background.png")

@st.cache_data
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception as e:
        st.error(f"Error loading background image: {e}")
        return None

img_base64 = get_base64_of_bin_file(background_path)

if img_base64:
    page_bg = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{img_base64}");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)
else:
    st.warning("Could not load custom background image.")

# ============================
# 🧠 MODEL LOADING
# ============================
@st.cache_resource
def load_model_and_data():
    Image_features = pkl.load(open(os.path.join("Dataset", "Images_features.pkl"), "rb"))
    filenames = pkl.load(open(os.path.join("Dataset", "filenames.pkl"), "rb"))

    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        GlobalMaxPool2D()
    ])

    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(Image_features)
    return Image_features, filenames, model, neighbors

# Set correct dataset path
dataset_images_path = os.path.join("Dataset", "images")

# ============================
# ⚙️ FEATURE EXTRACTION
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
# 💄 UI
# ============================
st.title('👗 Fashion AI Recommender 🤖')
st.subheader('✨ Discover Your Perfect Style Match with AI-Powered Recommendations ✨')

col1, col2, col3 = st.columns(3)
with col1: st.info('🎯 ResNet50', icon="🎯")
with col2: st.info('🚀 Deep Learning', icon="🚀")
with col3: st.info('💎 Smart Fashion', icon="💎")

st.divider()

with st.spinner('🔮 Loading AI Models...'):
    Image_features, filenames, model, neighbors = load_model_and_data()
st.success('✅ Models loaded successfully!')

st.divider()
st.info('📸 Upload a fashion item image and let our AI find similar styles for you!')
st.divider()

upload_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
st.divider()

if upload_file is not None:
    if not os.path.exists('upload'):
        os.makedirs('upload')

    file_path = os.path.join('upload', upload_file.name)
    with open(file_path, 'wb') as f:
        f.write(upload_file.getbuffer())

    st.subheader('🎨 Your Uploaded Fashion Item')
    st.image(file_path, use_container_width=True)

    with st.spinner('🔍 Analyzing your fashion item and finding perfect matches...'):
        input_img_features = extract_features_from_images(file_path, model)
        distance, indices = neighbors.kneighbors([input_img_features])

    st.subheader('✨ Top 5 Similar Fashion Recommendations ✨')
    cols = st.columns(5)

    for idx, col in enumerate(cols):
        with col:
            st.markdown(f"**Match #{idx+1}**")

            # Use correct image path
            img_name = os.path.basename(filenames[indices[0][idx+1]])
            img_path = os.path.join("Dataset", "images", img_name)

            

            if os.path.exists(img_path):
                st.image(img_path, width=160)
            else:
                st.warning(f"Image not found: {img_name}")

            similarity = max(0, (1 - distance[0][idx+1]) * 100)
            st.caption(f"✓ {similarity:.1f}% Similar")

    st.success('🎉 Found amazing fashion matches for you!')
else:
    st.warning('👆 Get Started: Upload a fashion item image to discover similar styles!')

st.divider()
st.info('💜 Powered by Deep Learning & Computer Vision 💜')
st.caption('Using ResNet50 Architecture for Advanced Fashion Recognition')

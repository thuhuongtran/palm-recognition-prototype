import streamlit as st
import numpy as np
import torch
import cv2
import os
from PIL import Image
import tempfile
from client.cosine_similarity import calculate_similarity, find_best_matches
from preprocess.preprocessor import preprocess_image
from self_supervised.model import PalmprintEncoder
st.set_page_config(page_title="Match Viewer", layout="wide")
# Custom CSS for background and styling
st.markdown(
    """
    <style>
    body {
        background-color: #f5f5f5;
        font-family: Arial, sans-serif;
    }
    .score-text {
        color: green;
        font-weight: bold;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

ENCODER_PATH = 'output/model/palmprint_encoder.pth'
PREPROCESSED_DATA_DIR = 'dataset/preprocessed_images'  # Path to preprocessed images

# Initialize preprocessor (adjust parameters if needed)
encoder = PalmprintEncoder().encoder
encoder.load_state_dict(torch.load(ENCODER_PATH))
encoder.eval()  # Set to evaluation mode
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # use GPU if available
encoder.to(device)

all_embeddings = np.load('output/embeddings/all_embeddings.npy')
image_names = np.load(
    'output/embeddings/image_names.npy').tolist()  # Load as list to avoid numpy string array truncation issues

# --- Streamlit App ---
st.title("Palmprint Recognition")

uploaded_file = st.file_uploader("Choose a palmprint image...", type=["jpg", "jpeg", "png", "tiff"])

if uploaded_file is not None:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tiff") as temp_file:  # Create a temporary TIFF file
            temp_file.write(uploaded_file.read())  # Write uploaded data to file
            temp_file_path = temp_file.name  # Get the path to the temporary file

        # --- Process uploaded image (using temp file path)
        image = Image.open(uploaded_file).convert("RGB")  # Still open original for display
        image = np.array(image)
        preprocessed_roi = preprocess_image(temp_file_path)  # Correct: Pass temp file path
        os.remove(temp_file_path)  # Remove temporary file (optional but good practice)

        if preprocessed_roi is None:
            st.error("Error: Could not preprocess the uploaded image. Please check the image quality.")
        else:
            with st.container():
                col1, col2 = st.columns([6.5, 4])  # 30% for uploaded image, 70% for matches

                # Display uploaded image
                with col1:
                    st.subheader("Uploaded Image")
                    st.image(image, use_container_width=True)

                # Display matched images
                with col2:
                    # --- Feature Extraction for Uploaded Image ---
                    preprocessed_roi = np.expand_dims(preprocessed_roi, axis=0)
                    preprocessed_roi = torch.from_numpy(preprocessed_roi).float().repeat(1, 3, 1, 1).to(
                        device)  # Add batch and channel dims, move to device
                    with torch.no_grad():  # disable gradient calculations
                        query_embedding = encoder(preprocessed_roi).cpu().numpy().flatten()

                    # --- Matching ---
                    similarity_scores = calculate_similarity(query_embedding, all_embeddings)
                    best_matches, match_found = find_best_matches(similarity_scores, image_names,
                                                                  threshold=0.8)  # Adjust threshold

                    st.subheader("Matching Results:")

                    if match_found:
                        st.success("Match found!")
                    else:
                        st.warning("No match found above the threshold.")

                    rows = 2
                    cols = 3
                    images_per_row = cols
                    total_matches = len(best_matches)

                    for i in range(0, total_matches, images_per_row):
                        row_images = best_matches[i:i + images_per_row]
                        columns = st.columns(3)
                        for col, (image_name, score) in zip(columns, row_images):
                            image_path = os.path.join(PREPROCESSED_DATA_DIR, image_name.replace('.npy',
                                                                                                '') + ".npy")  # Find path to preprocessed image
                            matched_image = np.load(image_path)

                            with col:
                                st.image(matched_image, use_container_width=True)
                                st.markdown(
                                    f"<div style='color: green; font-weight: bold;'>Score: {score * 100:.2f}%</div>",
                                    unsafe_allow_html=True)
                                st.write(f"**Image:** {image_name}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

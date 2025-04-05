import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import lightgbm as lgb
from joblib import load

# Load the model (do this once when the app starts)
@st.cache_resource
def load_model():
    try:
        # Try loading the joblib version first
        return load("E:/B Tech/Mini Project/Digit/Digit_classification.joblib")
    except:
        # Fall back to LightGBM native format
        model = lgb.LGBMClassifier()
        model._Booster = lgb.Booster(model_file='E:/B Tech/Mini Project/Digit/Digit_classification.txt')
        return model

model = load_model()

# Set up the app
st.title("🧠 Digit Recognition App")
st.write("✏️ Draw a digit (0-9) in the box below and click **'Predict'**")

# Create a canvas component
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=15,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Add buttons
col1, col2 = st.columns(2)
with col1:
    predict_btn = st.button("Predict")

# Prediction function
def predict_digit(image):
    # Resize to 28x28 and invert colors
    img_resized = image.resize((28, 28))
    img_inverted = ImageOps.invert(img_resized)

    # Normalize and flatten
    img_array = np.array(img_inverted) / 255.0
    img_array = img_array.reshape(1, -1)

    # Predict
    prediction = model.predict(img_array)
    return prediction[0]


# Handle predict button
if predict_btn:
    if canvas_result.image_data is not None:
        # Convert canvas to PIL Image
        img = Image.fromarray((canvas_result.image_data[:, :, :3]).astype('uint8'))  # Use only RGB
        img = img.convert("L")  # Convert to grayscale

        # Predict digit
        digit = predict_digit(img)
        st.success(f"✅ Predicted Digit: **{digit}**")
    else:
        st.warning("⚠️ Please draw a digit before clicking Predict.")

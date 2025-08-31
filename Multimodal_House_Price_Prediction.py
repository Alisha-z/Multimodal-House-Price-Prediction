import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# --- Optional: quiet TensorFlow logs ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# -------------------------------
# Utilities: model / cnn loading
# -------------------------------
@st.cache_resource
def load_pkl(path="housing_model.pkl"):
    obj = joblib.load(path)
    # Cases:
    # 1) (preprocessor, model)
    if isinstance(obj, (tuple, list)) and len(obj) >= 2:
        a, b = obj[0], obj[1]
        # Identify which is which
        pre, mdl = (a, b) if hasattr(a, "transform") and hasattr(b, "predict") else (b, a)
        return {"mode": "tuple", "preprocessor": pre, "model": mdl}
    # 2) sklearn Pipeline (has predict; may or may not expose steps)
    if hasattr(obj, "predict"):
        return {"mode": "single", "pipeline_or_model": obj}
    # Fallback
    return {"mode": "unknown", "pipeline_or_model": obj}

@st.cache_resource
def load_cnn():
    # Lazy import to speed up app startup
    import tensorflow as tf
    from tensorflow.keras.applications import ResNet50
    base = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    cnn = tf.keras.Model(inputs=base.input, outputs=base.output)
    return cnn

def extract_resnet50_features(pil_image):
    import tensorflow as tf
    from tensorflow.keras.applications.resnet50 import preprocess_input
    img = pil_image.convert("RGB").resize((224, 224))
    arr = np.array(img).astype("float32")
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)             # (1, 224, 224, 3)
    cnn = load_cnn()
    feats = cnn.predict(arr, verbose=0).flatten()  # 2048-dim
    return feats  # (2048,)

# ---------------------------------
# Streamlit UI
# ---------------------------------
st.set_page_config(page_title="🏠 Multimodal House Price Prediction", layout="wide")
st.title("🏠 Multimodal House Price Prediction")
st.caption("Tabular features + optional house image")

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("House Features")
    area = st.number_input("Area (sq ft)", min_value=300, max_value=20000, value=1200, step=50)
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3, step=1)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)
    stories = st.number_input("Stories", min_value=1, max_value=5, value=1, step=1)
    mainroad = st.selectbox("Main Road", ["yes", "no"])
    guestroom = st.selectbox("Guest Room", ["yes", "no"])
    basement = st.selectbox("Basement", ["yes", "no"])
    hotwaterheating = st.selectbox("Hot Water Heating", ["yes", "no"])
    airconditioning = st.selectbox("Air Conditioning", ["yes", "no"])
    parking = st.number_input("Parking", min_value=0, max_value=5, value=1, step=1)
    prefarea = st.selectbox("Preferred Area", ["yes", "no"])
    furnishingstatus = st.selectbox("Furnishing Status", ["furnished", "semi-furnished", "unfurnished"])

with col_right:
    st.subheader("Upload House Image")
    uploaded_img = st.file_uploader("Upload JPG/PNG (optional but recommended)", type=["jpg", "jpeg", "png"])
    if uploaded_img:
        from PIL import Image
        img = Image.open(uploaded_img)
        st.image(img, caption="Uploaded House Image", use_container_width=True)
    else:
        img = None
        st.info("No image uploaded — the model will predict using tabular features only (or pad zeros if it expects image features).")

# pack inputs in a DataFrame with the exact training column names
input_df = pd.DataFrame([{
    "area": area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "stories": stories,
    "mainroad": mainroad,
    "guestroom": guestroom,
    "basement": basement,
    "hotwaterheating": hotwaterheating,
    "airconditioning": airconditioning,
    "parking": parking,
    "prefarea": prefarea,
    "furnishingstatus": furnishingstatus
}])

st.divider()

if st.button("🔮 Predict Price", use_container_width=True):
    try:
        bundle = load_pkl("housing_model.pkl")

        # ---- Case A: you saved (preprocessor, model) and trained model on [tabular_preprocessed | image_2048] ----
        if bundle["mode"] == "tuple":
            pre = bundle["preprocessor"]
            mdl = bundle["model"]

            # Ensure all expected columns exist (add missing with safe defaults)
            expected_cols = getattr(pre, "feature_names_in_", None)
            if expected_cols is not None:
                for c in expected_cols:
                    if c not in input_df.columns:
                        # Numeric -> 0, Categorical -> most common 'no'
                        input_df[c] = 0 if c not in ["mainroad", "guestroom", "basement",
                                                     "hotwaterheating", "airconditioning",
                                                     "prefarea", "furnishingstatus"] else "no"
            # Transform tabular
            X_tab = pre.transform(input_df)
            X_tab = X_tab.toarray() if hasattr(X_tab, "toarray") else X_tab

            # Image features
            if img is not None:
                img_feats = extract_resnet50_features(img)  # (2048,)
            else:
                img_feats = np.zeros(2048, dtype="float32")  # pad zeros if not provided

            X_combined = np.hstack([X_tab, img_feats.reshape(1, -1)])

            # If model was actually trained on tabular-only (no image), adapt width automatically
            n_expected = getattr(mdl, "n_features_in_", None)
            if n_expected is not None and X_combined.shape[1] != n_expected:
                # try without image
                if X_tab.shape[1] == n_expected:
                    X_for_pred = X_tab
                # try trimming/padding to expected width
                elif X_combined.shape[1] > n_expected:
                    X_for_pred = X_combined[:, :n_expected]
                else:
                    pad = np.zeros((X_combined.shape[0], n_expected - X_combined.shape[1]))
                    X_for_pred = np.hstack([X_combined, pad])
            else:
                X_for_pred = X_combined

            yhat = mdl.predict(X_for_pred)[0]
            st.success(f"🏡 Estimated House Price: **${yhat:,.2f}**")

        # ---- Case B: you saved a single Pipeline or model ----
        elif bundle["mode"] == "single":
            pipe_or_model = bundle["pipeline_or_model"]
            # If it's a Pipeline with preprocessor inside, just pass the DataFrame.
            # (It likely ignores image features; if it expects image features use tuple mode above.)
            yhat = pipe_or_model.predict(input_df)[0]
            st.info("Prediction used tabular features only (single saved Pipeline/model).")
            st.success(f"🏡 Estimated House Price: **${yhat:,.2f}**")

        else:
            st.error("Unrecognized model format inside housing_model.pkl")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

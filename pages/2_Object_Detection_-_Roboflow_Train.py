import streamlit as st
import numpy as np
import cv2
from PIL import Image
from roboflow import Roboflow
import pandas as pd

# --------------------------------------------------
# DEFAULT VALUES
# --------------------------------------------------
DEFAULT_PROJECT_URL = "https://app.roboflow.com/kkwagh-63ouy/kkwagh-group15/1"
DEFAULT_PRIVATE_API_KEY = "C8Izi0EdivGuDkEYVXGB"

# --------------------------------------------------
# SESSION STATE INIT
# --------------------------------------------------
if "workspace_id" not in st.session_state:
    st.session_state.workspace_id = None
if "model_id" not in st.session_state:
    st.session_state.model_id = None
if "version_number" not in st.session_state:
    st.session_state.version_number = None
if "model" not in st.session_state:
    st.session_state.model = None

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.title("🌼 KK Wagh Group 15 – Gerbera Detection App")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.subheader("Upload Image")
    uploaded_file_od = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    st.subheader("Realtime Detection")
    if st.button("Realtime Detect using Camera"):
        st.markdown(
            '<a href="https://cameradetect.netlify.app/" target="_blank">Realtime Detection Link</a>',
            unsafe_allow_html=True
        )

    confidence_threshold = st.slider("Confidence Threshold (%)", 0, 100, 40)
    overlap_threshold = st.slider("Overlap Threshold (%)", 0, 100, 30)

    show_bbox = st.radio("Show Bounding Boxes", ["Yes", "No"])
    show_class_label = st.radio("Show Class Labels", ["Show Labels", "Hide Labels"])
    show_box_type = st.selectbox("Box Type", ["regular", "fill", "blur"])
    box_width = int(st.selectbox("Box Width", ["1", "2", "3", "4", "5"]))
    text_width = int(st.selectbox("Text Thickness", ["1", "2", "3"]))

# --------------------------------------------------
# PROJECT FORM
# --------------------------------------------------
with st.form("project_access"):
    project_url_od = st.text_input("Project URL", value=DEFAULT_PROJECT_URL)
    private_api_key = st.text_input("Private API Key", value=DEFAULT_PRIVATE_API_KEY, type="password")
    submitted = st.form_submit_button("Verify and Load Model")

    if submitted:
        try:
            extracted = project_url_od.replace("https://app.roboflow.com/", "").split("/")
            st.session_state.workspace_id = extracted[0]
            st.session_state.model_id = extracted[1]
            st.session_state.version_number = extracted[2]

            rf = Roboflow(api_key=private_api_key)
            project = rf.workspace(st.session_state.workspace_id).project(st.session_state.model_id)
            version = project.version(st.session_state.version_number)
            st.session_state.model = version.model

            st.success("✅ Model Loaded Successfully")

        except Exception as e:
            st.error(f"❌ Error loading model: {e}")

# --------------------------------------------------
# INFERENCE FUNCTION
# --------------------------------------------------
def run_inference(image_np):
    model = st.session_state.model

    result = model.predict(
        image_np,
        confidence=confidence_threshold,
        overlap=overlap_threshold
    ).json()

    return result

# --------------------------------------------------
# IMAGE PROCESSING
# --------------------------------------------------
if uploaded_file_od and st.session_state.model:

    image = Image.open(uploaded_file_od).convert("RGB")
    image_np = np.array(image)
    output_img = image_np.copy()

    predictions = run_inference(image_np)
    pred_list = predictions["predictions"]

    collected = []

    for pred in pred_list:
        x, y = int(pred["x"]), int(pred["y"])
        w, h = int(pred["width"]), int(pred["height"])
        cls = pred["class"]
        conf = pred["confidence"]

        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)

        if show_bbox == "Yes":
            cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 0, 0), box_width)

        if show_class_label == "Show Labels":
            cv2.putText(
                output_img,
                f"{cls} {conf:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                text_width
            )

        collected.append({
            "Class": cls,
            "Confidence": conf,
            "Width": w,
            "Height": h
        })

    # --------------------------------------------------
    # OUTPUT
    # --------------------------------------------------
    col1, col2 = st.columns(2)
    col1.image(image, caption="Uploaded Image", width=400)
    col2.image(output_img, caption="Inferenced Image", width=400)

    tab1, tab2 = st.tabs(["Results", "JSON Output"])
    with tab1:
        st.dataframe(pd.DataFrame(collected))
    with tab2:
        st.json(predictions)

elif uploaded_file_od:
    st.warning("⚠️ Load model first")

import streamlit as st
import numpy as np
import cv2
from PIL import Image
from roboflow import Roboflow
import pandas as pd
import tempfile

# ==================================================
# DEFAULT VALUES
# ==================================================
DEFAULT_PROJECT_URL = "https://app.roboflow.com/kkwagh-63ouy/kkwagh-group15/1"
DEFAULT_PRIVATE_API_KEY = "C8Izi0EdivGuDkEYVXGB"

# ==================================================
# SESSION STATE INIT
# ==================================================
if "workspace_id" not in st.session_state:
    st.session_state.workspace_id = None
if "model_id" not in st.session_state:
    st.session_state.model_id = None
if "version_number" not in st.session_state:
    st.session_state.version_number = None
if "model" not in st.session_state:
    st.session_state.model = None

# ==================================================
# TITLE
# ==================================================
st.title("🌼 KK Wagh Group 15 – Gerbera Detection App")

# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader(
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

    box_type = st.selectbox("Bounding Box Type", ["regular", "fill", "blur"])
    amount_blur = st.radio("Blur Amount", ["Low", "High"])

    box_width = int(st.selectbox("Box Width", ["1", "2", "3", "4", "5"]))
    text_width = int(st.selectbox("Text Thickness", ["1", "2", "3"]))

# ==================================================
# PROJECT ACCESS FORM
# ==================================================
with st.form("project_access"):
    project_url = st.text_input("Project URL", value=DEFAULT_PROJECT_URL)
    private_api_key = st.text_input(
        "Private API Key",
        value=DEFAULT_PRIVATE_API_KEY,
        type="password"
    )
    submitted = st.form_submit_button("Verify and Load Model")

    if submitted:
        try:
            extracted = project_url.replace(
                "https://app.roboflow.com/", ""
            ).split("/")

            st.session_state.workspace_id = extracted[0]
            st.session_state.model_id = extracted[1]
            st.session_state.version_number = extracted[2]

            rf = Roboflow(api_key=private_api_key)
            project = rf.workspace(
                st.session_state.workspace_id
            ).project(st.session_state.model_id)
            version = project.version(st.session_state.version_number)

            st.session_state.model = version.model
            st.success("✅ Model Loaded Successfully")

        except Exception as e:
            st.error(f"❌ Error loading model: {e}")

# ==================================================
# INFERENCE FUNCTION (FIXED & CORRECT)
# ==================================================
def run_inference(image_np):
    model = st.session_state.model

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        Image.fromarray(image_np).save(tmp.name)
        image_path = tmp.name

    result = model.predict(
        image_path,
        confidence=confidence_threshold,
        overlap=overlap_threshold
    ).json()

    return result

# ==================================================
# IMAGE PROCESSING
# ==================================================
if uploaded_file and st.session_state.model:

    image = Image.open(uploaded_file).convert("RGB")
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

        # -----------------------------
        # BOX DRAWING
        # -----------------------------
        if show_bbox == "Yes":

            if box_type == "regular":
                cv2.rectangle(
                    output_img,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 0),
                    box_width
                )

            elif box_type == "fill":
                cv2.rectangle(
                    output_img,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 0),
                    -1
                )

            elif box_type == "blur":
                roi = output_img[y1:y2, x1:x2]
                if roi.size > 0:
                    k = 51 if amount_blur == "High" else 31
                    roi = cv2.GaussianBlur(roi, (k, k), 0)
                    output_img[y1:y2, x1:x2] = roi

                cv2.rectangle(
                    output_img,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 0),
                    box_width
                )

        # -----------------------------
        # LABEL
        # -----------------------------
        if show_class_label == "Show Labels":
            cv2.rectangle(
                output_img,
                (x1, y1 - 25),
                (x1 + 120, y1),
                (0, 0, 0),
                -1
            )
            cv2.putText(
                output_img,
                f"{cls} {conf:.2f}",
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                text_width
            )

        collected.append({
            "Class": cls,
            "Confidence": conf,
            "Width": w,
            "Height": h,
            "BBox Area": w * h
        })

    # ==================================================
    # OUTPUT
    # ==================================================
    col1, col2 = st.columns(2)
    col1.image(image, caption="Uploaded Image", use_container_width=True)
    col2.image(output_img, caption="Inferenced Image", use_container_width=True)

    tab1, tab2 = st.tabs(["Inference Results", "JSON Output"])
    with tab1:
        st.dataframe(pd.DataFrame(collected))
    with tab2:
        st.json(predictions)

elif uploaded_file:
    st.warning("⚠️ Please load the model first")

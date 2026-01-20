import streamlit as st

# Title and introduction
st.title("Gerbera Flower Quantity Estimation System")
st.write("### Group No. 15 of KKWIEER")

# Description
st.write("""
#### Project Steps:
1. **Aerial Image Capture:** Using drones to capture aerial images of flowers from floriculture farms.
2. **Image Processing:** Utilizing advanced image analysis techniques to process captured images.
3. **Quantification:** Employing Convolutional Neural Networks (CNNs) to accurately quantify flower quantities.
""")

# Image
st.image("flower.jpg", caption="Aerial image of flowers captured by drone", use_column_width=True)

# Thank you message with pop animation and center alignment
st.write("""
### Thank you
""", unsafe_allow_html=True)

# Customizing the layout with animation effects and center alignment
st.markdown(
    """
    <style>
    .stApp {
        background-color: #000; /* Background color */
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 100vh;
    }
    .stMarkdown {
        color: #fff; /* Text color */
        animation: fadeIn 1s; /* Fade-in animation for text */
        text-align: center; /* Center-align text */
    }
    .stImage {
        border-radius: 10px; /* Rounded corners for the image */
        box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2); /* Shadow effect for the image */
        animation: pop 0.5s; /* Pop animation for the image */
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
        }
        to {
            opacity: 1;
        }
    }
    
    @keyframes pop {
        from {
            transform: scale(0);
        }
        to {
            transform: scale(1);
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

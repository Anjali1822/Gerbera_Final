import streamlit as st

# CSS styling
st.markdown("""
    <style>
        .instructions {
            padding: 10px;
            margin-bottom: 20px;
            border-radius: 5px;
            background-color: #000;
        }
    </style>
""", unsafe_allow_html=True)

st.write("""
Hello, we are group no. 15 of KKWIEER
""")

st.write("# How to Use this App:")

# English instructions
st.markdown("""
<div class="instructions">
    <h4>English Instructions:</h4>
    <ol>
        <li>First, go to the sidebar and select 'Menu Object Detection'.</li>
        <li>Select the image/video you want to count flowers, or choose the real-time option for live detection using the camera.</li>
        <li>Click on 'Verify and Load Model' after uploading the input image/video file.</li>
        <li>Wait for 5-6 seconds and see the magic!</li>
    </ol>
</div>
""", unsafe_allow_html=True)

# Hindi instructions
st.markdown("""
<div class="instructions">
    <h4>हिंदी निर्देश:</h4>
    <ol>
        <li>सबसे पहले, साइडबार में जाएं और 'मेनू ऑब्जेक्ट डिटेक्शन' का चयन करें।</li>
        <li>फूल गिनने के लिए चित्र / वीडियो का चयन करें, या कैमरा का उपयोग करके लाइव डिटेक्शन के लिए वास्तविक समय विकल्प का चयन करें।</li>
        <li>इनपुट छवि / वीडियो फ़ाइल अपलोड करने के बाद 'सत्यापित और मॉडल लोड करें' पर क्लिक करें।</li>
        <li>5-6 सेकंड के लिए प्रतीक्षा करें और जादू देखें!</li>
    </ol>
</div>
""", unsafe_allow_html=True)

# Marathi instructions
st.markdown("""
<div class="instructions">
    <h4>मराठी मार्गदर्शन:</h4>
    <ol>
        <li>पहिल्यांदा, साइडबारमध्ये जाऊन 'मेनू ऑब्जेक्ट डिटेक्शन' निवडा.</li>
        <li>फुल गणना करण्यासाठी चित्र / व्हिडिओ निवडा, किंवा कॅमेरा वापरून लाइव डिटेक्शनसाठी वास्तविक वेळ निवडा।</li>
        <li>इनपुट इमेज / व्हिडिओ फाईल अपलोड केल्यानंतर, 'सत्यापित आणि मॉडेल लोड करा' वर क्लिक करा।</li>
        <li>5-6 सेकंदं वाट पाहा आणि जादू पहा!</li>
    </ol>
</div>
""", unsafe_allow_html=True)

st.write("""
### Thank you
""")

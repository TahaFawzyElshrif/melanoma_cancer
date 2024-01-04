import streamlit as st
from keras.models import load_model
import cv2
import numpy as np
import base64

def classify_image(image, model):
    resized_image = cv2.resize(image, (200, 300))
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    data = np.ndarray(shape=(1,  300,200,1), dtype=np.float32)
    data[0] = resized_image.reshape(300,200,1)

    prediction = model.predict(data)
    confidence_score = prediction[0][0]

    class_name = "malignant" if confidence_score >= 0.5 else "benign"
    return class_name, confidence_score
def set_background(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

def main():
    set_background('b2.jpg')


    st.title('Melanoma Cancer classification')
    st.header('Please upload a  image')

    file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

    model = load_model('CNN.h5')

    if file is not None:
        image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image, use_column_width=True)

        class_name, confidence_score = classify_image(image, model)

        st.write("## {}".format(class_name))
        st.write("### score: {}%".format(confidence_score * 100))

if __name__ == "__main__":
    main()

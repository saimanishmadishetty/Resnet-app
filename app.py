import streamlit as st
from PIL import Image
from vipas import model
from vipas.exceptions import UnauthorizedException, NotFoundException, RateLimitExceededException
import base64
import io

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'button_clicked' not in st.session_state:
    st.session_state['button_clicked'] = False

# Set the title and description with new font style and colors
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
        }
        .title {
            font-size: 2.5rem;
            color: #FF6347;
            text-align: center;
        }
        .description {
            font-size: 1.25rem;
            color: #2F4F4F;
            text-align: center;
            margin-bottom: 2rem;
        }
        .uploaded-image {
            border: 2px solid #FF6347;
            border-radius: 8px;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
        .prediction-container {
            margin-top: 20px;
            text-align: center;
        }
        .prediction-title {
            font-size: 24px;
            color: #333;
            text-align: center;
        }
        .prediction-class {
            font-size: 20px;
            color: #FF6347;
            text-align: center;
        }
        .confidence {
            font-size: 20px;
            color: #4682B4;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">ImageInsight</div>', unsafe_allow_html=True)
st.markdown('<div class="description">Unleash the power of AI with ImageInsight. Upload an image and let our state-of-the-art ResNet50 model accurately classify objects and scenes with remarkable precision.</div>', unsafe_allow_html=True)

# Upload image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Center the classify button
st.markdown("""
    <style>
        .stButton button {
            display: block;
            margin-left: auto;
            margin-right: auto;
            background-color: #FF6347;
            color: white;
            padding: 10px 24px;
            font-size: 16px;
            cursor: pointer;
            border: none;
            border-radius: 8px;
        }
        .stButton button:hover {
            background-color: #FF4500;
        }
    </style>
""", unsafe_allow_html=True)

if uploaded_file is not None:
    vps_model_client = model.ModelClient()
    model_id = "mdl-vpjyi79hacuqr"
    image = Image.open(uploaded_file)
    
    # Convert the image to base64
    buffered = io.BytesIO()
    image_format = image.format
    image.save(buffered, format=image_format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    if st.button('🔍 Classify'):
        st.session_state['button_clicked'] = True

    if st.session_state['button_clicked'] and st.session_state['data'] is None:
        try:
            api_response = vps_model_client.predict(model_id=model_id, input_data=img_str, async_mode=False)
            st.session_state['data'] = api_response
            detected_classes = api_response[0].split(', ')
            confidence = api_response[1]
        except UnauthorizedException as e:
            st.error(f"Unauthorized exception: {str(e)}")
        except NotFoundException as e:
            st.error(f"Not found exception: {str(e)}")
        except RateLimitExceededException as e:
            st.error(f"Rate limit exceeded exception: {str(e)}")
        except Exception as e:
            st.error(f"Exception when calling model->predict: {str(e)}")
    else:
        detected_classes = st.session_state['data'][0].split(', ') if st.session_state['data'] else []
        confidence = st.session_state['data'][1] if st.session_state['data'] else 0.0

    # Layout for image and prediction
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption='Uploaded Image', use_column_width=True)

    with col2:
        if st.session_state['data'] is not None:
            st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
            st.markdown('<p class="prediction-title"><strong>Prediction:</strong></p>', unsafe_allow_html=True)
            
            for detected_class in detected_classes:
                st.markdown(f'<p class="prediction-class">{detected_class}</p>', unsafe_allow_html=True)
                
            st.markdown(f'<p class="confidence">Confidence: {confidence:.2%}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="prediction-container">
                    <p class="prediction-title"><strong>Prediction:</strong></p>
                    <p class="confidence">Upload an image and click "Classify" to see the prediction.</p>
                </div>
            """, unsafe_allow_html=True)

# Add some styling with Streamlit's Markdown
st.markdown("""
    <style>
        .stApp {
            background-color: #f5f5f5;
            padding: 0;
        }
        .stApp > header {
            position: fixed;
            top: 0;
            width: 100%;
            z-index: 1;
            background: #ffffff;
            border-bottom: 1px solid #e0e0e0;
        }
        .stApp > main {
            margin-top: 4rem;
            padding: 2rem;
        }
        pre {
            background: #e0f7fa;
            padding: 15px;
            border-radius: 8px;
            white-space: pre-wrap;
            word-wrap: break-word;
            border: 1px solid #FF6347;
        }
        .css-1cpxqw2.e1ewe7hr3 {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

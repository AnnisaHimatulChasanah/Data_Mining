import streamlit as st
import numpy as np
import cv2
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import tflite_runtime.interpreter as tflite

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Deteksi Real-time dengan TFLite", page_icon="📸", layout="centered")
st.title("Deteksi Gambar Real-time Menggunakan Model TFLite")
st.write("Aplikasi ini mendeteksi kelas dari gambar secara real-time menggunakan model yang ringan (TFLite).")

# --- Load Model TFLite ---
@st.cache_resource
def load_tflite_model():
    interpreter = tflite.Interpreter(model_path="model_inceptionv3_best.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    st.success("✅ Model TFLite berhasil dimuat!")
    return interpreter, input_details, output_details

interpreter, input_details, output_details = load_tflite_model()

CLASS_NAMES = [
    'Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold',
    'Septoria_leaf_spot', 'Spider_mites Two-spotted_spider_mite',
    'Target_Spot', 'Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_mosaic_virus',
    'healthy', 'powdery_mildew'
]
INPUT_SHAPE = (299, 299)

# --- Preprocessing ---
def preprocess_image(image_array, target_size):
    image_pil = Image.fromarray(image_array)
    image_resized = image_pil.resize(target_size)
    image_array_resized = np.asarray(image_resized, dtype=np.float32) / 255.0
    image_array_expanded = np.expand_dims(image_array_resized, axis=0)
    return image_array_expanded

# --- Prediction ---
def predict_frame(frame):
    preprocessed = preprocess_image(frame, INPUT_SHAPE)
    interpreter.set_tensor(input_details[0]['index'], preprocessed)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = np.argmax(output_data[0])
    confidence = np.max(output_data[0]) * 100
    return CLASS_NAMES[predicted_index], confidence

# --- Stream Kamera ---
st.subheader("Ambil Gambar dari Kamera")
st.write("Klik 'Mulai Deteksi' untuk mengaktifkan kamera dan melihat hasil deteksi real-time.")

start_button = st.button("Mulai Deteksi")
stop_button = st.button("Hentikan Deteksi")

if start_button:
    st.session_state.run_camera = True
if stop_button:
    st.session_state.run_camera = False

if 'run_camera' not in st.session_state:
    st.session_state.run_camera = False

if st.session_state.run_camera:

    class VideoProcessor(VideoTransformerBase):
        def __init__(self):
            self.result_text = "Menunggu deteksi..."

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pred_class, confidence = predict_frame(img_rgb)
            self.result_text = f"Deteksi: {pred_class} ({confidence:.2f}%)"
            cv2.putText(img, self.result_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return img

    st.info("Memulai stream kamera. Harap izinkan akses kamera di browser Anda.")

    ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_transformer_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True,
    )

    if ctx.state.playing:
        st.write("Stream kamera aktif.")
        if ctx.video_transformer:
            st.write(ctx.video_transformer.result_text)
else:
    st.warning("Tekan 'Mulai Deteksi' untuk memulai kamera.")

st.markdown("---")
st.write("Aplikasi ini menggunakan model TFLite yang ringan dan cocok untuk deployment di cloud 🌥️.")

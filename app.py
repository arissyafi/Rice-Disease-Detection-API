import os

# --- MAGIC CODE (WAJIB PALING ATAS) ---
os.environ["TF_USE_LEGACY_KERAS"] = "1"
# --------------------------------------

import cv2
import numpy as np
import gdown
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from pyngrok import ngrok
import gc

# ==========================================
# PERBAIKAN IMPORT: PAKE TF_KERAS LANGSUNG
# ==========================================
import tf_keras as keras
from tf_keras.models import load_model
from tf_keras.applications.efficientnet import preprocess_input

# ==========================================
# 1. KONFIGURASI
# ==========================================
# Masukkan Token Ngrok Anda
NGROK_AUTH_TOKEN = "36rioVsfwi4eUJHyR87ObXPQ1qO_QoEA2WSvxT4wA8KA7j4t" 
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('templates', exist_ok=True)

# ==========================================
# 2. LOAD MODEL
# ==========================================
MODEL_FILE = 'best_model.h5'
FILE_ID = "1wWHM3R1T-3D9tyxYeuRF6UTKaARR0V-n" 
URL = f"https://drive.google.com/uc?id={FILE_ID}"

if not os.path.exists(MODEL_FILE):
    print("⬇️ Download Model...")
    gdown.download(URL, MODEL_FILE, quiet=False, fuzzy=True)

print("⏳ Memuat Model...")
try:
    # KITA PAKSA PAKAI tf_keras AGAR BISA BACA TFOpLambda
    model = keras.models.load_model(MODEL_FILE, compile=False)
    print("✅ Model Siap (tf_keras Mode)!")
except Exception as e:
    print(f"❌ Gagal Load: {e}")
    # Solusi Cadangan (Re-Download jika corrupt)
    if os.path.exists(MODEL_FILE):
        os.remove(MODEL_FILE)
    print("Silakan restart app untuk download ulang model.")

CLASS_NAMES = ['BrownSpot', 'Healthy', 'Hispa', 'LeafBlast']

# ==========================================
# 3. FILTER "GATEKEEPER"
# ==========================================
def check_texture_and_color(image):
    h, w = image.shape[:2]
    total_pixels = h * w
    
    # A. Deteksi Wajah
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    if len(faces) > 0:
        return False, "⛔ TERDETEKSI WAJAH: Sistem hanya menerima foto daun padi."

    # B. Cek Warna
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Hijau
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])
    # Coklat
    lower_brown = np.array([10, 50, 40]) 
    upper_brown = np.array([25, 255, 255])
    
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
    
    green_pixels = np.count_nonzero(mask_green)
    brown_pixels = np.count_nonzero(mask_brown)
    
    green_ratio = (green_pixels / total_pixels) * 100
    total_plant_ratio = ((green_pixels + brown_pixels) / total_pixels) * 100
    
    if green_ratio < 2.0:
         return False, f"⛔ TIDAK ADA KLOROFIL: Unsur hijau hanya {green_ratio:.2f}%. Pastikan objek adalah tanaman hidup."

    if total_plant_ratio < 20.0:
        return False, f"⛔ BUKAN PADI: Objek tanaman terlalu kecil ({total_plant_ratio:.1f}%)."

    # C. Cek Tekstur
    texture_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if texture_score < 40:
        return False, "⛔ TEKSTUR HALUS: Objek terlalu polos/blur."

    return True, "OK"

# ==========================================
# 4. PREPROCESSING
# ==========================================
def crop_center_square(image):
    h, w = image.shape[:2]
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    return image[start_y:start_y+min_dim, start_x:start_x+min_dim]

def prepare_pipeline(path):
    img = cv2.imread(path)
    if img is None: return None, "File rusak"
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    valid, msg = check_texture_and_color(img) 
    if not valid:
        return None, msg
    
    img = crop_center_square(img)
    img_resized = cv2.resize(img, (224, 224))
    
    batch_images = [img_resized, cv2.flip(img_resized, 1)]
    # Pakai preprocess_input milik tf_keras
    batch_x = preprocess_input(np.array(batch_images))
    
    return batch_x, "OK"

def apply_temperature_scaling(logits, temperature=0.5):
    powered = np.power(logits, 1.0 / temperature) 
    normalized = powered / np.sum(powered)
    return normalized

# ==========================================
# 5. ROUTE
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files: return jsonify({'error': 'No file'})
    file = request.files['file']
    if file.filename == '': return jsonify({'error': 'No filename'})

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            data, msg = prepare_pipeline(filepath)
            
            if data is None:
                os.remove(filepath)
                return jsonify({'error': msg})

            preds_batch = model.predict(data)
            final_pred = np.mean(preds_batch, axis=0)
            calibrated_pred = apply_temperature_scaling(final_pred, temperature=0.4)
            
            idx = np.argmax(calibrated_pred)
            confidence = float(calibrated_pred[idx]) * 100
            result_class = CLASS_NAMES[idx]
            
            sorted_indices = np.argsort(final_pred)[::-1]
            gap = final_pred[sorted_indices[0]] - final_pred[sorted_indices[1]]
            
            if confidence < 45.0 or gap < 0.10:
                os.remove(filepath)
                return jsonify({
                    'error': f"⛔ OBJEK MERAGUKAN: AI Bingung (Gap: {gap:.2f})."
                })

            status = "high" if confidence > 70 else "low"

            gc.collect()
            os.remove(filepath)

            return jsonify({
                'class': result_class,
                'confidence': f"{confidence:.2f}%",
                'status': status
            })

        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    ngrok.kill()
    try:
        public_url = ngrok.connect(5000).public_url
        print(f"\n🚀 LINK WEB: {public_url}\n")
        app.run(port=5000)
    except Exception as e:
        print(f"Error: {e}")

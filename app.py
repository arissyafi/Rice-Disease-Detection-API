import os
import cv2
import numpy as np
import gdown
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from werkzeug.utils import secure_filename
from pyngrok import ngrok
import gc

# ==========================================
# 1. KONFIGURASI NGROK (Wajib Diisi)
# ==========================================
NGROK_AUTH_TOKEN = "36rioVsfwi4eUJHyR87ObXPQ1qO_QoEA2WSvxT4wA8KA7j4t"  # <--- GANTI INI
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# ==========================================
# 2. SETUP FLASK & FOLDER
# ==========================================
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# Pastikan folder templates ada
os.makedirs('templates', exist_ok=True)

# ==========================================
# 3. DOWNLOAD & LOAD MODEL
# ==========================================
MODEL_FILE = 'best_model.h5'
FILE_ID = "1wWHM3R1T-3D9tyxYeuRF6UTKaARR0V-n" # ID GDrive Anda
URL = f"https://drive.google.com/uc?id={FILE_ID}"

if not os.path.exists(MODEL_FILE):
    print("⬇️ Sedang mendownload Model dari GDrive...")
    gdown.download(URL, MODEL_FILE, quiet=False, fuzzy=True)

print("⏳ Memuat Model ke Memori...")
# compile=False agar aman dari error custom loss
model = load_model(MODEL_FILE, compile=False)
CLASS_NAMES = ['BrownSpot', 'Healthy', 'Hispa', 'LeafBlast']
print("✅ Model Siap!")

# ==========================================
# 4. PREPROCESSING (Versi Cepat untuk Demo)
# ==========================================
def process_image(path):
    # Baca gambar
    img = cv2.imread(path)
    if img is None: return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize langsung ke 224x224 (Skip GrabCut agar cepat di demo)
    # Jika ingin pakai GrabCut, copy fungsi dari chat sebelumnya ke sini
    img_resized = cv2.resize(img, (224, 224))
    
    # Preprocess EfficientNet
    x = np.expand_dims(img_resized, axis=0)
    x = preprocess_input(x)
    return x

# ==========================================
# 5. ROUTE WEB
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
            # Proses & Prediksi
            data = process_image(filepath)
            preds = model.predict(data)
            
            idx = np.argmax(preds[0])
            confidence = float(np.max(preds[0])) * 100
            result_class = CLASS_NAMES[idx]

            # Bersihkan Memori
            del data
            gc.collect()
            os.remove(filepath)

            return jsonify({
                'class': result_class,
                'confidence': f"{confidence:.2f}%"
            })
        except Exception as e:
            return jsonify({'error': str(e)})

# ==========================================
# 6. JALANKAN SERVER
# ==========================================
if __name__ == '__main__':
    # Matikan tunnel lama
    ngrok.kill()
    
    # Buka Tunnel Baru di Port 5000
    public_url = ngrok.connect(5000).public_url
    print("\n" + "="*50)
    print(f"🚀 LINK WEBSITE PUBLIC: {public_url}")
    print("="*50 + "\n")
    
    # Jalankan Flask
    app.run(port=5000)

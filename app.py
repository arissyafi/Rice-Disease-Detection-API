import os
import cv2
import numpy as np
import gdown
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from werkzeug.utils import secure_filename

# --- INISIALISASI FLASK ---
app = Flask(__name__)

# --- KONFIGURASI ---
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Batas upload 16MB
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================================================
#  BAGIAN DOWNLOAD MODEL DARI GOOGLE DRIVE
# =========================================================
MODEL_FILE = 'best_model.h5'
# ID File Google Drive Anda
FILE_ID = "1wWHM3R1T-3D9tyxYeuRF6UTKaARR0V-n" 
URL = f"https://drive.google.com/uc?id={FILE_ID}"


if not os.path.exists(MODEL_FILE):
    print("⬇️ Model belum ada. Sedang mendownload dari Google Drive...")
    try:
        # fuzzy=True penting untuk bypass warning virus scan Google
        gdown.download(URL, MODEL_FILE, quiet=False, fuzzy=True)
        print("✅ Download Selesai!")
    except Exception as e:
        print(f"❌ Gagal download model: {e}")
else:
    print("✅ File model sudah tersedia di server.")

# =========================================================
#  LOAD MODEL
# =========================================================
print("⏳ Memuat Model EfficientNetB0 ke Memori...")
model = None
try:
    # compile=False wajib karena kita tidak butuh fungsi loss training
    model = load_model(MODEL_FILE, compile=False)
    print("🚀 Model Siap Digunakan!")
except Exception as e:
    print(f"❌ Gagal memuat model: {e}")
    print("Hapus file .h5 dan restart server untuk download ulang.")

# Label Kelas
CLASS_NAMES = ['BrownSpot', 'Healthy', 'Hispa', 'LeafBlast']

# =========================================================
#  BAGIAN PREPROCESSING (PIPELINE ILMIAH)
# =========================================================

def find_leaf_bounding_box_fast(image, downscale_max=800):
    """Mencari kotak pembatas daun (ROI)"""
    h0, w0 = image.shape[:2]
    scale = 1.0
    if max(h0, w0) > downscale_max:
        scale = downscale_max / max(h0, w0)
    small = cv2.resize(image, (int(w0*scale), int(h0*scale)), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    lower_green = np.array([20, 20, 20])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return (int(w0*0.1), int(h0*0.1), int(w0*0.8), int(h0*0.8))

    largest = max(contours, key=cv2.contourArea)
    x_s, y_s, w_s, h_s = cv2.boundingRect(largest)

    x = int(x_s / scale); y = int(y_s / scale)
    w = int(w_s / scale); h = int(h_s / scale)

    pad = int(max(w0, h0) * 0.02)
    x_new = max(0, x - pad); y_new = max(0, y - pad)
    w_new = min(w0 - x_new, w + 2*pad); h_new = min(h0 - y_new, h + 2*pad)
    return (x_new, y_new, w_new, h_new)

def adaptive_grabcut_optimized(image):
    """Segmentasi Background Otomatis"""
    rect = find_leaf_bounding_box_fast(image)
    x, y, rw, rh = rect
    if rw <= 2 or rh <= 2: return image

    roi = image[y:y+rh, x:x+rw]
    roi_h, roi_w = roi.shape[:2]
    
    scale = 1.0
    max_dim = max(roi_h, roi_w)
    if max_dim > 256: 
        scale = 256 / max_dim
    
    small_roi = cv2.resize(roi, (int(roi_w*scale), int(roi_h*scale)), interpolation=cv2.INTER_AREA)
    
    hsv = cv2.cvtColor(small_roi, cv2.COLOR_RGB2HSV)
    mask_init = cv2.inRange(hsv, np.array([20, 20, 20]), np.array([85, 255, 255]))
    mask_gc = np.where(mask_init>0, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype('uint8')
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)

    try:
        cv2.grabCut(small_roi, mask_gc, None, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_MASK)
    except:
        return image

    mask2 = np.where((mask_gc==cv2.GC_FGD)|(mask_gc==cv2.GC_PR_FGD), 1, 0).astype('uint8')
    mask_up = cv2.resize(mask2, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
    
    full_mask = np.zeros(image.shape[:2], dtype='uint8')
    full_mask[y:y+rh, x:x+rw] = mask_up
    
    result = image.copy()
    result[full_mask==0] = [0, 0, 0] 
    return result

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

def full_pipeline(image_path):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None: return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Pipeline: GrabCut -> CLAHE -> Resize -> Preprocess Input
    segmented = adaptive_grabcut_optimized(img_rgb)
    enhanced = apply_clahe(segmented)
    final_img = cv2.resize(enhanced, (224, 224))
    
    x = np.expand_dims(final_img, axis=0)
    x = preprocess_input(x)
    return x

# =========================================================
#  ROUTE & CONTROLLER
# =========================================================

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'File tidak ditemukan'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Tidak ada file dipilih'})

    if file:
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Jika model belum siap (misal download gagal), beri tahu user
            if model is None:
                return jsonify({'error': 'Model AI belum siap. Coba refresh halaman.'})

            processed_data = full_pipeline(filepath)
            
            if processed_data is None:
                return jsonify({'error': 'Gagal memproses gambar'})

            preds = model.predict(processed_data)
            class_idx = np.argmax(preds[0])
            confidence = float(np.max(preds[0])) * 100
            
            result_class = CLASS_NAMES[class_idx]
            os.remove(filepath)

            return jsonify({
                'class': result_class,
                'confidence': f"{confidence:.2f}%",
                'original_file': filename
            })

        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

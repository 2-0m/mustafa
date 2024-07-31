import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from flask import Flask, request, jsonify
import os
import flask_cors
app = Flask(__name__)
flask_cors.CORS(app)
# تحميل النموذج
model_path = 'porn_detector_model.h5'
model = load_model(model_path)

def prepare_image(file_path):
    img = load_img(file_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_image(file_path):
    img = prepare_image(file_path)
    prediction = model.predict(img)[0][0]
    result = 'pornographic' if prediction > 0.5 else 'non-pornographic'
    return result

@app.route('/predict', methods=['POST'])
def predict_pornographic_content():
    try:
        # تحقق من وجود ملف مرفوع
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']

        # حفظ الملف المرفوع مؤقتاً
        file_path = f"temp_{file.filename}"
        file.save(file_path)

        # توقع محتوى الصورة
        result = predict_image(file_path)

        # إزالة الصورة بعد التوقع
        os.remove(file_path)

        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)


import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
import shutil
app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],  # يسمح بالوصول من أي مصدر. قم بتقييد هذا في الإنتاج
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)
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

@app.post("/predict")
async def predict_pornographic_content(file: UploadFile = File(...)):
    try:
        # حفظ الملف المرفوع مؤقتاً
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # توقع محتوى الصورة
        result = predict_image(file_path)

        # إزالة الصورة بعد التوقع
        os.remove(file_path)

        return JSONResponse(content={"result": result})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

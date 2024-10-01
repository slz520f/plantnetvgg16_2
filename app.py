from flask import Flask, request, jsonify
from flask_cors import CORS  # CORSのインポート
import tensorflow as tf
from keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)
CORS(app)  # すべてのオリジンを許可

# 事前に学習済みモデルをロード
MODEL_PATH = '/Users/mame/plantnetvgg16_2/plantnetvgg16_2/plantnet_clone_1/data/MobileNetV2_model.keras'
model = load_model(MODEL_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    image = Image.open(file)
    
    # 画像の前処理
    image = image.resize((224, 224))  # モデルに合わせたサイズ
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # 推論
    prediction = model.predict(image)
    result = np.argmax(prediction, axis=1)[0]

    return jsonify({'prediction': int(result)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)

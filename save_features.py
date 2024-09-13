import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'plantnet_clone.settings')
django.setup()

import numpy as np
from django.core.files import File
from django.conf import settings
from identify.models import PlantImage
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

# グローバルにモデルを定義
model = VGG16(weights='imagenet', include_top=False)

def extract_features(img_path):
    # 画像の前処理
    img_tensor = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img_tensor)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # 特徴量の抽出
    features = model.predict(img_array)
    
    # 特徴量を1次元配列に変換
    features = features.flatten()
    return features

def save_features_from_dataset(dataset_path):
    for root, _, files in os.walk(dataset_path):
        for filename in files:
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, filename)
                
      
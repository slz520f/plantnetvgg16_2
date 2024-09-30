from django.db import models
<<<<<<< HEAD
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

# MobileNetV2モデルをグローバルに準備してキャッシュする
model = MobileNetV2(weights='imagenet', include_top=False)
=======
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

# モデルをグローバルに準備してキャッシュする
model = VGG16(weights='imagenet', include_top=False)
>>>>>>> c747ad193cedbf2e526a8863f75191d56d9e590a

class PlantImage(models.Model):
    image = models.ImageField(upload_to='images/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    features = models.BinaryField(default=b'')  # 特徴量をバイナリとして保存

    def extract_features(self):
        # 画像の前処理
        img_tensor = image.load_img(self.image.path, target_size=(224, 224))
        img_array = image.img_to_array(img_tensor)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # 特徴量の抽出
        features = model.predict(img_array)

        # 特徴量を1次元配列に変換
        features = features.flatten()
        if np.all(features == 0):
            print("Warning: All features are zero!")

        return features

class PlantFeature(models.Model):
    features = models.BinaryField()  # 特徴量をバイナリとして保存

    def save_features(self, features):
        # 特徴量をバイナリ形式で保存
        features_bin = features.tobytes()
        self.features = features_bin
        self.save()

    def get_features(self):
        # バイナリデータを numpy 配列に変換して返す
        features_array = np.frombuffer(self.features, dtype=np.float32)
<<<<<<< HEAD
        return features_array
=======
        return features_array
>>>>>>> c747ad193cedbf2e526a8863f75191d56d9e590a

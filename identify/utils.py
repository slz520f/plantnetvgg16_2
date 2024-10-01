import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.keras.utils import load_img, img_to_array
from identify.models import PlantImage

# ... 残りのコード ...


model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
def extract_features(image_path):
    img = load_img(image_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features.flatten()

def identify_plant(uploaded_image_path):
    # アップロードされた画像から特徴量を抽出
    features = extract_features(uploaded_image_path)
    
    # データベースからすべてのPlantImageを取得
    all_images = PlantImage.objects.all()
    
    closest_match = None
    min_distance = float('inf')
    
    for plant_image in all_images:
        db_features = np.frombuffer(plant_image.features, dtype=np.float32)
        distance = np.linalg.norm(features - db_features)
        if distance < min_distance:
            min_distance = distance
            closest_match = plant_image

    return closest_match

import tensorflow as tf
from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse
from .forms import UploadImageForm
from PIL import Image
import numpy as np
import json
import os
import requests

# Djangoプロジェクトのベースディレクトリを取得
base_dir = settings.BASE_DIR

# データフォルダへのパスを設定
data_dir = os.path.join(base_dir, 'data')

# クラスインデックスのパスを設定
class_indices_path = os.path.join(data_dir, 'class_indices.json')

# クラスインデックスのロード
with open(class_indices_path, 'r') as f:
    class_indices = json.load(f)

# クラスインデックスを反転させる
class_indices_reversed = {v: k for k, v in class_indices.items()}

# モデルをロード
model = tf.keras.models.load_model('data/MobileNetV2_model.keras')

def preprocess_image(image):
    """画像を前処理してモデル入力用に変換する。"""
    image = Image.open(image).resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def identify_plant(request):
    # JSONファイルのパスを設定
    metadata_file_path = os.path.join(base_dir, 'data', 'plantnet300K_metadata.json')
    species_file_path = os.path.join(base_dir, 'data', 'plantnet300K_species_id_2_name.json')

    # JSONファイルの読み込み
    with open(metadata_file_path, 'r') as f:
        metadata = json.load(f)

    with open(species_file_path, 'r') as f:
        species_data = json.load(f)

    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            # Flask APIに画像を送信して予測を取得する
            api_url = 'http://52.192.88.39/predict'  # EC2のパブリックIP
            files = {'image': image}
            response = requests.post(api_url, files=files)
            prediction = response.json()

            predicted_class_id = prediction.get('predicted_class', None)

            if predicted_class_id is None:
                return JsonResponse({"name": "エラーが発生しました", "description": "予測結果が取得できませんでした", "metadata": "情報が見つかりません"})

            print(f"Predicted class ID from API: {predicted_class_id}")

            # クラスIDから植物情報を取得
            try:
                # クラスIDからspecies_infoを取得
                species_info = species_data.get(predicted_class_id, None)
                print(f"Species Info: {species_info}")

                if species_info is None:
                    raise ValueError("Species info not found in species_data")

                # メタデータから該当するspecies_idを持つデータを検索
                plant_metadata = next((entry_value for entry_value in metadata.values() if entry_value["species_id"] == predicted_class_id), None)

                if plant_metadata is None:
                    raise ValueError(f"Metadata not found for species_id: {predicted_class_id}")

                print(f"Plant Metadata: {plant_metadata}")

                predicted_class_name = species_info

            except Exception as e:
                print(f"An error occurred: {e}")
                predicted_class_name = 'エラーが発生しました'
                plant_metadata = '情報が見つかりません'

            # すべてのメタデータ情報をJSONレスポンスに含める
            result = {
                "name": predicted_class_name,
                "description": f"This is a description of the predicted plant: {predicted_class_name}.",
                "metadata": plant_metadata  # すべてのメタデータ情報を含める
            }

            print("Result JSON:", result)
            return JsonResponse(result)

    else:
        form = UploadImageForm()

    return render(request, 'identify/upload.html', {'form': form})

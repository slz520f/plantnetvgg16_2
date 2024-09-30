import tensorflow as tf
from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse
from .forms import UploadImageForm
from PIL import Image
import numpy as np
import json
import os
import requests  # 追加: requestsモジュールをインポート

# Djangoプロジェクトのベースディレクトリを取得
base_dir = settings.BASE_DIR

# データフォルダへのパスを設定
data_dir = os.path.join(base_dir, 'data')

# ファイルパスを設定
class_indices_path = os.path.join(data_dir, 'class_indices.json')

# クラスインデックスのロード
with open(class_indices_path, 'r') as f:
    class_indices = json.load(f)

# クラスインデックスを反転させて、インデックスからクラスIDに変換できるようにする
class_indices_reversed = {v: k for k, v in class_indices.items()}

# モデルをロード
model = tf.keras.models.load_model('data/MobileNetV2_model.keras')

def preprocess_image(image):
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

    # デバッグ用: species_data の内容をログに出力
    print("Species Data:", species_data)

    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            image_array = preprocess_image(image)
            predictions = model.predict(image_array)

            # 予測結果のログ
            predicted_index = np.argmax(predictions)
            print(f"Predicted index: {predicted_index}")

            # インデックスからクラスIDに変換
            predicted_class_id = class_indices_reversed.get(predicted_index, "Unknown Class")
            print(f"Predicted class ID: {predicted_class_id}")

            # クラスIDから植物情報を取得
            try:
                species_info = species_data.get(predicted_class_id, '情報が見つかりません')
                if isinstance(species_info, str):
                    predicted_class_name = species_info
                else:
                    print(f"Species info is not a string: {species_info}")
                    predicted_class_name = '情報が見つかりません'
            except Exception as e:
                print(f"An error occurred: {e}")
                predicted_class_name = 'エラーが発生しました'

            # 結果をAPI経由で取得する部分
            api_url = 'http://57.180.20.167:5000/predict'  # EC2のパブリックIPを指定
            input_data = {'input': predicted_class_id}  # 予測されたクラスIDをAPIに送信
            try:
                response = requests.post(api_url, json=input_data)
                if response.status_code == 200:
                    api_result = response.json()  # APIからの結果を取得
                    print("API Result:", api_result)
                    # APIから取得した結果を使って処理を続けることができます
                    # 例えば、以下のようにして表示する内容を決めることができます
                    result = {
                        "name": api_result.get('name', predicted_class_name),
                        "description": api_result.get('description', "この植物の説明は取得できませんでした。")
                    }
                else:
                    print(f"API呼び出しに失敗しました。ステータスコード: {response.status_code}")
                    result = {"name": predicted_class_name, "description": "API呼び出しに失敗しました。"}
            except Exception as e:
                print(f"API呼び出し中にエラーが発生しました: {e}")
                result = {"name": predicted_class_name, "description": "API呼び出し中にエラーが発生しました。"}

            print("Result JSON:", result)
            return JsonResponse(result)

    else:
        form = UploadImageForm()

    return render(request, 'identify/upload.html', {'form': form})

import tensorflow as tf
from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse
from .forms import UploadImageForm
from PIL import Image
import numpy as np
import json
import os

# Djangoプロジェクトのベースディレクトリを取得
base_dir = settings.BASE_DIR

# データフォルダへのパスを設定
data_dir = os.path.join(base_dir, 'data')

<<<<<<< HEAD
# クラスインデックスのパスを設定
=======
# ファイルパスを設定
>>>>>>> c747ad193cedbf2e526a8863f75191d56d9e590a
class_indices_path = os.path.join(data_dir, 'class_indices.json')

# クラスインデックスのロード
with open(class_indices_path, 'r') as f:
    class_indices = json.load(f)

<<<<<<< HEAD
# クラスインデックスを反転させる
class_indices_reversed = {v: k for k, v in class_indices.items()}

# モデルをロード
model = tf.keras.models.load_model('data/mobilenetv2_model.tflite')

def preprocess_image(image):
    """画像を前処理してモデル入力用に変換する。"""
=======
# クラスインデックスを反転させて、インデックスからクラスIDに変換できるようにする
class_indices_reversed = {v: k for k, v in class_indices.items()}


# モデルをロード
model = tf.keras.models.load_model('data/plant_model.h5')

def preprocess_image(image):
>>>>>>> c747ad193cedbf2e526a8863f75191d56d9e590a
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

<<<<<<< HEAD
=======
    # デバッグ用: species_data の内容をログに出力
    print("Species Data:", species_data)

>>>>>>> c747ad193cedbf2e526a8863f75191d56d9e590a
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
<<<<<<< HEAD
            predicted_class_id = class_indices_reversed.get(predicted_index, None)
            if predicted_class_id is None:
                print("Predicted class ID not found")
                return JsonResponse({"name": "エラーが発生しました", "description": "Class ID not found", "metadata": "情報が見つかりません"})

=======
            predicted_class_id = class_indices_reversed.get(predicted_index, "Unknown Class")
>>>>>>> c747ad193cedbf2e526a8863f75191d56d9e590a
            print(f"Predicted class ID: {predicted_class_id}")

            # クラスIDから植物情報を取得
            try:
<<<<<<< HEAD
                # クラスIDからspecies_infoを取得
                species_info = species_data.get(predicted_class_id, None)
                print(f"Species Info: {species_info}")

                if species_info is None:
                    raise ValueError("Species info not found in species_data")

                # メタデータから該当するspecies_idを持つデータを検索
                plant_metadata = None
                for entry_key, entry_value in metadata.items():
                    if entry_value["species_id"] == predicted_class_id:
                        plant_metadata = entry_value
                        break

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
=======
                species_info = species_data.get(predicted_class_id, '情報が見つかりません')
                if isinstance(species_info, str):
                    predicted_class_name = species_info
                else:
                    print(f"Species info is not a string: {species_info}")
                    predicted_class_name = '情報が見つかりません'
            except Exception as e:
                print(f"An error occurred: {e}")
                predicted_class_name = 'エラーが発生しました'

            result = {"name": predicted_class_name, "description": "This is a description of the predicted plant: " + predicted_class_name}
>>>>>>> c747ad193cedbf2e526a8863f75191d56d9e590a
            print("Result JSON:", result)
            return JsonResponse(result)

    else:
        form = UploadImageForm()

<<<<<<< HEAD
    return render(request, 'identify/upload.html', {'form': form})
=======
    return render(request, 'identify/upload.html', {'form': form})
>>>>>>> c747ad193cedbf2e526a8863f75191d56d9e590a

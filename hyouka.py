import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# モデルの読み込み
model = tf.keras.models.load_model('/Users/mame/plantnetvgg16_2/plantnet_clone_1/data/MobileNetV2_model.keras')

# テストデータのディレクトリ
test_image_dir = '/Users/mame/Downloads/plantnet_300K/images/test'

# 画像データの前処理
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# テストデータを生成
test_generator = test_datagen.flow_from_directory(
    test_image_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',  # one-hotエンコーディングされたラベルの場合
    shuffle=False
)

# モデルの評価
loss, accuracy = model.evaluate(test_generator)

# 結果を表示
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# 評価結果をグラフに表示
labels = ['Test Loss', 'Test Accuracy']
values = [loss, accuracy]

plt.bar(labels, values)
plt.title('Model Evaluation Results')
plt.show()

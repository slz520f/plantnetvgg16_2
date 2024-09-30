import tensorflow as tf

# 学習済みモデルの読み込み
model = tf.keras.models.load_model('MobileNetV2_model.keras')

# TensorFlow Liteへの変換
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# TensorFlow Liteモデルの保存
with open('mobilenetv2_model.tflite', 'wb') as f:
    f.write(tflite_model)

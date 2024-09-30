import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import json
import os
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt
# Mixed precision policy
mixed_precision.set_global_policy('mixed_float16')

# MobileNetV2 のインポート
from tensorflow.keras.applications import MobileNetV2

# GPU メモリの設定
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# データパスの設定
train_data_dir = '/Users/mame/Downloads/plantnet_300K/images/train'
validation_data_dir = '/Users/mame/Downloads/plantnet_300K/images/validation'

# データジェネレーターの設定
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],  # 明るさの調整
    channel_shift_range=0.2        # カラーチャネルのシフト
)


val_datagen = ImageDataGenerator(rescale=1./255)

# データの読み込み
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=8,
    class_mode='categorical',
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(224, 224),
    batch_size=8,
    class_mode='categorical'
)

# クラス数の設定
num_classes = len(train_generator.class_indices)

# MobileNetV2ベースモデルの作成
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 新しいトップ層の追加
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)  # 中間層
predictions = Dense(num_classes, activation='softmax')(x)  # 出力層

# モデルの定義
model = Model(inputs=base_model.input, outputs=predictions)

# MobileNetV2ベースモデルの一部のレイヤーをトレーニング可能にする
for layer in base_model.layers[:100]:  # 最初の100層だけをフリーズ
    layer.trainable = False
for layer in base_model.layers[100:]:  # それ以降のレイヤーはトレーニング可能に
    layer.trainable = True


# モデルのコンパイル
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# コールバック設定
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

# steps_per_epoch と validation_steps を設定
steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = validation_generator.samples // validation_generator.batch_size

# モデルのトレーニング
history = model.fit(
    train_generator,  # 修正: tf.data.Dataset.from_generator を削除
    steps_per_epoch=steps_per_epoch,

    epochs=5,

    validation_data=validation_generator,  # 修正: tf.data.Dataset.from_generator を削除
    validation_steps=validation_steps,
    callbacks=[early_stopping, reduce_lr]
)

# モデルとクラスインデックスの保存
os.makedirs('data', exist_ok=True)
model.save('data/MobileNetV2_model.keras')

with open('data/class_indices.json', 'w') as f:
    json.dump(train_generator.class_indices, f)
# 訓練履歴から精度と損失を取得
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# エポックの数を計算
epochs_range = range(len(acc))

# グラフを描画
plt.figure(figsize=(12, 6))

# 精度のグラフ
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# 損失のグラフ
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

# グラフを表示
plt.show()
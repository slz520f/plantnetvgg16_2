import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import json
import os
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import mixed_precision

from tensorflow.keras.regularizers import l2


# Mixed precision policy
mixed_precision.set_global_policy('mixed_float16')

# ResNet152 のインポート
from tensorflow.keras.applications import ResNet101

# GPU メモリの設定
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# データパスの設定
train_data_dir = '/Users/mame/Downloads/plantnet_300K/images/train_2'
validation_data_dir = '/Users/mame/Downloads/plantnet_300K/images/validation_2'

# データジェネレーターの設定（拡張）
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8,1.2],
    channel_shift_range=50.0,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# データの読み込み
# ImageDataGenerator の workers 設定
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),  # 画像サイズを小さくしてパフォーマンスを確認
    batch_size=32,  # Batch size を減らしてGPUの負荷を軽減
    class_mode='categorical',
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# クラス数の設定
num_classes = len(train_generator.class_indices)

# ResNet152ベースモデルの作成
base_model = ResNet101(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 新しいトップ層の追加（L2正則化の追加）
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu', kernel_regularizer=l2(0.01))(x)  # L2正則化
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.01))(x)  # L2正則化

# モデルの定義
model = Model(inputs=base_model.input, outputs=predictions)

# ResNet152モデルの全層をトレーニング可能にする
for layer in base_model.layers:
    layer.trainable = True

# 学習率スケジューラーの設定
initial_learning_rate = 0.001
lr_schedule = ExponentialDecay(
    initial_learning_rate,
    decay_steps=10000,
    decay_rate=0.9,
    staircase=True
)

# Optimizerとlearning rate schedulerを再設定
model.compile(optimizer=Adam(learning_rate=1e-5),  # 小さい学習率でファインチューニング
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# コールバック設定
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

# steps_per_epoch と validation_steps を設定
steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = validation_generator.samples // validation_generator.batch_size

# 各クラスのサンプル数を取得
class_counts = train_generator.classes
count = {i: 0 for i in range(num_classes)}

for cls in class_counts:
    count[cls] += 1

# クラスの重みを計算
class_weights = {i: max(train_generator.samples / (num_classes * count[i]), 1.0) for i in range(num_classes)}

# モデルのトレーニング

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[early_stopping],
    
)

# モデルとクラスインデックスの保存
os.makedirs('data', exist_ok=True)
model.save('data/ResNet152_model.keras')

with open('data/class_indices.json', 'w') as f:
    json.dump(train_generator.class_indices, f)

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow_addons.losses import CategoricalFocalLoss

import json
import os

# データセットのパス
train_data_dir = 'Users/mame/Downloads/plantnet_300K/images/train'
validation_data_dir = 'Users/mame/Downloads/plantnet_300K/images/validation'
test_data_dir = 'Users/mame/Downloads/plantnet_300K/images/test'

# ハイパーパラメータ
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
INITIAL_LEARNING_RATE = 0.001

# データ拡張の設定
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
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    validation_data_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# クラス数の設定
num_classes = len(train_generator.class_indices)

# モデルの構築
base_model = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(*IMAGE_SIZE, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# ベースモデルの重みを一部固定
for layer in base_model.layers[:-30]:
    layer.trainable = False

# 学習率スケジューラーの設定
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    INITIAL_LEARNING_RATE,
    decay_steps=10000,
    decay_rate=0.9,
    staircase=True)

# モデルのコンパイル
optimizer = Adam(learning_rate=lr_schedule)
loss = CategoricalFocalLoss(gamma=2.0, alpha=0.25)  # Focal Lossの使用
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# コールバックの設定
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

# モデルの訓練
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=[early_stopping, reduce_lr]
)

# モデルの保存
os.makedirs('output', exist_ok=True)
model.save('output/plantnet_300k_model.h5')

# クラスインデックスの保存
with open('output/class_indices.json', 'w') as f:
    json.dump(train_generator.class_indices, f)

# テストデータの評価
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# top-k accuracy の評価関数
def top_k_accuracy(y_true, y_pred, k=5):
    return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=k)

# average-k accuracy の評価関数
def average_k_accuracy(y_true, y_pred, k=5):
    sorted_indices = tf.argsort(y_pred, direction='DESCENDING')
    topk_indices = sorted_indices[:, :k]
    topk_values = tf.gather(y_pred, topk_indices, batch_dims=1)
    threshold = tf.reduce_min(topk_values, axis=1)
    return tf.reduce_mean(tf.cast(tf.reduce_any(tf.greater_equal(y_pred, tf.expand_dims(threshold, 1)) & tf.cast(y_true, tf.bool), axis=1), tf.float32))

# テストデータでの評価
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc:.4f}")

# top-5 accuracy と average-5 accuracy の計算
y_pred = model.predict(test_generator)
y_true = test_generator.classes
y_true_one_hot = tf.keras.utils.to_categorical(y_true, num_classes=num_classes)

top_5_acc = top_k_accuracy(y_true_one_hot, y_pred, k=5)
avg_5_acc = average_k_accuracy(y_true_one_hot, y_pred, k=5)

print(f"Top-5 accuracy: {top_5_acc:.4f}")
print(f"Average-5 accuracy: {avg_5_acc:.4f}")

# 結果の保存
results = {
    "test_accuracy": float(test_acc),
    "top_5_accuracy": float(top_5_acc),
    "average_5_accuracy": float(avg_5_acc)
}

with open('output/evaluation_results.json', 'w') as f:
    json.dump(results, f)

print("Training and evaluation completed. Results saved in the 'output' directory.")


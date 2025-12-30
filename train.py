import os
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import kagglehub

path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

print("Path to dataset files:", path)

# Veri Yolu
data_dir = "/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray"
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
val_dir = os.path.join(data_dir, 'val')

# Görüntü Ayarları
IMG_SIZE = (224, 224)
BATCH_SIZE = 128

print("Veriler hazırlanıyor...")

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_generator = test_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224), 
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# --- TRANSFER LEARNING (VGG16) ---
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Hazır modelin ağırlıklarını dondurma
for layer in base_model.layers:
    layer.trainable = False

# Yeni başlık
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x) 

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Eğitim başlıyor.")

history = model.fit(
    train_generator,
    epochs=5,
    validation_data=val_generator
)
print("Eğitim sonra erdi.")

# Model Testi
loss, acc = model.evaluate(test_generator)

print(f"Test sonucu: %{acc*100:.2f}")

# Modeli Kaydetme
model.save("xray_model.h5")
print("Model hazır: xray_model.h5 ")
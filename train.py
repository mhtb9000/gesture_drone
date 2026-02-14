import tensorflow as tf
from tensorflow.keras import layers,models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
tf.random.set_seed(41)
np.random.seed(41)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    brightness_range=[0.8,1.2],
    horizontal_flip=False
)

val_datagen = ImageDataGenerator(
rescale=1./255
)

reduce_lr = ReduceLROnPlateau(
monitor='val_loss',
factor=0.3,
patience=3,
min_lr=0.00001
)

train_data = train_datagen.flow_from_directory(
    "data/train",
    target_size=(160,160),
    batch_size=32,
    class_mode='categorical',
    shuffle=True,
    seed=41
)

val_data = val_datagen.flow_from_directory(
    "data/val",
    target_size=(160,160),
    batch_size=32,
    class_mode='categorical',
    shuffle=False,
    seed=41
)



class_weights = compute_class_weight(
'balanced',
classes=np.unique(train_data.classes),
y=train_data.classes
)

class_weights = dict(enumerate(class_weights))



model=models.Sequential([

layers.Conv2D(32,(3,3),
padding='same',
activation='relu',
input_shape=(160,160,3)),
layers.BatchNormalization(),
layers.MaxPooling2D(),

layers.Conv2D(64,(3,3),padding='same',activation='relu'),
layers.BatchNormalization(),
layers.MaxPooling2D(),

layers.Conv2D(128,(3,3),padding='same',activation='relu'),
layers.BatchNormalization(),
layers.MaxPooling2D(),

layers.Conv2D(256,(3,3),padding='same',activation='relu'),
layers.BatchNormalization(),
layers.MaxPooling2D(),

layers.Flatten(),

layers.Dense(256,activation='relu'),
layers.Dropout(0.5),

layers.Dense(128,activation='relu'),
layers.Dropout(0.3),

layers.Dense(4,activation='softmax')
])

model.compile(
optimizer='adam',
loss='categorical_crossentropy',
metrics=['accuracy']
)



early_stop = EarlyStopping(
monitor='val_loss',
patience=5,
restore_best_weights=True
)


history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=60,
    callbacks=[early_stop,reduce_lr],
    class_weight=class_weights
)

model.save("model.h5")
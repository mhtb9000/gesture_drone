import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix,classification_report
import seaborn as sns
import matplotlib.pyplot as plt

model=tf.keras.models.load_model("model.h5")

datagen=ImageDataGenerator(rescale=1./255)

val_data=datagen.flow_from_directory(
"data/val",
target_size=(160,160),
batch_size=32,
class_mode='categorical',
shuffle=False
)

pred=model.predict(val_data)

y_pred=np.argmax(pred,axis=1)
y_true=val_data.classes

print(classification_report(
y_true,
y_pred,
target_names=val_data.class_indices.keys()
))

cm=confusion_matrix(y_true,y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(cm,
annot=True,
fmt='d',
xticklabels=val_data.class_indices.keys(),
yticklabels=val_data.class_indices.keys())

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
#downloading the dataset
import kagglehub
path = kagglehub.dataset_download("andrewmvd/bone-marrow-cell-classification")
print("Path to dataset files:", path)

#getting the path of directories and sub-directories
import os
print("Contents of the dataset directory:", os.listdir(path))
def list_all_files(root_dir):
    for root, dirs, files in os.walk(root_dir):
        print(f"Contents of {root}:")
        print(dirs)
        print(files[:10])  
list_all_files(path)

#training and evaluating the 2 CNN models step by step
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

dataset_dir=os.path.join(path, 'bone_marrow_cell_dataset')
subdirectories=os.listdir(dataset_dir)
print("Subdirectories containing images:", subdirectories)

image_files=[]
labels=[]
for subdir in subdirectories:
    subdir_path=os.path.join(dataset_dir, subdir)
    for file in os.listdir(subdir_path):
        if file.endswith('.jpg'):
            image_files.append(os.path.join(subdir_path, file))
            labels.append(subdir)
print("Number of image files:", len(image_files))
data=pd.DataFrame({
    'filename': image_files,
    'label': labels
})
print(data.head())

train_df, test_df = train_test_split(data, test_size=0.3, random_state=42)

print("Number of training samples after split:", len(train_df))
print("Number of testing samples after split:", len(test_df))


datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator=datagen.flow_from_dataframe(
    train_df,
    x_col='filename',
    y_col='label',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator=datagen.flow_from_dataframe(
    train_df,
    x_col='filename',
    y_col='label',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

test_generator=datagen.flow_from_dataframe(
    test_df,
    x_col='filename',
    y_col='label',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)


def build_custom_model():
    model=Sequential([
        Conv2D(32,(3,3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(2,2),
        Conv2D(64,(3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128,(3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(512,activation='relu'),
        Dropout(0.5),
        Dense(len(train_generator.class_indices), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

custom_model = build_custom_model()
custom_model.summary()

base_model=VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(512, activation='relu')(x)
x=Dropout(0.5)(x)
predictions=Dense(len(train_generator.class_indices), activation='softmax')(x)

vgg_model=Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable=False

vgg_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
vgg_model.summary()

#training custom model
custom_history = custom_model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)

#training VGG-16
vgg_history = vgg_model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)

def evaluate_model(model, test_generator, model_name):
    test_generator.reset()
    pred=model.predict(test_generator)
    pred_classes=np.argmax(pred, axis=1)
    true_classes=test_generator.classes
    class_labels=list(test_generator.class_indices.keys())

    print(f"{model_name} Classification Report:")
    print(classification_report(true_classes, pred_classes, target_names=class_labels))

    accuracy=accuracy_score(true_classes, pred_classes)
    precision=precision_score(true_classes, pred_classes, average='weighted')
    recall=recall_score(true_classes, pred_classes, average='weighted')
    f1=f1_score(true_classes, pred_classes, average='weighted')

    cm=confusion_matrix(true_classes, pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    return accuracy, precision, recall, f1

custom_metrics=evaluate_model(custom_model, test_generator, "Custom CNN")
vgg_metrics=evaluate_model(vgg_model, test_generator, "VGG16")

results_df=pd.DataFrame({
    'Model': ['Custom CNN', 'VGG16'],
    'Accuracy': [custom_metrics[0], vgg_metrics[0]],
    'Precision': [custom_metrics[1], vgg_metrics[1]],
    'Recall': [custom_metrics[2], vgg_metrics[2]],
    'F1-Score': [custom_metrics[3], vgg_metrics[3]]
})
print(results_df)

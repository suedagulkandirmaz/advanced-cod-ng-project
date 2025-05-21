import os
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Veri hazırlama
def prepare_data(path, image_size=(128, 128), batch=32):
    data_gen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.3,
        rotation_range=50,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2
    )

    train_data = data_gen.flow_from_directory(
        path,
        target_size=image_size,
        batch_size=batch,
        class_mode='categorical',
        subset='training'
    )

    val_data = data_gen.flow_from_directory(
        path,
        target_size=image_size,
        batch_size=batch,
        class_mode='categorical',
        subset='validation'
    )

    return train_data, val_data

# Model oluşturma
def create_CNN_model(input_shape=(128, 128, 3), num_classes=2):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0006),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Eğitim grafiği
def plot_training_graph(history, save=False, file_name="training_plot.png"):
    plt.style.use("ggplot")

    plt.figure(figsize=(14, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='green')
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', color='red')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    if save:
        plt.savefig(file_name)
        print(f"Plot saved as {file_name}")
    else:
        plt.show()

# Ana fonksiyon
if __name__ == "__main__":
    dataset_path = 'PlantVillage'  # Dataset klasörünün adı
    train_data, val_data = prepare_data(dataset_path)

    num_classes = len(train_data.class_indices)
    model = create_CNN_model(num_classes=num_classes)

    history = model.fit(
        train_data,
        epochs=10,
        validation_data=val_data
    )

    # Eğitim grafiğini göster
    plot_training_graph(history, save=True)

    # Modeli kaydet
    model.save("plant_disease_model.h5")

    # Sınıf bilgilerini kaydet
    with open("class_indices.json", "w") as f:
        json.dump(train_data.class_indices, f)

    print("Model ve sınıf etiketleri kaydedildi.")


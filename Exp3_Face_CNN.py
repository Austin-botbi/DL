from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

TRAIN = "/content/drive/MyDrive/Face Images/Final Training Images"  # change if needed

train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(TRAIN, target_size=(64,64), batch_size=32, class_mode="categorical")

n_classes = len(train_gen.class_indices)
m = Sequential([
    Conv2D(32,(3,3),activation="relu", input_shape=(64,64,3)),
    MaxPooling2D(2,2),
    Conv2D(64,(3,3),activation="relu"),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation="relu"),
    Dense(n_classes, activation="softmax")
])
m.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
m.fit(train_gen, epochs=5)
m.save("face_model.h5")
print("Saved face_model.h5")

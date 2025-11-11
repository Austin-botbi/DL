import numpy as np, pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

data = pd.read_csv("/content/drive/MyDrive/A_Z Handwritten Data.csv").astype("float32")
X = data.drop("0", axis=1).values
y = data["0"].values
X = X.reshape(-1,28,28,1); y = to_categorical(y, 26)
tx, ex, ty, ey = train_test_split(X, y, test_size=0.2)

m = Sequential([
    Conv2D(32,(3,3),activation="relu", input_shape=(28,28,1)),
    MaxPooling2D(2,2),
    Conv2D(64,(3,3),activation="relu", padding="same"),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(26, activation="softmax")
])
m.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
m.fit(tx, ty, epochs=1, validation_data=(ex, ey))
print(m.evaluate(ex,ey,verbose=0))

import numpy as np
from keras.models import Sequential
from keras.layers import Dense

X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype="float32")
y = np.array([[0],[1],[1],[0]], dtype="float32")

m = Sequential([Dense(8, input_dim=2, activation="relu"),
                Dense(1, activation="sigmoid")])
m.compile(loss="mean_squared_error", optimizer="adam", metrics=["binary_accuracy"])
m.fit(X, y, epochs=500, verbose=2)
print(m.predict(X).round())

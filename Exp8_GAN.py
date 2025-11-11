from keras.models import Sequential, Model
from keras.layers import Dense, LeakyReLU, Flatten, Reshape, Input
import numpy as np

z_dim = 100
g = Sequential([
    Dense(128, input_dim=z_dim),
    LeakyReLU(0.2),
    Dense(28*28, activation='tanh'),
    Reshape((28,28,1))
])
d = Sequential([
    Flatten(input_shape=(28,28,1)),
    Dense(128),
    LeakyReLU(0.2),
    Dense(1, activation='sigmoid')
])
d.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

z = Input(shape=(z_dim,))
img = g(z)
d.trainable = False
valid = d(img)
combined = Model(z, valid)
combined.compile(optimizer='adam', loss='binary_crossentropy')
print("GAN components ready")

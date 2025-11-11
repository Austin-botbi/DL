from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
import numpy as np

data = "Jack and Jill went up the hill to fetch a pail of water"
t = Tokenizer(); t.fit_on_texts([data])
enc = t.texts_to_sequences([data])[0]

X = [enc[i-1] for i in range(1,len(enc))]
y = [enc[i] for i in range(1,len(enc))]
X = np.array(X)
y = to_categorical(y, num_classes=len(t.word_index)+1)

m = Sequential([Embedding(len(t.word_index)+1, 10, input_length=1),
                LSTM(50),
                Dense(len(t.word_index)+1, activation="softmax")])
m.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
m.fit(X, y, epochs=100, verbose=0)

# test: predict next word after 'Jill'
seq = t.texts_to_sequences(["Jill"])[0]
pred = m.predict(np.array(seq), verbose=0)
print("Predicted index:", pred.argmax(axis=1))

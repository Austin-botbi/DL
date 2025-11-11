from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense

VOCAB, MAXLEN = 10000, 200
(train_x, train_y), (test_x, test_y) = imdb.load_data(num_words=VOCAB)
train_x = pad_sequences(train_x, MAXLEN)
test_x = pad_sequences(test_x, MAXLEN)

m = Sequential([Embedding(VOCAB, 32),
                SimpleRNN(32),
                Dense(1, activation="sigmoid")])
m.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
m.fit(train_x, train_y, epochs=3, validation_data=(test_x, test_y))
print("Eval:", m.evaluate(test_x, test_y, verbose=0))

import numpy as np
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

inputs = ["I love coding", "This is a pen", "She sings well"]
targets = ["PRP VB NNP", "DT VBZ DT NN", "PRP VBZ RB"]

# build vocabularies
iw = sorted(set(" ".join(inputs).split()))
tw = sorted(set(" ".join(targets).split()) | {"<sos>","<eos>"})
iw2i = {w:i for i,w in enumerate(iw)}
tw2i = {w:i for i,w in enumerate(tw)}

enc = [[iw2i[w] for w in s.split()] for s in inputs]
dec = [[tw2i[w] for w in s.split()] for s in targets]

enc = pad_sequences(enc)
dec = pad_sequences(dec)

en_in = Input(shape=(None,))
de_in = Input(shape=(None,))

en_e = Embedding(len(iw), 16)(en_in)
de_e = Embedding(len(tw), 16)(de_in)

en_o, h, c = LSTM(32, return_state=True)(en_e)
de_o = LSTM(32, return_sequences=True)(de_e, initial_state=[h,c])
out = Dense(len(tw), activation="softmax")(de_o)

m = Model([en_in, de_in], out)
m.compile(optimizer="adam", loss="categorical_crossentropy")
print("POS model built (toy).")

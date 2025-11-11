import numpy as np
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

src = ["I love coding", "This is a pen", "She sings well"]
tgt = ["Ich liebe Coden", "Das ist ein Stift", "Sie singt gut"]

sw = sorted(set(" ".join(src).split()))
tw = sorted(set(" ".join(tgt).split()) | {"<sos>","<eos>"})
sw2i = {w:i for i,w in enumerate(sw)}
tw2i = {w:i for i,w in enumerate(tw)}

enc = pad_sequences([[sw2i[w] for w in s.split()] for s in src])
dec = pad_sequences([[tw2i[w] for w in t.split()] for t in tgt])

en_in = Input(shape=(None,))
de_in = Input(shape=(None,))
en_e = Embedding(len(sw),16)(en_in)
de_e = Embedding(len(tw),16)(de_in)
en_o,h,c = LSTM(32, return_state=True)(en_e)
de_o = LSTM(32, return_sequences=True)(de_e, initial_state=[h,c])
out = Dense(len(tw), activation="softmax")(de_o)

m = Model([en_in,de_in], out)
m.compile(optimizer="adam", loss="categorical_crossentropy")
print("Translation toy model ready.")

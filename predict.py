import streamlit as st
from functools import partial
import torch
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

alphabet = 'NACGT'
dna2int = { a: i for a, i in zip(alphabet, range(5))}
dnaseq_to_intseq = partial(map, dna2int.get)
model = torch.jit.load('dna_model.pt')
model.eval()
title = st.text_input('DNA Sequence', '')
pred=0
if len(title)>0:
# lower the text
    x=[*title]
    print(x)
    pred_x=[]
    for i in x:
        s=dnaseq_to_intseq(i)
        pred_x.extend(list(s))
# pad
    print(pred_x)
    pred_x=[pred_x]
    x = pad_sequences(pred_x, maxlen=128)
# create dataset
    x = torch.tensor(x, dtype=torch.long)
    pred = model(x.float()).detach()
    print(pred)

st.write('Predicted Result : ',round(float(pred),2) )
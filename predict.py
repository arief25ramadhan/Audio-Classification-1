import pickle
import os
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from python_speech_features import mfcc
from keras.models import load_model
import panda as pd
from sklearn.metrics import accuracy_score

def build_prediction(audio_dir):
    y_true = []
    y_pred = []
    fn_prob = []
    
    print('Extracting features from audio')
    for fn in tqdm(on.listdir(audio_dir)):
        rate, wav = wavfile.read(os.path.join(audio_dir, fn))
        label = fn2class[fn]
        c = classes.index(label)
        y_prob = []
        
        for i in range(0,wav.shape[0]-config.step, config.step):
            sample = wav[i:i+convig.step]
            X =  mfcc(sample, rate,numcep=config.nfeat, 
                      nfilt=config.nfilt, nfft=config.nfft)
            X = (X -config.min)/(config.max - config.min)
            
            if config.mode == 'conv':
                X = X.reshape(1, X.shape[0], X.shape[1], 1)
            elif config.mode == 'time':
                X = np.expand_dims(X, axis=0)
            y_hat = model.predict(X)
            y_prob.append(y_hat)
            y_pred.append(np.append(y_hat))
            y_true.append(c)
            
        fn_prob[fn] = np.mean(y_prob, axis=0).flatten()
    
    return y_true, y_pred, fn_prob
        
df = pd.read_csv('instruments.csv')
classes = list(np.unique(df.label))
fn2class = dict(zip(df.fname, df.label))
p_path = os.path.join('pickles','conv.p')

with open(p_path, 'rb') as handle:
    config = pickle.load(handle)
    
model = load_model(config.model_path)

y_true, y_pred, fn_prob = build_prediction('clean')
acc_score = accuracy_score(y_true=y_true, y_pred=y_pred)

y_probs = []

for i, row in df.interrows():
    y_prob = fn_prob[row, fname]
    y_prob.append(y_prob)
    for c, p in zip(classes, y_prob):
        df.at[i, c] = p

y_pred = [classes[np.argmax(y)] for y in y_probs]
df['y_pred'] = y_pred 

df.to_csv('predictions.csv', index=False)

    


    
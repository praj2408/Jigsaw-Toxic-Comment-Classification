# preprocessing and splitting

import tensorflow as tf
import numpy as np
import os
import argparse
import pandas as pd
import gradio as gr

from get_data import read_params
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy


def preprocess_and_train(config_path):
    
    config = read_params(config_path)

    df = pd.read_csv(config['train.csv'])

    df.drop('id', axis=1, inplace = True)

    X = df['comment_text']
    y = df[df.columns[1:]].values

    MAX_FEATURES = config['MAX_FEATURES'] # number of words in the vocab

    #Tokentization
    vectorizer = TextVectorization(max_tokens=MAX_FEATURES,
                                output_sequence_length=1800,
                                output_mode='int')

    vectorizer.adapt(X.values)
    vectorized_text = vectorizer(X.values)

    #MCSHBAP - map, chache, shuffle, batch, prefetch  from_tensor_slices, list_file
    dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y))
    dataset = dataset.cache()
    dataset = dataset.shuffle(160000)
    dataset = dataset.batch(16)
    dataset = dataset.prefetch(8) # helps bottlenecks


    # Splitting the dataset
    train = dataset.take(int(len(dataset)*.7))
    val = dataset.skip(int(len(dataset)*.7)).take(int(len(dataset)*.2))
    test = dataset.skip(int(len(dataset)*.9)).take(int(len(dataset)*.1))


    # Create Sequential Model
    model = Sequential()
    # Create the embedding layer 
    model.add(Embedding(MAX_FEATURES +1, 32)) #32 values long
    # Bidirectional LSTM Layer
    model.add(Bidirectional(LSTM(32, activation='tanh')))
    # Feature extractor Fully connected layers
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    # Final layer 
    model.add(Dense(6, activation='sigmoid'))


    model.compile(loss='BinaryCrossentropy', optimizer='Adam')

    model.summary()

    history = model.fit(train, epochs=10, validation_data=val)
    
    
    model_path = config['model_path']
    
    model.save(model_path)




    # Predictions
    # batch = test.as_numpy_iterator().next()
    # input_text = vectorizer("you freaking suck")
    # model.predict(np.array([input_text]))
    
    
    pre = Precision()
    re = Recall()
    acc = CategoricalAccuracy()
        
    for batch in test.as_numpy_iterator(): 
        # Unpack the batch 
        X_true, y_true = batch
        # Make a prediction 
        yhat = model.predict(X_true)
        
        # Flatten the predictions
        y_true = y_true.flatten()
        yhat = yhat.flatten()
        
        pre.update_state(y_true, yhat)
        re.update_state(y_true, yhat)
        acc.update_state(y_true, yhat)
    
    
    print(f'Precision: {pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')
    
    
    def score_comment(comment):

        # MAX_FEATURES = 200000
        model = tf.keras.models.load_model('saved_models/rnn_base/toxicity.h5')

        # vectorizer = TextVectorization(max_tokens=MAX_FEATURES,
        #                         output_sequence_length=1800,
        #                         output_mode='int')
        
        vectorized_comment = vectorizer([comment])
        results = model.predict(vectorized_comment)
        
        text = ''
        df = pd.read_csv('D:/Projects/Jigsaw-Toxic-Comment-classification/data/raw/train.csv')
        for idx, col in enumerate(df.columns[2:]):
            text += '{}: {}\n'.format(col, results[0][idx]>0.5)
        
        return text
    
    
    
    interface = gr.Interface(fn=score_comment, 
                         inputs=gr.inputs.Textbox(lines=2, placeholder='Comment to score'),
                        outputs='text')

    interface.launch(share=False)
    
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', default='params.yaml')
    parsed_args = args.parse_args()
    preprocess_and_train(config_path=parsed_args.config)
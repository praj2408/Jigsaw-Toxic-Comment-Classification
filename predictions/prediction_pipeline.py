import tensorflow as tf
import gradio as gr
import argparse
import pandas as pd
from src.get_data import read_params
from tensorflow.keras.layers import TextVectorization


def score_comment(comment):

    MAX_FEATURES = 200000
    model = tf.keras.models.load_model('saved_models/rnn_base/toxicity.h5')

    vectorizer = TextVectorization(max_tokens=MAX_FEATURES,
                               output_sequence_length=1800,
                               output_mode='int')
     
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



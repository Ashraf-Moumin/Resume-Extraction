#import tensorflow as tf
from sentence_transformers import SentenceTransformer
import numpy as np

#model = tf.keras.models.load_model('embeddings/model.safetensors')

words_positive = ["skills", "education", "work experience", "experience", "internship", 
                  "training", "languages", "programming language", "university",
                  "position", "role", "industry", "engineer", "consultant", "location", 
                  "analyst", "computer", "test", "system"]

model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')

embedded_positive = model.encode(words_positive)

def evaluate(anchor_text):
    
    if len(anchor_text.split(r'\n'))>4:
        anchor_text = anchor_text.split(r'\n')
    else:
        anchor_text = anchor_text.split(";")

    embedded_anchor = model.encode(anchor_text)

    similarities = np.array(model.similarity(embedded_positive, embedded_anchor))
    
    summed = np.sum(similarities>0.5, axis= 0)

    answer = np.array(anchor_text)

    return answer[summed!=0]

import iris
from transformers import AutoTokenizer,AutoModelForSequenceClassification,TrainingArguments,Trainer,AutoModel,TFAutoModelForSequenceClassification,DefaultDataCollator

import json
import random

# Get rid of datasets and build own metric
from datasets import load_metric,load_dataset,Dataset

import numpy as np

import tensorflow as tf

### MODEL TOKENIZER
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)


### DATASET :
sql_select = """
    SELECT 
        ReviewLabel,ReviewText
    FROM iris.Review
    WHERE ID < 1000
    """
res = iris.sql.exec(sql_select)
 
### TRAIN
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=tf.metrics.SparseCategoricalAccuracy(),
)

try:
    model.fit(x=list(map(lambda x:x[1],res)),y=list(map(lambda x:x[0],res)),epochs=30,verbose=1,shuffle=True)
except Exception as e:
    print(str(e))
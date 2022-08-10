import iris
from transformers import AutoTokenizer,AutoModelForSequenceClassification,TrainingArguments,Trainer,AutoModel,TFAutoModelForSequenceClassification,DefaultDataCollator


import pyarrow as pa
import pyarrow.dataset as ds

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

#dataset = list(map(lambda x:{**{"label":int(x[0]) - 1,"text":x[1],} , **dict(tokenizer(x[1],padding="max_length",truncation=True))},res))


#size = len(dataset)
# split_limit = int(size*0.8) + 1
# train_data = dataset[:split_limit]
# test_data = dataset[split_limit:]

# train_data = Dataset(pa.Table.from_pylist(dataset[:split_limit]))
# test_data = Dataset(pa.Table.from_pylist(dataset[split_limit:]))

# tf_train_dataset = train_data.to_tf_dataset(
#     columns=["attention_mask", "input_ids", "token_type_ids"],
#     label_cols=["labels"],
#     shuffle=True,
#     collate_fn=data_collator,
#     batch_size=8,
# )

# tf_validation_dataset = test_data.to_tf_dataset(
#     columns=["attention_mask", "input_ids", "token_type_ids"],
#     label_cols=["labels"],
#     shuffle=False,
#     collate_fn=data_collator,
#     batch_size=8,
# )

#------
# dataset = load_dataset("yelp_review_full")
# def tokenize_function(examples):
#     return tokenizer(examples["text"], padding="max_length", truncation=True)


# tokenized_datasets = dataset.map(tokenize_function, batched=True)
# small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(200))
# small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(200))


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
import iris
from transformers import AutoTokenizer,AutoModelForSequenceClassification,TrainingArguments,Trainer,AutoModel
import json
import random

# Get rid of datasets and build own metric
from datasets import load_metric,load_dataset,Dataset

import numpy as np

import pyarrow as pa
import pyarrow.dataset as ds

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

### DATASET :
# sql_select = """
#     SELECT 
#         ReviewLabel,ReviewText
#     FROM iris.Review
#     WHERE ID < 1000
#     """
# res = iris.sql.exec(sql_select)

# dataset = list(map(lambda x:{**{"label":int(x[0]) - 1,"text":x[1],} , **dict(tokenizer(x[1],padding="max_length",truncation=True))},res))
# size = len(dataset)
# split_limit = int(size*0.8) + 1

# train_data = Dataset(pa.Table.from_pylist(dataset[:split_limit]))
# test_data = Dataset(pa.Table.from_pylist(dataset[split_limit:]))

#------
dataset = load_dataset("yelp_review_full")
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(200))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(200))


### MODEL AND METRCIS
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased",num_labels=5)
training_args = TrainingArguments(
    output_dir="/irisdev/app/src/model/outputdir/test_trainer",
)

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    #train_dataset=train_data,
    eval_dataset=small_eval_dataset,
    #eval_dataset=test_data,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

try:
    trainer.train()
except Exception as e:
    print(str(e))
import iris
from transformers import AutoTokenizer,AutoModelForSequenceClassification,TrainingArguments,Trainer,AutoModel
import json
import random

from datasets import load_dataset,Dataset
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

import numpy as np

import pyarrow as pa
import pyarrow.dataset as ds

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

### DATASET :
sql_select = """
    SELECT 
        ReviewLabel,ReviewText
    FROM iris.Review
    WHERE ID < 1000
    """
res = iris.sql.exec(sql_select)

dataset = list(map(lambda x:{**{"label":int(x[0]) - 1,"text":x[1],} , **dict(tokenizer(x[1],padding="max_length",truncation=True))},res))
size = len(dataset)
split_limit = int(size*0.8) + 1

train_data = Dataset(pa.Table.from_pylist(dataset[:split_limit]))
test_data = Dataset(pa.Table.from_pylist(dataset[split_limit:]))

#------
# dataset = load_dataset("yelp_review_full")
# def tokenize_function(examples):
#     return tokenizer(examples["text"], padding="max_length", truncation=True)


# tokenized_datasets = dataset.map(tokenize_function, batched=True)
# small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(200))
# small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(200))


### MODEL AND METRCIS
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased",num_labels=5)
# training_args = TrainingArguments(
#     output_dir="/irisdev/app/src/model/outputdir/test_trainer",
#     num_train_epochs=1,              
#     per_device_train_batch_size=1,  
#     per_device_eval_batch_size=1,   
#     warmup_steps=0,                
#     weight_decay=0.01,               
#     do_eval=True,
#     logging_steps=20,
#     evaluation_strategy='steps',
#     save_strategy='steps',
#     learning_rate=0.001,
#     load_best_model_at_end=True,
#     metric_for_best_model='eval_loss',
#     greater_is_better=False,
#     # prediction_loss_only=True
# )

training_args = TrainingArguments(
    output_dir="/irisdev/app/src/model/outputdir/test_trainer",
    evaluation_strategy="steps",
    learning_rate=0.01,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    seed=0,
    load_best_model_at_end=True,
)


def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "precision": precision, "f1": f1}

trainer = Trainer(
    model=model,
    args=training_args,
    #train_dataset=small_train_dataset,
    train_dataset=train_data,
    #eval_dataset=small_eval_dataset,
    eval_dataset=test_data,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

trainer.train()
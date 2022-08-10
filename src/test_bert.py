from transformers import AutoModelForSequenceClassification,Trainer,AutoTokenizer
import numpy as np

from datasets import Dataset
import pyarrow as pa
import pyarrow.dataset as ds

model = AutoModelForSequenceClassification.from_pretrained("src/model/outputdir/test_trainer/checkpoint-800",num_labels=5)
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
test_trainer = Trainer(model)

def tokenize_all(l):
    res = []
    for line in l:
        res.append(tokenizer(line,padding=True,truncation=True))
    return Dataset(pa.Table.from_pylist((res)))

txt = "Really great place, loved it"
raw_pred, _, _ = test_trainer.predict([tokenizer(txt)])
y_pred = np.argmax(raw_pred, axis=1)
print(txt," : ",y_pred)

txt = "Horrible place, never go there"
raw_pred, _, _ = test_trainer.predict([tokenizer(txt)])
y_pred = np.argmax(raw_pred, axis=1)
print(txt," : ",y_pred)

txt = "It was alright"
raw_pred, _, _ = test_trainer.predict([tokenizer(txt)])
y_pred = np.argmax(raw_pred, axis=1)
print(txt," : ",y_pred)
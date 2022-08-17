from transformers import pipeline

pipeline = pipeline(model="src/model/bert-base-cased",task="text-classification")

res = pipeline(["This was an experience"])

print(res)
- [1. iris-fine-tuned-ml](#1-iris-fine-tuned-ml)
- [2. Installation](#2-installation)
  - [2.1. Starting the Production](#21-starting-the-production)
  - [2.2. Access the Production](#22-access-the-production)
  - [2.3. Closing the Production](#23-closing-the-production)
- [3. How it works](#3-how-it-works)
- [4. Tunning the model](#4-tunning-the-model)
  - [4.1. Download the model](#41-download-the-model)
  - [4.2. Settings](#42-settings)
  - [4.3. Train the model](#43-train-the-model)
- [5. Use the model](#5-use-the-model)
  - [5.1. Settings](#51-settings)
  - [5.2. Test the model](#52-test-the-model)
- [6. TroubleShooting](#6-troubleshooting)
- [7. Conclusion](#7-conclusion)


# 1. iris-fine-tuned-ml

Fine-Tuning of Machine Learning models in IRIS using Python based on a IRIS DataBase;<br>

The objective of this GitHub is to simulate a simple use case of Machine Learning in IRIS :<br>We have an IRIS Operation that, on command, can fetch data from the IRIS DataBase to train an existing model in local, then if the new model is better, the user can override the old one with the new one.<br>That way, every x days, if the DataBase has been extended by the users for example, you can train the model on the new data or on all the data and choose to keep or let go this new model.<br>We are aware that training the model multiple times on the same data is detrimental but this is just an example of usage.<br>Moreover the user can of course use the current model to predict some results.

# 2. Installation
## 2.1. Starting the Production

While in the iris-fine-tuned-ml folder, open a terminal and enter :
```
docker-compose up
```
The very first time, it may take a few minutes to build the image correctly and install all the needed modules for Python.

## 2.2. Access the Production

Following this link, access the production : Access the Production

## 2.3. Closing the Production

docker-compose down

# 3. How it works

For now, some models may not work with this implementation since everything is automatically done, which means, no matter what model you input, we will try to make it work through transformers pipeline and trainer library.

Pipeline and trainer are powerful tools by the HuggingFace team that will scan the folder in which we downloaded the model, then understand what library it should use between PyTorch, Keras, Tensorflow or JAX to then load that model using AutoModel.<br>
From here, by inputting the task, the pipeline knows what to do with the model, tokenizer or even feature-extractor in this folder, and manage your input automatically, tokenize it, process it, pass it into the model, then give back the output in a decoded form usable directly by us.<br>
From here, trainer will use your parameters to train the model it loaded using pipeline on the data.

# 4. Tunning the model

## 4.1. Download the model

In order to use this GitHub you need to have a model from HuggingFace compatible with pipeline to use and train, and have a dataset you want to train your model on.<br>

In order to help, we let you the possibility to use the Python script in `src/utils/download_bert.py`. It will download for you the `"https://huggingface.co/bert-base-cased"` model and put inside the `src/model/bert-base-cased` folder.<br>
Moreover we also give you a DataSet to train the bert model on, this dataset wa@s already loaded inside the IRIS DataBase and nothing else needs to be done.

To use the script, if you are inside the container, you can execute it without worry, if you are in local, you may need to `pip3 install requests` and `pip3 install beautifulsoup4`


## 4.2. Settings

If you want to use the bert-base-cased model, and you did downloaded it using the script, nothing needs to be added to the settings and you can advance to the [train the model part](#43-train-the-model).

If you want to use your own model, click on the `Python.TuningOperation`, and select in the right tab `Settings`, then `Python` then in the `%settings` part, enter the path to the model, the name of the folder and the number of label you want it trained on.

Example  :
```
path=/irisdev/app/src/model/
model_name=bert-base-cased
num_labels=5
```

## 4.3. Train the model

To train the model you must go the `Production` following this link :
```
http://localhost:52795/csp/irisapp/EnsPortal.ProductionConfig.zen?PRODUCTION=iris.Production
```

And connect using :<br>
```SuperUser``` as username and ```SYS``` as password.

<br><br>

To call the training, click on the `Python.TuningOperation`, and select in the right tab `Actions`, you can `Test` the demo.

In this test window, select :

Type of request : Grongier.PEX.Message

For the classname you must enter :
```
msg.TrainRequest
```

And for the json, you must enter every arguments needed by the trainer to train.
Here is an example that train on the first 20 rows ( This isn't a proper training but it is fast ):

```
{
    "columns":"ReviewLabel,ReviewText",
    "table":"iris.Review",
    "limit":20,
    "p_of_train":0.8,

    "output_dir":"/irisdev/app/src/model/checkpoints",
    "evaluation_strategy":"steps",
    "learning_rate":0.01,
    "num_train_epochs":1
}
```

As you can see, you must enter
- `table` to use.
- `columns` to use ( first is the `label`, second is the `input` to be tokenized )
- `limit` of rows to take in ( if you don't precise a number of rows, all the data will be used )
- `p_of_train` the percentage of training data to take from the dataset and `1 - p_of_train` the percentage of testing data to take from the dataset.
  
After that, the other parameters are up to you and can be anything according to `https://huggingface.co/docs/transformers/main_classes/trainer` parameters.

**NOTE** that the batch size for training and testing is automatically calculated if not input in the request. ( It's the biggest divider of the number of rows that's less than the square root of the number of row and less than 32 )

Click Invoke Testing Service and close the testing widow without waiting.<br>
Now access the `Python.TuningOperation`, and select in the right tab `log` ; Here you can see the advancement of the training and evaluations.<br>
Once it is over, you will see a log saying that the new model was saved in a temporary folder.<br>
Now access the `Python.TuningOperation`, and select in the right tab `message` and select the last one by clicking on it's header. Here you can see the advancement of the training and evaluations and at the end you can have access to the Metrics of the old and the new model for you to compare.

**If you want to keep the old model**, nothing must be done, the old one will stay on the non-temporary folder and is still loaded for further training.

**If you want to keep the new model**, you must click on the `Python.TuningOperation`, and select in the right tab `Actions` and test.
In this test window, select :

Type of request : Grongier.PEX.Message

For the classname you must enter :
```
msg.OverrideRequest
```
And for the json, empty brackets:

```
{}
```
Click Invoke Testing Service and see the response message. The new model was moved from the temporary folder to the non-temporary folder.


# 5. Use the model
Training a model is interesting but you can also try it out.

## 5.1. Settings
If you want to use the bert-base-cased model, and you did downloaded it using the script, nothing needs to be added to the settings and you can advance to the [test the model part](#test-the-model).

If you want to use your own model, click on the `Python.TuningOperation`, and select in the right tab `Settings`, then `Python` then in the `%settings` part, enter the parameter to add to the pipeline.
## 5.2. Test the model
To test the model you must go the `Production` following this link :
```
http://localhost:52795/csp/irisapp/EnsPortal.ProductionConfig.zen?PRODUCTION=iris.Production
```

And connect using :<br>
```SuperUser``` as username and ```SYS``` as password.

<br><br>

To call the testing, click on the `Python.MLOperation`, and select in the right tab `Actions`, you can `Test` the demo.

In this test window, select :

Type of request : Grongier.PEX.Message

For the classname you must enter :
```
msg.MLRequest
```

And for the json, you must enter every arguments needed by the model to work

```
{
    "inputs":"This was a really bad experience"
}
```
Press `Call test services` and then watch the result.

# 6. TroubleShooting

If you have issues, reading is the first advice we can give you, most errors are easily understood just by reading the logs as almost all errors will be captured by a try / catch and logged.

If you need to install a new module, or Python dependence, open a terminal inside the container and enter for example : "pip install new-module"
To open a terminal there are many ways,

    If you use the InterSystems plugins, you can click in the below bar in VSCode, the one looking like docker:iris:52795[IRISAPP] and select Open Shell in Docker.
    In any local terminal enter : docker-compose exec -it iris bash
    From Docker-Desktop find the IRIS container and click on Open in terminal

Some models may require some changes for the pipeline or the settings for example, it is your task to add in the settings and in the request the right information.
# 7. Conclusion

From here you should be able to use any model that you need or own on IRIS.
NOTE that you can create a Python.MLOperation ( Hugging face operation ) for each of your model and have them on at the same time.
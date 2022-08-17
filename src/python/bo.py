from grongier.pex import BusinessOperation
import iris
from msg import TrainRequest,TrainResponse,OverrideRequest,OverrideResponse,MLRequest,MLResponse
from handler import IrisHandler

from transformers import AutoTokenizer,AutoModelForSequenceClassification,TrainingArguments,Trainer,AutoModel,pipeline

from datasets import Dataset
from sklearn.metrics import accuracy_score,precision_score,f1_score

import numpy as np
import pyarrow as pa
from math import sqrt

from transformers.utils import logging

from os.path import exists
from os import mkdir,rename
import shutil


class TuningOperation(BusinessOperation):
    def on_init(self):
        logging.add_handler(IrisHandler(self))

        if not hasattr(self,"path"):
            self.path = "/irisdev/app/src/model/"
        if not hasattr(self,"model_name"):
            self.model_name = "bert-base-cased"
        if not hasattr(self,"num_labels"):
            self.num_labels = 5
        else:
            self.num_labels = int(self.num_labels)

        try:
            config_attr = set(dir(self)).difference(set(dir(BusinessOperation))).difference(set(['model_name','path','on_train_request','compute_metrics','logger','on_override_request']))
            config_dict = dict()
            for attr in config_attr:
                config_dict[attr] = getattr(self,attr)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.path + self.model_name,**config_dict)
            self.tokenizer = AutoTokenizer.from_pretrained(self.path + self.model_name,model_max_length=512)
            self.log_info("Model and config loaded")
        except Exception as e:
            self.log_warning(str(e))
            self.log_info("Error while loading the model and the config")
        return


    def on_message(self,request):
        return

    def on_train_request(self,request:TrainRequest):
        args = dict()
        for key,value in request.__dict__.items():
            if key[0] != "_" and key not in ["columns","table","limit","p_of_train"]:
                args[key] = value

        resp = TrainResponse()
        try: 
            if request.limit:
                sql_select = f"""
                    SELECT TOP {request.limit} {request.columns}
                    FROM {request.table}
                    """
            else:
                sql_select = f"""
                SELECT {request.columns}
                FROM {request.table}
                """
            res = iris.sql.exec(sql_select)
            dataset = list(map(lambda x:{**{"label":int(x[0]) - 1,"text":x[1],} , **dict(self.tokenizer(x[1],padding="max_length",truncation=True))},res))
            size = len(dataset)
            split_limit = int(size*request.p_of_train)

            train_data = Dataset(pa.Table.from_pylist(dataset[:split_limit]))
            test_data = Dataset(pa.Table.from_pylist(dataset[split_limit:]))
            
            if not hasattr(request,"per_device_eval_batch_size"):
                size_test = len(test_data)
                test_batch_size = 1
                for i in range(2,min(33,int(sqrt(size_test))+1)):
                    if size_test % i == 0:
                        test_batch_size = i
                args["per_device_eval_batch_size"] = test_batch_size

            if not hasattr(request,"per_device_train_batch_size"):
                size_train = len(train_data)
                train_batch_size = 1
                for i in range(2,min(33,int(sqrt(size_train))+1)):
                    if size_train % i == 0:
                        train_batch_size = i
                args["per_device_train_batch_size"] = train_batch_size

    
            training_args = TrainingArguments(**args)

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_data,
                eval_dataset=test_data,
                compute_metrics=self.compute_metrics,
                tokenizer=self.tokenizer
            )

            self.log_info("Evaluating old model")
            old_evaluate = trainer.evaluate()

            self.log_info("Training old model")
            trainer.train()

            self.log_info("Evaluating new model")
            new_evaluate = trainer.evaluate()

            resp.old_model = old_evaluate
            resp.new_model = new_evaluate
            resp.info = "If you want to override the old model, send an OverrideRequest"

            all_path = self.path + "temp-" + self.model_name
            self.log_info(f"Saving new model in {all_path}")
            trainer.save_model(all_path)


        except Exception as e:
            self.log_info(str(e))

        return resp

    def on_override_request(self,request:OverrideRequest):
        if not exists(self.path + "temp-" +self.model_name):
            return OverrideResponse("No temporary model detected, run a training first.")
        else:
            shutil.rmtree(self.path + self.model_name)
            rename(self.path + "temp-" +self.model_name,self.path + self.model_name)
            self.on_init()
            return OverrideResponse("Old model was succesfully overrided")

 

    def compute_metrics(self,p):
        pred, labels = p
        pred = np.argmax(pred, axis=1)

        accuracy = accuracy_score(y_true=labels, y_pred=pred)
        precision = precision_score(y_true=labels, y_pred=pred,average="weighted")
        f1 = f1_score(y_true=labels, y_pred=pred,average="weighted")

        return {"accuracy": accuracy, "precision": precision, "f1": f1}


class MLOperation(BusinessOperation):
    def on_init(self):
        if not hasattr(self,"path"):
            self.path = "/irisdev/app/src/model/"
        if not hasattr(self,"model_name"):
            self.model_name = "bert-base-cased"
        if not hasattr(self,"task"):
            self.task = "text-classification"

        # Get all the attributes of self to put it into the model
        config_attr = set(dir(self)).difference(set(dir(BusinessOperation))).difference(set(['model_name','path','on_ml_request']))
        config_dict = dict()
        for attr in config_attr:
            config_dict[attr] = getattr(self,attr)
        # Loading the model and the config from the folder.
        try:
            self.pipeline = pipeline(model=self.path + self.model_name, **config_dict)
            self.log_info("Model and config loaded")
        except Exception as e:
            self.log_info("Error in loading the model")
            self.log_info(str(e))
        
    def on_ml_request(self,request:MLRequest):
        resp = MLResponse()
        resp.response = self.pipeline([request.inputs])
        return resp

    def on_message(self,request):
        return request
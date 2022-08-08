from grongier.pex import BusinessOperation
import iris

from msg import TrainRequest,MLResponse

from os.path import exists
from os import mkdir

from math import ceil
import random

from io import BytesIO
import base64

from PIL import Image,ImageDraw

import json
import requests

from bs4 import BeautifulSoup as BS

from transformers import pipeline

class TuningOperation(BusinessOperation):
    def on_init(self):
        if not hasattr(self,"path"):
            self.path = "/irisdev/app/src/model/"

        if not hasattr(self,"name"):
            self.name = "gpt2"

        if not hasattr(self,'task'):
            self.task = "text-generation"

        try:
            config_attr = set(dir(self)).difference(set(dir(BusinessOperation))).difference(set(['name','model_url','path','download','on_ml_request']))
            config_dict = dict()
            for attr in config_attr:
                config_dict[attr] = getattr(self,attr)
            self.generator = pipeline(model=self.path + self.name, tokenizer=self.path + self.name, **config_dict)
            self.log_info("Model and config loaded")
        except Exception as e:
            self.log_warning(str(e))
            self.log_info("Error while loading the model and the config")
        return


    def on_message(self,request):
        return

    def on_ml_request(self,request:TrainRequest):
        args = dict()
        for key,value in request.__dict__.items():
            if key[0] != "_":
                args[key] = value

        resp = MLResponse()
        try: 

            if self.task == "object-detection" or self.task == "image-segmentation" or self.task == "image-classification":
                resp = self.object_detection_segmentation(request)
            else:
                ret = self.generator(**args)
                resp.output = ret

        except Exception as e:
            self.log_info(str(e))

        return resp


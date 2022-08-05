from grongier.pex import BusinessOperation
import iris

from msg import MLRequest,MLResponse

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

        if hasattr(self,"model_url"):
            try:
                soup = BS(requests.get(self.model_url + "/tree/main").text)
                elem = soup.findAll('a',{"download":True,"href":True})
                for el in elem:
                    href = el['href']
                    tmp_name = href.split("/")[-1]
                    self.download(tmp_name,"https://huggingface.co" + href)
                self.log_info("All downloads are completed or cached ; loading the model and the config from folder " + self.path + self.name)
            except Exception as e:
                self.log_info(str(e))
                self.log_info("Impossible to request from HuggingFace ; loading the model and the config from existing folder " + self.path + self.name)
        else:
            self.log_info("No given model_url ; trying to load the model and the config from the folder " + self.path + self.name)


        try:
            config_attr = set(dir(self)).difference(set(dir(BusinessOperation))).difference(set(['name','model_url','path','download','on_ml_request']))
            config_dict = dict()
            for attr in config_attr:
                config_dict[attr] = getattr(self,attr)
            self.generator = pipeline(model=self.path + self.name, tokenizer=self.path + self.name, **config_dict)
            self.log_info("Model and config loaded")
        except Exception as e:
            self.log_info(str(e))
            self.log_info("Error while loading the model and the config")
        return

    def download(self,name,url):
        try:
            if not exists(self.path + self.name):
                mkdir(self.path + self.name)
            if not exists(self.path + self.name + "/" + name):
                with open(self.path + self.name + "/" + name, "wb") as f:
                    self.log_info("Downloading %s" % name)
                    response = requests.get(url, stream=True)
                    total_length = response.headers.get('content-length')

                    if total_length is None or int(total_length) < 0.2E9: # no content length header
                        f.write(response.content)
                    else:
                        try:
                            nb_chunck = min(20,ceil(ceil(total_length)*1E-8))
                        except Exception as e:
                            self.log_info(str(e))
                            nb_chunck = 20
                        dl = 0
                        total_length = int(total_length)
                        for data in response.iter_content(chunk_size=int(total_length/nb_chunck)):
                            dl += len(data)
                            f.write(data)
                            done = ceil(nb_chunck * dl / total_length)
                            self.log_info(f"[{'#' * done + ' -' * (nb_chunck-done)}] " + f"{round(dl*1E-9,2)}go/{round(total_length*1E-9,2)}go")
                self.log_info("Download complete for " + name) 
            else:
                self.log_info("Existing files found for " + name)
        except Exception as e:
            self.log_info(str(e))

    def on_message(self,request):
        return

    def on_ml_request(self,request:MLRequest):
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


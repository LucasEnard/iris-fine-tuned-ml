from os.path import exists
from os import mkdir
from math import ceil
import requests
from bs4 import BeautifulSoup as BS


def download(tmp_name):
    if not exists(path + name):
        mkdir(path + name)
    if not exists(path + name + "/" + tmp_name):
        with open(path + name + "/" + tmp_name, "wb") as f:
            print("Downloading %s" % tmp_name)
            response = requests.get(url, stream=True)
            total_length = response.headers.get('content-length')

            if total_length is None or int(total_length) < 0.1E9: # no content length header
                f.write(response.content)
            else:
                try:
                    nb_chunck = min(20,ceil(ceil(int(total_length))*1E-8))
                except Exception as e:
                    print(str(e))
                    nb_chunck = 20
                dl = 0
                total_length = int(total_length)
                for data in response.iter_content(chunk_size=int(total_length/nb_chunck)):
                    dl += len(data)
                    f.write(data)
                    done = ceil(nb_chunck * dl / total_length)
                    print(f"[{'#' * done + ' -' * (nb_chunck-done)}] " + f"{round(dl*1E-9,2)}go/{round(total_length*1E-9,2)}go")
        print("Download complete for " + name) 
    else:
        print("Existing files found for " + name)


path = "src/model/"
name = "bert-base-cased"
url = "https://huggingface.co/bert-base-cased"
try:
    soup = BS(requests.get(url + "/tree/main").text,features="html.parser")
    elem = soup.findAll('a',{"download":True,"href":True})
    for el in elem:
        href = el['href']
        tmp_name = href.split("/")[-1]
        download(tmp_name)
    print("All downloads are completed or cached ; loading the model and the config from folder " + path + name)
except Exception as e:
    print(str(e))
    print("Impossible to request from HuggingFace ; loading the model and the config from existing folder " + path + name)


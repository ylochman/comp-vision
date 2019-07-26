import os
import urllib.request
from tqdm import tqdm

def download_single(url, filepath):
    urllib.request.urlretrieve(url, filepath)

def download_all(dirpath):
    for i in tqdm(range(0, 2000)):
        filename = "ukbench{:0>5}.jpg".format(i)
        url = "http://www.ee.columbia.edu/~rj2349/index_files/Homework1/" + filename
        download_single(url, os.path.join(dirpath, filename))

if __name__ == "__main__":
    download_all("./data")
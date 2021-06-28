import os
import numpy as np
from rootpath import ROOT_PATH

currentpath = os.path.dirname(__file__)
stopwordpath = "{}/stopwords".format(currentpath)
stopwordnamefiles = []
stopwords = []
for (dirpath, dirnames, filenames) in os.walk(stopwordpath):
    stopwordnamefiles.extend(filenames)
    break

for filename in stopwordnamefiles:
    stopwordfile = open("{}/{}".format(stopwordpath, filename), 'r')
    stopword = stopwordfile.read().strip().split("\n")
    stopwords.extend(stopword)

stopwords = np.unique(stopwords).tolist()
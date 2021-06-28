import os
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from time import time
from tqdm import tqdm
import datetime
import numpy as np
from dict.loadsDict import stopwords
import re

currentpath = os.path.dirname(__file__)
resourcespath = "{}/resources".format(currentpath)

resourcefiles = []

for (dirpath, dirnames, filenames) in os.walk(resourcespath):
    resourcefiles.extend(filenames)

resources = []
print("Read resources ...\n")
pbar = tqdm(total=len(resourcefiles),colour='green')
for index,resourcefile in enumerate(resourcefiles):
    resources.append(pd.read_excel("{}/{}".format(resourcespath,resourcefile)))
    pbar.update()
pbar.close()
resources = pd.concat(resources)

tweets = resources.loc[:,['tweet']].to_numpy().flatten()
labels = resources.loc[:,['label']].to_numpy().flatten()

stemmer = StemmerFactory().create_stemmer()

print("\nStemming Processs ...\n")
pbar = tqdm(total=len(labels),colour='green')
for index,tweet in enumerate(tweets):
    tweet = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', tweet, flags=re.MULTILINE)
    tweets[index] = stemmer.stem(tweet)
    pbar.update()
pbar.close()

features_train, features_test, labels_train, labels_test = train_test_split(tweets,labels,test_size=0.3,train_size=0.7)
vectorizer = TfidfVectorizer(stop_words=stopwords)
features_train  = vectorizer.fit_transform(features_train)
features_test = vectorizer.transform(features_test)

t0 = time()
model = GaussianNB()
model.fit(features_train.todense(), labels_train)
print(f"\nTraining time: {round(time()-t0, 3)}s")
t0 = time()
score_train = model.score(features_train.todense(), labels_train)
print(f"Prediction time (train): {round(time()-t0, 3)}s")
t0 = time()
score_test = model.score(features_test.todense(), labels_test)
print(f"Prediction time (test): {round(time()-t0, 3)}s")

resource_feature = vectorizer.transform(tweets)
predicts = np.zeros(resource_feature.shape[0])
for index,item in enumerate(predicts):
    predicts[index] = model.predict(resource_feature[index].todense())[0]

resources['predict'] = predicts
resources['tweet_stemmed'] = tweets
tn, fp, fn, tp = confusion_matrix(labels, predicts).ravel()

acc = (tp+tn)/len(labels)
recall = tp/(tp+fn)
precision = tp/(tp+fp)
f_measure =  (2*recall*precision)/(recall+precision)

print("=================================================\n")
print("\t\tCONFUSION MATRIX\n")
print("=================================================\n")
print("TN: {} | FP: {} | FN: {} | TP: {}\n".format(tn,fp,fn,tp))
print("=================================================\n")
print("ACC : {} | RECALL: {} | PRECISION: {} | F-1: {}".format(acc,recall,precision,f_measure))
print("=================================================\n")


# (tn, fp, fn, tp)
conf_matrix = {
    'TN' : [tn],
    'FP' : [fp],
    'FN' : [fn],
    'TP' : [tp]
}
pd.DataFrame(conf_matrix).to_excel("{}/results/{}-confusion-matrix.xlsx".format(currentpath,datetime.date.today()))

resources.to_excel("{}/results/{}.xlsx".format(currentpath,datetime.date.today()),index=False)
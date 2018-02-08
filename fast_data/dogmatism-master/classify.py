import re
import fileinput
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import numpy as np
from textblob import TextBlob
from empath import Empath

lexicon = Empath()

cv = CountVectorizer(min_df=3,stop_words="english",ngram_range=(1,3))
model = LogisticRegression(C=1,class_weight="balanced")

docs, scores, conf, yous, comps, poss, subjs, sen_lens = [], [], [], [], [], [], [], []

extra = False

def all_empath(doc):
    keys = sorted(list(lexicon.cats.keys()))
    cats = lexicon.analyze(doc)
    return [cats[k] for k in keys]

def confidence(doc):
    count = 0.0
    for x in re.finditer("I thought|I think|I don’t know|likely|probably|seem to be|I understood|I understand|I heard|maybe|I wonder|I wondered|personally",doc):
      count += 1.0
    return count #/ len(doc.split(" "))

def you(doc):
    count = 0.0
    for x in re.finditer("You are|you are|you're|your",doc):
        count += 1.0
    return count #/ len(doc.split(" "))

def me(doc):
    count = 0.0
    for x in re.finditer("I am|I'm|my",doc):
        count += 1.0
    return count

def comprimise(doc):
    count = 0.0
    for x in re.finditer("I see|I agree|agreed|Yes|yes|Thanks|thanks",doc):
        count += 1.0
    return count #/ len(doc.split(" "))

def sent_len(doc):
    b = TextBlob(doc)
    return np.average([len(sen.split(" ")) for sen in b.sentences])

def pos(doc):
   b = TextBlob(doc)
   return np.average([sentence.sentiment.polarity for sentence in b.sentences])

def subj(doc):
   b = TextBlob(doc)
   return np.average([sentence.sentiment.subjectivity for sentence in b.sentences])

for line in fileinput.input("old_politics-1000.txt"):
  doc, score = line.split("\t")
  score = int(score)
  if doc[0] == "\"": doc = doc[1:]
  if doc[-1] == "\"": doc = doc[:-1]
  conf.append(confidence(doc))
  yous.append(you(doc))
  comps.append(comprimise(doc))
  poss.append(pos(doc))
  subjs.append(subj(doc))
  sen_lens.append(me(doc))
  scores.append(score)
  docs.append(doc)
  #if score <= 6:
  #  scores.append(0)
  #  docs.append(doc)
  #elif score >= 14:
  #  scores.append(1)
  #  docs.append(doc)


score_d = defaultdict(int)
for s in scores: score_d[s] += 1

print(score_d)

x = cv.fit_transform(docs).toarray()
if extra:
  conf = np.array(conf)
  conf = conf.reshape(len(conf),1)
  yous = np.array(yous).reshape(len(yous),1)
  poss = np.array(poss).reshape(len(poss),1)
  comps = np.array(comps).reshape(len(comps),1)
  subjs = np.array(subjs).reshape(len(subjs),1)
  sen_lens = np.array(sen_lens).reshape(len(sen_lens),1)
  #x = np.append(x,np.array(empaths),1)
  x = np.append(x,conf,1)
  x = np.append(x,yous,1)
  x = np.append(x,comps,1)
  x = np.append(x,poss,1)
  x = np.append(x,subjs,1)
  x = np.append(x,sen_lens,1)


data_z = sorted(list(zip(x,scores)),key=lambda x: x[1])

n = 300

x = [b[0] for b in data_z[:n]] + [b[0] for b in data_z[-n:]]
scores = [0 for b in data_z[:n]] + [1 for b in data_z[-n:]]

x = np.array(x)

print(len(x),len(scores))

print(type(x))

train_x = x[:900]
test_x = x[900:]
train_y = scores[:900]
test_y = scores[900:]

model.fit(x,scores)

from sklearn import cross_validation

print(np.average(cross_validation.cross_val_score(model, x, scores, cv=15, scoring="f1")))
print(np.average(cross_validation.cross_val_score(model, x, scores, cv=15, scoring="precision")))
print(np.average(cross_validation.cross_val_score(model, x, scores, cv=15, scoring="recall")))
print(np.average(cross_validation.cross_val_score(model, x, scores, cv=15, scoring="accuracy")))

vocab = {v:k for k,v in cv.vocabulary_.items()}
if extra:
  vocab[x.shape[1]-6] = "unconfidence_features"
  vocab[x.shape[1]-5] = "you_features"
  vocab[x.shape[1]-4] = "comprimise_features"
  vocab[x.shape[1]-3] = "pos_features"
  vocab[x.shape[1]-2] = "subjs_features"
  vocab[x.shape[1]-1] = "sen_len_features"
  #for i,k in enumerate(sorted(list(lexicon.cats.keys()),reverse=True)):
  #    vocab[x.shape[1]-5-i] = "empath_"+k
coefs = [(vocab[i],c) for i,c in enumerate(model.coef_[0])]

pos_s = sorted([x for x in coefs if x[1] > 0],key=lambda x: x[1],reverse=True)[:200]
neg_s = sorted([x for x in coefs if x[1] < 0],key=lambda x: x[1])[:200]

for x in pos_s: print(x)
for x in neg_s: print(x)


ddd = sorted(list(zip(docs,scores)),key=lambda x: x[1])
print("pos")
for x in ddd[:n]: print(x)
print("neg")
for x in ddd[n:]: print(x)

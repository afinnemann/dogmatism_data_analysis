import fileinput
from collections import defaultdict
import numpy as np

docs = defaultdict(list)
doc_sum = []

worker_scores = defaultdict(list)

for line in fileinput.input():
    cols = line.strip().split("\t")
    doc = cols[27]
    score = int(cols[28])
    id_ = cols[15]
    worker_scores[id_].append(score)
    docs[doc].append(score)

for k,v in worker_scores.items():
    print(k,v)

most_neg = []
neg = []
pos = []
most_pos = []

agree_workers = 0
workers = 0

for d,s in docs.items():
    doc_sum.append((d,sum(s)))
    agree = set(s)
    if len(agree) == 1:
        agree_workers += 3
    elif len(agree) == 2:
        agree_workers += 2
    workers += 3

for d,s in docs.items():
    if all([x >= 4 for x in s]):
        most_neg.append(d)
    elif np.average(s) > 3:
        neg.append(d)
    elif all([x <= 2 for x in s]):
        most_pos.append(d)
    elif np.average(s) <= 3:
        pos.append(d)

print("Most intransigent".upper(),len(most_neg))
#for x in most_neg: print(x+"\n")
print()
print("Intransigent".upper(),len(neg))
#for x in neg: print(x+"\t"+,file=data_f)
print()
print("Less intransigent".upper(),len(pos))
#for x in pos: print(x+"\n")
print()
print("Not intransigent".upper(),len(most_pos))
#for x in most_pos: print(x+"\n")

print(float(agree_workers)/workers)

data_f = open("reddit-997","w")

for e in sorted(doc_sum,key=lambda x: x[1],reverse=True):
    print(e[0]+"\t"+str(e[1]), file=data_f)

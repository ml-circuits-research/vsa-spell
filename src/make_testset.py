import random, re, sys, json
import os, codespell_lib
cs = os.path.join(os.path.dirname(codespell_lib.__file__), 'data', 'dictionary.txt')
freq = {}
for i,l in enumerate(open('data/en-80k.txt')):
    w,c = l.split(); freq[w]=int(c)
top = set(list(freq)[:50000])
pairs=[]
for l in open(cs):
    if '->' not in l: continue
    a,b = l.strip().split('->',1)
    b=b.rstrip(',')
    if ',' in b: continue
    if not re.fullmatch('[a-z]{3,}',a) or not re.fullmatch('[a-z]{3,}',b): continue
    if a in freq or b not in top: continue
    pairs.append((a,b))
random.seed(42); random.shuffle(pairs)
test=pairs[:3000]
json.dump(test,open('data/test_nonword.json','w'))
json.dump(pairs[3000:4000],open('data/dev_nonword.json','w'))  # dev split for tuning, disjoint from test
print(len(pairs),'eligible; test',len(test)); print(test[:12])

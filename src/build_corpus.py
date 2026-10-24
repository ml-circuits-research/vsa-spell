"""Assemble a training corpus (one sentence per line) from NLTK corpora,
excluding anything that also occurs in big.txt (the test source)."""
import os, re, json, glob, random

C = 'corpus'
out = []

def sents(text):
    text = re.sub(r'\s+', ' ', text)
    return [s for s in re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', text) if 20 < len(s) < 400]

def add(name, texts):
    n0 = len(out)
    for t in texts: out.extend(sents(t))
    print(f'{name:16s} {len(out) - n0:8d} sentences')

def read(p): return open(p, encoding='utf8', errors='ignore').read()

# brown: word/tag tokens
def brown():
    for p in glob.glob(f'{C}/brown/c[a-r][0-9][0-9]'):
        toks = [w.rsplit('/', 1)[0] for w in read(p).split()]
        t = ' '.join(toks)
        t = re.sub(r" ([.,;:!?')])", r'\1', t).replace('`` ', '"').replace(" ''", '"')
        yield t
add('brown', brown())
add('reuters', (read(p) for p in glob.glob(f'{C}/reuters/*/*')))
add('gutenberg', (read(p) for p in glob.glob(f'{C}/gutenberg/*.txt')))
add('movie_reviews', (' '.join(l.strip().capitalize() for l in open(p, encoding='utf8', errors='ignore')) for p in glob.glob(f'{C}/movie_reviews/*/*.txt')))
add('abc', (read(p) for p in glob.glob(f'{C}/abc/*.txt')))
add('webtext', (read(p) for p in glob.glob(f'{C}/webtext/*.txt')))
add('europarl_en', (read(p) for p in glob.glob(f'{C}/europarl_raw/english/*')))
add('state_union', (read(p) for p in glob.glob(f'{C}/state_union/*.txt')))
add('inaugural', (read(p) for p in glob.glob(f'{C}/inaugural/*.txt')))
def tweets():
    for p in glob.glob(f'{C}/twitter_samples/*.json'):
        for l in open(p, encoding='utf8'):
            try: t = json.loads(l)['text']
            except Exception: continue
            t = re.sub(r'(https?://\S+|@\w+|#\w+|RT )', '', t)
            yield t + ' .'
add('twitter', tweets())

# exclude overlap with big.txt
big = set(sents(read('data/big.txt')))
bignorm = {re.sub(r'[^a-z]', '', s.lower()) for s in big}
keep = [s for s in out if re.sub(r'[^a-z]', '', s.lower()) not in bignorm]
print(f'removed {len(out) - len(keep)} sentences overlapping big.txt')
random.seed(3); random.shuffle(keep)
open('data/train_corpus.txt', 'w').write('\n'.join(keep))
print('total sentences', len(keep), 'words', sum(len(re.findall('[A-Za-z]+', s)) for s in keep))

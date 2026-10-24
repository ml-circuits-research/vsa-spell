import json, time
from symspellpy import SymSpell, Verbosity

test = json.load(open('data/test_nonword.json'))
res = {}
for ed in (2, 3):
    t = time.time()
    s = SymSpell(max_dictionary_edit_distance=ed, prefix_length=7)
    s.load_dictionary('data/en-80k.txt', 0, 1)
    build = time.time() - t
    t1 = t5 = 0; fails = []
    t = time.time()
    for bad, good in test:
        sug = [x.term for x in s.lookup(bad, Verbosity.ALL, max_edit_distance=ed)[:5]]
        if sug and sug[0] == good: t1 += 1
        elif len(fails) < 15: fails.append(f"{bad}->{sug[0] if sug else None} (want {good})")
        if good in sug: t5 += 1
    ms = (time.time() - t) * 1000 / len(test)
    res[f'SymSpell ed{ed}'] = dict(top1=t1/len(test), top5=t5/len(test), msPerWord=ms, buildS=build, fails=fails)
    print(f'SymSpell ed{ed}', res[f'SymSpell ed{ed}'])
json.dump(res, open('results/res_symspell.json', 'w'), indent=1)

"""Regenerate the 174 gadget entries and their exact S4 normal forms."""
from fractions import Fraction as F
from collections import defaultdict
from .model import comp,invp,idx,ident
INV=dict(zip('aAbBxX','AaBbXx'))
REL=['xx','xaxA'*3,'xaaxAAXaaXAA','abABX','axAbXB']
def inverse(w): return ''.join(INV[x] for x in w[::-1])
def offset(w): return w.count('a')-w.count('A'), w.count('b')-w.count('B')
def reduced(w):
    out=[]
    for x in w:
        if out and out[-1]==INV[x]:out.pop()
        else:out.append(x)
    return ''.join(out)
def trans(i,j,L=4):
    p=list(range(L));p[i],p[j]=p[j],p[i];return tuple(p)
I=ident(4); S=[trans(k,k+1) for k in range(3)]
EXCEPT={'abA','abAB','abABX'}
SPECIAL={'b':(I,'b'),'B':(I,'B'),'ab':(I,'ab'),
         'abA':(S[0],'b'),'abAB':(S[0],''),'abABX':(I,''),
         'axAb':(S[1],'b'),'axAbX':(I,'b'),'axAbXB':(I,''),'bB':(I,'')}
def normal(w):
    if 'b' in w or 'B' in w:
        if w not in SPECIAL:raise ValueError('No normal form recorded for b-word: '+w)
        return SPECIAL[w]
    h=I;k=0
    for x in w:
        if x=='a':k+=1
        elif x=='A':k-=1
        elif x in 'xX':
            if k not in (0,1,2):raise ValueError('Conjugate outside the S4 window.')
            h=comp(h,S[k])
        else:raise ValueError(x)
    return h,('a'*k if k>=0 else 'A'*(-k))
SYMBOLS=list('aAbBxX')+[r[:k] for r in REL for k in range(2,len(r))]
TRI=[(r[:k],r[k]) for r in REL for k in range(1,len(r))]+[(x,INV[x]) for x in 'abx']
# Encoding: (row,column,literal word,sign,squared denominator).
ENT=[(0,0,'',1,1)]+[(i+1,i+1,w,1,1) for i,w in enumerate(SYMBOLS)]
for j,(x,y) in enumerate(TRI):
    k=34+2*j
    ENT += [(k,k,'',1,2),(k,k+1,y,1,2),
            (k+1,k,x,1,2),(k+1,k+1,x+y,-1,2)]
def label(w):
    h,t=normal(w)
    return offset(t),h
GROUPS=defaultdict(list)
for e in ENT:GROUPS[label(e[2])].append(e)
WORDS=list(dict.fromkeys(e[2] for e in ENT))
def fraction_leakage_coefficient():
    total=F(0)
    for word,bad_word in [('',REL[3]),('x',REL[3][:-1])]:
        es=GROUPS[label(word)]
        mass=sum(F(1,e[4]) for e in es)
        bad=sum(F(1,e[4]) for e in es if e[2]==bad_word)
        total += 2*bad*(mass-bad)/(104*mass)
    return total

def certificate():
    """Regenerate finite witness data; no saved numeric data enters acceptance."""
    from collections import Counter
    classes = defaultdict(list)
    for key, entries in GROUPS.items():
        translation, h = key
        word = min((e[2] for e in entries), key=lambda w: (len(w), w))
        classes[translation].append((h, word))
    exported = []
    for translation in sorted(classes):
        items = sorted(classes[translation], key=lambda item: (len(item[1]), item[1]))
        hs = [item[0] for item in items]
        rows = [Counter(idx(comp(h, invp(g))) for g in hs if g != h) for h in hs]
        weights = ([7 if r[1] == 4 else 6 if r[1] == 3 else 4 for r in rows]
                   if len(hs) == 8 else [1] * len(hs))
        exported.append({
            'translation': list(translation),
            'words': [item[1] for item in items],
            'normal_forms': [list(h) for h in hs],
            'weights': weights,
            'cycle_indices': [[idx(comp(h, invp(g))) for g in hs] for h in hs]
        })
    return {
        'format': 'corner-normal-forms-v1', 'q': 16,
        'norm_bound': [3, 14], 'leakage_coefficient': [122, 2925],
        'bad_clock_count': 2, 'classes': exported,
        'literal_normal_forms': [
            {'word': w, 'permutation': list(normal(w)[0]),
             'translation_word': normal(w)[1], 'exceptional': w in EXCEPT}
            for w in WORDS
        ]
    }

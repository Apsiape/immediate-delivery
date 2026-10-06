"""Exact corner permutations, with right-action composition.

No colorings or exponential bath matrices are enumerated. The coordinate
register has length 4M+4: all later coordinates are fixed by every link.
"""
from fractions import Fraction as F
from collections import defaultdict

def ident(L): return tuple(range(L))
def comp(*ps):
 p=ident(len(ps[0]))
 for q in ps:p=tuple(q[k] for k in p)
 return p

def invp(p):
 q=[0]*len(p)
 for i,j in enumerate(p):q[j]=i
 return tuple(q)
def cycle(r,L):return tuple([r-1]+list(range(r-1))+list(range(r,L)))
def swap(L):return tuple([1,0]+list(range(2,L)))
def idx(p):
 seen=set();c=0
 for i in range(len(p)):
  if i in seen:continue
  c+=1;k=i
  while k not in seen:seen.add(k);k=p[k]
 return len(p)-c
class Model:
 def __init__(self,M):
  if not isinstance(M,int) or M<2: raise ValueError('M must be an integer at least two.')
  self.M=M;self.ell=2*M+2;self.L=2*self.ell;self.I=ident(self.L);self.tau=swap(self.L)
  D=[cycle(2*self.ell-M+1,self.L)]
  for i in range(M-1):D.append(comp(invp(cycle(self.ell-i-(M-1),self.L)),self.tau,D[-1],cycle(self.ell-i,self.L)))
  self.D=D
 def letter(self,i,j,w):
  M=self.M
  if w=='a':return (i+1)%M,j,cycle(self.ell-i-j,self.L)
  if w=='b':return i,(j+1)%M,self.D[i] if j==M-1 else cycle(2*self.ell-j,self.L)
  if w in 'xX':return i,j,self.tau
  if w=='A':
   ni=(i-1)%M;return ni,j,invp(cycle(self.ell-ni-j,self.L))
  if w=='B':
   nj=(j-1)%M;return i,nj,invp(self.D[i] if nj==M-1 else cycle(2*self.ell-nj,self.L))
  raise ValueError(w)
 def word(self,i,j,w):
  p=self.I
  for c in w:
   i,j,q=self.letter(i,j,c);p=comp(p,q)
  return i,j,p
REL=['xx','xaxA'*3,'xaaxAAXaaXAA','abABX','axAbXB']

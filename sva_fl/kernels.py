"""Single-threaded compiled uint64-limb kernels; no change to F_q or sampling."""
import sys
from pathlib import Path

import numpy as np

deps=Path(__file__).resolve().parents[1]/'.deps_sva_native_v4'
if deps.is_dir():sys.path.insert(0,str(deps))
from numba import njit

Q=2**252+27742317777372353535851937790883648493
QL=np.array([(Q>>(64*k))&((1<<64)-1) for k in range(4)],dtype=np.uint64)
ZERO=np.uint64(0);ONE=np.uint64(1);MASK=np.uint64(2**32-1)


@njit(cache=True)
def add_mod(a,b,sign):
    n=len(a);out=np.empty_like(a)
    for i in range(n):
        carry=ZERO
        for k in range(4):
            x=a[i,k];y=b[i,k]
            if sign==1:
                s=x+y;c1=s<x;t=s+carry;c2=t<s
            else:
                s=x-y;c1=x<y;t=s-carry;c2=s<carry
            out[i,k]=t;carry=ONE if c1 or c2 else ZERO
        if sign==1:
            reduce=True
            for k in range(3,-1,-1):
                if out[i,k]<QL[k]:reduce=False;break
                if out[i,k]>QL[k]:break
            if reduce:
                borrow=ZERO
                for k in range(4):
                    x=out[i,k];s=x-QL[k];b1=x<QL[k];t=s-borrow;b2=s<borrow
                    out[i,k]=t;borrow=ONE if b1 or b2 else ZERO
        elif carry:
            carry=ZERO
            for k in range(4):
                x=out[i,k];s=x+QL[k];c1=s<x;t=s+carry;c2=t<s
                out[i,k]=t;carry=ONE if c1 or c2 else ZERO
    return out


@njit(cache=True)
def centered_int64(a):
    out=np.empty(len(a),dtype=np.int64)
    for i in range(len(a)):
        if a[i,1]==ZERO and a[i,2]==ZERO and a[i,3]==ZERO and a[i,0]<=np.uint64(2**63-1):
            out[i]=np.int64(a[i,0]);continue
        borrow=ZERO;lo=ZERO;high=ZERO
        for k in range(4):
            x=QL[k];y=a[i,k];s=x-y;b1=x<y;t=s-borrow;b2=s<borrow
            borrow=ONE if b1 or b2 else ZERO
            if k==0:lo=t
            else:high|=t
        if high!=ZERO or lo>np.uint64(2**63):return out,False
        out[i]=np.int64(ZERO-lo)
    return out,True


@njit(cache=True,inline='always')
def _multiply64(x,y):
    x0=x&MASK;x1=x>>np.uint64(32);y0=y&MASK;y1=y>>np.uint64(32)
    p0=x0*y0;p1=x0*y1;p2=x1*y0;p3=x1*y1
    middle=(p0>>np.uint64(32))+(p1&MASK)+(p2&MASK)
    low=(middle<<np.uint64(32))|(p0&MASK)
    high=p3+(p1>>np.uint64(32))+(p2>>np.uint64(32))+(middle>>np.uint64(32))
    return low,high


@njit(cache=True,inline='always')
def _add_word(acc,index,word):
    while word!=ZERO:
        if index>=len(acc):raise OverflowError('Projection accumulator capacity exceeded')
        previous=acc[index];acc[index]=previous+word
        word=ONE if acc[index]<previous else ZERO
        index+=1


@njit(cache=True)
def projection_signed(a,values):
    positive=np.zeros(6,dtype=np.uint64);negative=np.zeros(6,dtype=np.uint64)
    for i in range(len(a)):
        x=values[i];mag=ZERO-np.uint64(x) if x<0 else np.uint64(x)
        target=negative if x<0 else positive
        for k in range(4):
            low,high=_multiply64(a[i,k],mag)
            _add_word(target,k,low);_add_word(target,k+1,high)
    return positive,negative


@njit(cache=True)
def projection_general(a,b):
    acc=np.zeros(9,dtype=np.uint64)
    for i in range(len(a)):
        for j in range(4):
            for k in range(4):
                low,high=_multiply64(a[i,j],b[i,k])
                _add_word(acc,j+k,low);_add_word(acc,j+k+1,high)
    return acc


@njit(cache=True)
def accepted_limbs(raw):
    out=np.empty_like(raw);count=0
    for i in range(len(raw)):
        high=raw[i,3]&np.uint64((1<<61)-1)
        valid=False
        for k in range(3,-1,-1):
            value=high if k==3 else raw[i,k]
            if value<QL[k]:valid=True;break
            if value>QL[k]:break
        if valid:
            for k in range(3):out[count,k]=raw[i,k]
            out[count,3]=high;count+=1
    return out[:count]


def warmup():
    a=np.zeros((2,4),dtype='<u8');a[1,0]=1
    add_mod(a,a,1);add_mod(a,a,-1);centered_int64(a)
    projection_signed(a,np.array([1,-1],dtype=np.int64));projection_general(a,a)
    accepted_limbs(a)
    accepted_limbs(np.frombuffer(a.tobytes(),dtype='<u8').reshape(-1,4))

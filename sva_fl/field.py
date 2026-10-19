"""Exact F_q vectors in four uint64 limbs; unchanged ChaCha20 field stream.

The native backend uses only unsigned integer arithmetic. Without the isolated
compiler dependency, projections fall back to exact Python integer arithmetic.
"""
import hashlib
import hmac

import numpy as np
from Cryptodome.Cipher import ChaCha20

from sva_fl.field_stream import Q, accepted_raw
try:
    from sva_fl import kernels as native
except ModuleNotFoundError as exc:
    if exc.name not in ('numba','llvmlite'):raise
    native=None

NATIVE_READY=native is not None

QLIMBS=np.array([(Q>>(64*k))&((1<<64)-1) for k in range(4)],dtype='<u8')
U64=np.uint64
BLOCK=65536


def _raw_add(a,b):
    size=len(a) if a.ndim==2 else len(b)
    out=np.empty((size,4),dtype='<u8');carry=np.zeros(size,dtype=np.uint64)
    for k in range(4):
        x=a[:,k] if a.ndim==2 else a[k]
        y=b[:,k] if b.ndim==2 else b[k]
        s=x+y;c1=s<x;t=s+carry;c2=t<s
        out[:,k]=t;carry=(c1|c2).astype(np.uint64)
    return out,carry


def _raw_sub(a,b):
    size=len(a) if a.ndim==2 else len(b)
    out=np.empty((size,4),dtype='<u8');borrow=np.zeros(size,dtype=np.uint64)
    for k in range(4):
        x=a[:,k] if a.ndim==2 else a[k]
        y=b[:,k] if b.ndim==2 else b[k]
        s=x-y;b1=x<y;t=s-borrow;b2=s<borrow
        out[:,k]=t;borrow=(b1|b2).astype(np.uint64)
    return out,borrow


def add_mod(a,b,sign=1):
    if sign not in [1,-1]:raise ValueError('Expected sign +1 or -1')
    if native is not None:return native.add_mod(a,b,sign)
    if sign==1:
        summed,_=_raw_add(a,b)
        reduced,borrow=_raw_sub(summed,QLIMBS)
        return np.where(borrow[:,None].astype(bool),summed,reduced)
    if sign==-1:
        diff,borrow=_raw_sub(a,b)
        corrected,_=_raw_add(diff,QLIMBS)
        return np.where(borrow[:,None].astype(bool),corrected,diff)
    raise ValueError('Expected sign +1 or -1')


class FieldVector:
    def __init__(self,limbs):
        a=np.asarray(limbs)
        if a.ndim!=2 or a.shape[1]!=4 or a.dtype!=np.dtype('<u8'):
            raise ValueError('Expected N x 4 little-endian uint64 limbs')
        self.limbs=np.ascontiguousarray(a)

    def __len__(self):return len(self.limbs)

    def __eq__(self,other):
        if isinstance(other,FieldVector):return np.array_equal(self.limbs,other.limbs)
        if isinstance(other,(list,tuple)):return list(self)==list(other)
        return NotImplemented

    def __iter__(self):
        for row in self.limbs:yield int.from_bytes(row.tobytes(),'little')

    def __getitem__(self,key):
        if isinstance(key,slice):return FieldVector(self.limbs[key].copy())
        return int.from_bytes(self.limbs[key].tobytes(),'little')

    def __setitem__(self,key,value):
        self.limbs[key]=np.frombuffer((int(value)%Q).to_bytes(32,'little'),dtype='<u8')

    def tobytes(self):return self.limbs.tobytes(order='C')

    def centered_int64(self):
        if native is not None:
            result,valid=native.centered_int64(self.limbs)
            if not valid:raise OverflowError('Not a signed64 vector')
            return result
        a=self.limbs
        positive=(a[:,1:]==0).all(axis=1)&(a[:,0]<=U64(2**63-1))
        magnitude,_=_raw_sub(QLIMBS,a)
        negative=(magnitude[:,1:]==0).all(axis=1)&(magnitude[:,0]<=U64(2**63))
        if not np.all(positive|negative):raise OverflowError('Not a signed64 vector')
        unsigned=np.where(positive,a[:,0],U64(0)-magnitude[:,0])
        return unsigned.view('<i8')


def from_values(values):
    if isinstance(values,FieldVector):return values[:]
    a=np.asarray(values)
    if a.ndim!=1:raise ValueError('Expected one-dimensional integer vector')
    if a.dtype.kind in 'iu' and a.dtype.itemsize<=8:
        a=a.astype('<i8' if a.dtype.kind=='i' else '<u8',copy=False)
        words=np.zeros((len(a),4),dtype='<u8')
        if a.dtype.kind=='u':
            words[:,0]=a;return FieldVector(words)
        neg=a<0;u=a.view('<u8')
        mag=np.where(neg,U64(0)-u,u)
        words[:,0]=mag
        minus,_=_raw_sub(QLIMBS,words)
        words=np.where(neg[:,None],minus,words)
        return FieldVector(words)
    raw=b''.join((int(x)%Q).to_bytes(32,'little') for x in values)
    return FieldVector(np.frombuffer(raw,dtype='<u8').reshape(-1,4).copy())


def zeros(n):return FieldVector(np.zeros((n,4),dtype='<u8'))


def iter_limb_blocks(key,context,purpose,count,block=BLOCK):
    from sva_fl.field_wire import enc
    if count<0 or block<1:raise ValueError('Invalid field stream length')
    derived=hmac.new(key,enc(['SVA/ChaCha20-field/v3',context,purpose]),hashlib.sha256).digest()
    cipher=ChaCha20.new(key=derived,nonce=b'\0'*12)
    remaining=count;pending=[];pending_n=0
    while remaining:
        need=min(block,remaining)
        while pending_n<need:
            candidates=min(block,max(16,2*(need-pending_n)))
            raw=cipher.encrypt(bytes(32*candidates))
            if native is not None:
                a=native.accepted_limbs(np.frombuffer(raw,dtype='<u8').reshape(-1,4))
            else:
                a=np.frombuffer(accepted_raw(raw),dtype='<u8').reshape(-1,4)
            if len(a):
                pending.append(a);pending_n+=len(a)
        all_values=pending[0] if len(pending)==1 else np.concatenate(pending)
        yield all_values[:need]
        tail=all_values[need:]
        pending=[tail] if len(tail) else [];pending_n=len(tail);remaining-=need


def fields(key,context,purpose,count):
    blocks=list(iter_limb_blocks(key,context,purpose,count))
    return FieldVector(np.concatenate(blocks) if blocks else np.empty((0,4),dtype='<u8'))


class ReplayableMask:
    def __init__(self,key,context,purpose,count):self.args=(key,context,purpose,count)
    def __len__(self):return self.args[-1]
    def __iter__(self):return iter(self.materialize())
    def blocks(self):return iter_limb_blocks(*self.args)
    def materialize(self):return fields(*self.args)


def apply_mask(vector,mask,sign):
    offset=0
    for block in mask.blocks():
        length=len(block)
        vector.limbs[offset:offset+length]=add_mod(vector.limbs[offset:offset+length],block,sign)
        offset+=length
    if offset!=len(vector):raise ValueError('Mask length mismatch')
    return vector


def add_inplace(vector,other,sign=1):
    if len(vector)!=len(other):raise ValueError('Field vector length mismatch')
    for offset in range(0,len(vector),BLOCK):
        vector.limbs[offset:offset+BLOCK]=add_mod(vector.limbs[offset:offset+BLOCK],other.limbs[offset:offset+BLOCK],sign)
    return vector


def aggregate(masked,survivors,delta):
    result=zeros(len(delta))
    for uid in survivors:add_inplace(result,masked[uid])
    return add_inplace(result,delta,-1)


def forged_first(vector):
    result=vector[:];result[0]=result[0]+1;return result


def dot(left,right):
    if len(left)!=len(right):raise ValueError('Projection length mismatch')
    try:signed=right.centered_int64()
    except OverflowError:signed=None
    if native is not None:
        if signed is not None:
            positive,negative=native.projection_signed(left.limbs,signed)
            return (int.from_bytes(positive.tobytes(),'little')-
                    int.from_bytes(negative.tobytes(),'little'))%Q
        limbs=native.projection_general(left.limbs,right.limbs)
        return int.from_bytes(limbs.tobytes(),'little')%Q
    return sum(a*b for a,b in zip(left,right))%Q


def centered(vector):
    if isinstance(vector,FieldVector):
        try:return vector.centered_int64()
        except OverflowError:pass
    return [x if x<=Q//2 else x-Q for x in vector]

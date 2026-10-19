"""Four-byte ChaCha20 masks in Z/(2**32); bounded signed updates for F_q checks.

This is a new numeric backend, not byte-equivalent to the archived F_q backend.
Only vector masks use the 32-bit ring. Challenges and commitment blinders retain Q.
"""
import hashlib
import hmac
import numpy as np
from Cryptodome.Cipher import ChaCha20
from sva_fl import field as field

Q=field.Q


class RingVector:
    def __init__(self, values):
        a=np.asarray(values)
        if a.ndim!=1 or a.dtype!=np.dtype('<u4'):
            raise ValueError('Expected a one-dimensional little-endian uint32 vector')
        self.values=np.ascontiguousarray(a)
    def __len__(self):return len(self.values)
    def __getitem__(self, index):
        if isinstance(index,slice):return RingVector(self.values[index].copy())
        return int(self.values[index])
    def __setitem__(self,index,value):self.values[index]=int(value)%(2**32)
    def __eq__(self,other):
        return isinstance(other,RingVector) and np.array_equal(self.values,other.values)
    def signed(self):return self.values.view('<i4').astype('<i8')


def signed_input(values):
    a=np.asarray(values)
    if a.ndim!=1 or a.dtype.kind not in 'iu' or a.dtype.itemsize>8:
        raise ValueError('Expected integer updates, without float conversion')
    if a.size and (int(a.min())<-(2**31) or int(a.max())>=2**31):
        raise ValueError('Update outside signed32 encoding range')
    return a.astype('<i8',copy=True)


def validate_bounds(updates):
    # A sufficient public encoding condition: every possible survivor sum fits
    # signed32. Do not infer the lift from the server's claimed aggregate.
    bound=sum(max(abs(int(a.min())),abs(int(a.max()))) if a.size else 0
              for a in updates.values())
    if bound>=2**31:raise ValueError('Signed32 aggregate bound exceeded')


def encode_input(values):return RingVector(values.astype('<u4'))
def zeros(count):return RingVector(np.zeros(count,dtype='<u4'))


class ReplayableMask:
    def __init__(self,key,context,purpose,count):
        if count<0:raise ValueError('Negative mask length')
        self.args=key,context,purpose,count
    def __len__(self):return self.args[-1]
    def materialize(self):
        from sva_fl.wire import enc
        key,context,purpose,count=self.args
        derived=hmac.new(key,enc(['SVA/ChaCha20-uint32/v5',context,purpose]),hashlib.sha256).digest()
        cipher=ChaCha20.new(key=derived,nonce=b'\0'*12)
        # Exactly 4*count stream bytes. No prime-field rejection or widening.
        raw=cipher.encrypt(bytes(4*count))
        return RingVector(np.frombuffer(raw,dtype='<u4').copy())


def add_inplace(vector,other,sign=1):
    if len(vector)!=len(other):raise ValueError('Mask length mismatch')
    if sign not in (-1,1):raise ValueError('Invalid mask sign')
    operation=np.add if sign==1 else np.subtract
    operation(vector.values,other.values,out=vector.values)
    return vector


def apply_mask(vector,mask,sign):return add_inplace(vector,mask.materialize(),sign)


def aggregate(masked,survivors,delta):
    result=zeros(len(delta))
    for uid in survivors:add_inplace(result,masked[uid])
    return add_inplace(result,delta,-1)


def forged_first(vector):
    result=vector[:];result[0]=result[0]+1;return result


def fields(key,context,purpose,count):
    from sva_fl.wire import enc
    # F_q sampling retained for the scalar blinder and challenge, with the new
    # canonical transcript bytes bound into the derivation.
    return field.fields(key,enc(context),purpose,count)


def dot(alpha,values):
    signed=values.signed() if isinstance(values,RingVector) else signed_input(values)
    if len(alpha)!=len(signed):raise ValueError('Projection length mismatch')
    if field.native is not None:
        positive,negative=field.native.projection_signed(alpha.limbs,signed)
        return (int.from_bytes(positive.tobytes(),'little')-
                int.from_bytes(negative.tobytes(),'little'))%Q
    return sum(int(a)*int(x) for a,x in zip(alpha,signed))%Q


def centered(values):return values.signed()

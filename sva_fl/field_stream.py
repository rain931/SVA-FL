"""Native ChaCha20 stream; uniform rejection into the unchanged scalar field."""
import hashlib,hmac
import numpy as np
from Cryptodome.Cipher import ChaCha20
Q=2**252+27742317777372353535851937790883648493
LIMBS=np.array([(Q>>(64*i))&((1<<64)-1) for i in range(4)],dtype='<u8')

def accepted_raw(raw):
    a=np.frombuffer(raw,dtype='<u8').reshape(-1,4).copy()
    a[:,3]&=np.uint64((1<<61)-1)
    smaller=np.zeros(len(a),dtype=bool);equal=np.ones(len(a),dtype=bool)
    for k in [3,2,1,0]:
        smaller|=equal&(a[:,k]<LIMBS[k]);equal&=a[:,k]==LIMBS[k]
    return a[smaller].tobytes()

def iter_fields(key,context,purpose,count,chunk=4096):
    from sva_fl.protocol_core import enc
    if count<0 or chunk<1:raise ValueError('Invalid stream length')
    derived=hmac.new(key,enc(['SVA/ChaCha20-field/v3',context,purpose]),hashlib.sha256).digest()
    cipher=ChaCha20.new(key=derived,nonce=b'\0'*12)
    remaining=count
    while remaining:
        raw=accepted_raw(cipher.encrypt(bytes(32*min(chunk,max(16,2*remaining)))))
        for off in range(0,len(raw),32):
            yield int.from_bytes(raw[off:off+32],'little')
            remaining-=1
            if not remaining:return

def fields(key,context,purpose,count):return list(iter_fields(key,context,purpose,count))

class ReplayableMask:
    """Retain only seed/context; expand bounded blocks when an endpoint uses it."""
    def __init__(self,key,context,purpose,count):self.args=(key,context,purpose,count)
    def __iter__(self):return iter_fields(*self.args)
    def __len__(self):return self.args[-1]

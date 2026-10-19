"""Byte-identical v3 canonical encoding, with packed-vector and streaming hash support."""
import hashlib
from sva_fl.field import FieldVector


def length(n):return n.to_bytes(8,'big')


def chunks(value):
    if isinstance(value,FieldVector):
        # Match v3's empty-list encoding, including inside nested structures.
        if not len(value):
            yield b'L'+length(0);return
        yield b'V'+length(len(value))
        yield memoryview(value.limbs).cast('B')
    elif value is None:yield b'N'
    elif isinstance(value,bool):yield b'T' if value else b'F'
    elif isinstance(value,int):
        raw=abs(value).to_bytes(max(1,(abs(value).bit_length()+7)//8),'little')
        yield b'I'+bytes([value<0])+length(len(raw))+raw
    elif isinstance(value,str):
        raw=value.encode('utf-8');yield b'S'+length(len(raw))+raw
    elif isinstance(value,bytes):yield b'B'+length(len(value))+value
    elif isinstance(value,(list,tuple)):
        if value and all(type(x) is int and 0<=x<2**256 for x in value):
            yield b'V'+length(len(value))
            yield b''.join(x.to_bytes(32,'little') for x in value)
        else:
            yield b'L'+length(len(value))
            for x in value:yield from chunks(x)
    elif isinstance(value,dict):
        if not all(isinstance(k,str) for k in value):raise TypeError('Control keys must be strings')
        yield b'D'+length(len(value))
        for key in sorted(value):
            yield from chunks(key);yield from chunks(value[key])
    else:raise TypeError('Unsupported canonical wire type: '+str(type(value)))


def enc(value):
    if isinstance(value,FieldVector) and not len(value):return b'L'+length(0)
    return b''.join(chunks(value))


def digest(value):
    h=hashlib.sha256()
    for part in chunks(value):h.update(part)
    return h.hexdigest()

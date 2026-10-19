"""Versioned canonical encoding: uint32 vectors occupy four bytes per coordinate."""
import hashlib
from sva_fl.masking import RingVector
from sva_fl.field_wire import chunks as old_chunks, length


def chunks(value):
    if isinstance(value,RingVector):
        yield b'U'+length(len(value))
        yield memoryview(value.values).cast('B')
    elif isinstance(value,dict):
        if not all(isinstance(key,str) for key in value):raise TypeError('Control keys must be strings')
        yield b'D'+length(len(value))
        for key in sorted(value):
            yield from chunks(key);yield from chunks(value[key])
    elif isinstance(value,(list,tuple)) and not (value and all(type(x) is int and 0<=x<2**256 for x in value)):
        yield b'L'+length(len(value))
        for item in value:yield from chunks(item)
    else:yield from old_chunks(value)


def enc(value):return b''.join(chunks(value))
def digest(value):
    h=hashlib.sha256()
    for part in chunks(value):h.update(part)
    return h.hexdigest()

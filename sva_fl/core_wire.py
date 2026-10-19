"""Versioned injective binary encoding; shared by signing, hashing and counting."""
def length(n):return n.to_bytes(8,'big')

def enc(value):
    if value is None:return b'N'
    if isinstance(value,bool):return b'T' if value else b'F'
    if isinstance(value,int):
        magnitude=abs(value)
        raw=magnitude.to_bytes(max(1,(magnitude.bit_length()+7)//8),'little')
        return b'I'+bytes([value<0])+length(len(raw))+raw
    if isinstance(value,str):
        raw=value.encode('utf-8');return b'S'+length(len(raw))+raw
    if isinstance(value,bytes):return b'B'+length(len(value))+value
    if isinstance(value,(list,tuple)):
        if value and all(type(x) is int and 0<=x<2**256 for x in value):
            return b'V'+length(len(value))+b''.join(x.to_bytes(32,'little') for x in value)
        return b'L'+length(len(value))+b''.join(enc(x) for x in value)
    if isinstance(value,dict):
        if not all(isinstance(k,str) for k in value):raise TypeError('Control keys must be strings')
        return b'D'+length(len(value))+b''.join(enc(k)+enc(value[k]) for k in sorted(value))
    raise TypeError('Unsupported canonical wire type: '+str(type(value)))

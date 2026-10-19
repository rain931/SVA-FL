"""Authenticated SVA with four-byte ring masks and bounded signed F_q projections.

The mask domain differs from the manuscript's abstract F_q masking domain.
All ACK, recovery authorization, certificate and commitment checks are retained.
"""
import copy
import hashlib
import inspect
from sva_fl import protocol_core
from sva_fl import masking as vector
from sva_fl.wire import enc,digest


def _derive():
    source=inspect.getsource(protocol_core)
    replacements={
        'from sva_fl.field_stream import ReplayableMask':'from sva_fl.masking import ReplayableMask',
        'from sva_fl.core_wire import enc':'from sva_fl.wire import enc',
        'return hashlib.sha256(enc(x)).hexdigest()':'return _stream_digest(x)',
        'from sva_fl.field_stream import fields as native_fields':'from sva_fl.masking import fields as native_fields',
        'updates = {i:[int(x)%Q for x in update_fn(i)] for i in eligible}':
            'updates = {i:_vector.signed_input(update_fn(i)) for i in eligible}\n            _vector.validate_bounds(updates)',
        'vector, scalar = updates[i][:], 0':'vector, scalar = _vector.encode_input(updates[i]), 0',
        'vector = [(x+sign*y)%Q for x,y in zip(vector,r)]':'vector = _vector.apply_mask(vector,r,sign)',
        'delta, rho_open = [0]*dimension,0':'delta, rho_open = _vector.zeros(dimension),0',
        'r = list(r) # disclose only authorized edge; retain wire-compatible values':'r = r.materialize()',
        'delta = [(a+sign*b)%Q for a,b in zip(delta,r)]':'delta = _vector.add_inplace(delta,r,sign)',
        'y = [(sum(masked[i][k] for i in survivors)-delta[k])%Q for k in range(dimension)]':'y = _vector.aggregate(masked,survivors,delta)',
        'y[0] = (y[0]+1)%Q':'y[0] = y[0]+1',
        'z = sum(a*x for a,x in zip(alpha,updates[i]))%Q':'z = _vector.dot(alpha,updates[i])',
        "tr = {**tr,'y':[(y[0]+1)%Q]+y[1:]}":"tr = {**tr,'y':_vector.forged_first(y)}",
        'rhs = scalar_mul(sum(a*x for a,x in zip(alpha,y))%Q,G)':'rhs = scalar_mul(_vector.dot(alpha,y),G)',
    }
    for old,new in replacements.items():
        if source.count(old)!=1:raise RuntimeError('Frozen source differs: '+old)
        source=source.replace(old,new)
    anchor='        start, self.bytes = time.perf_counter(), 0'
    source=source.replace(anchor,anchor+'''
        _phase_times={}
        _phase_last=start
        def _mark_phase(name):
            nonlocal _phase_last
            now=time.perf_counter()
            _phase_times[name]=now-_phase_last
            _phase_last=now''')
    stages=[('            updates = {i:_vector.signed_input(update_fn(i)) for i in eligible}','admission'),
            ('            pair = {}','input_encoding'),
            ('            masked, rho, acks, held = {},{}, {}, {i:{} for i in p}','mask_setup'),
            ('            survivors = eligible[:]','masking_upload_ack'),
            ('            delta, rho_open = _vector.zeros(dimension),0','recovery_authorization'),
            ('            y = _vector.aggregate(masked,survivors,delta)','recovery_disclosure'),
            ("            tr = {'ctx':ctx_hash,'rec':rec_hash,'S':survivors,'y':y}",'aggregation'),
            ('            commits = []','transcript_challenge'),
            ("            cfinal = self.cert(t,rid,'final',tr_hash,final,offline); phases.append('final')",'commitment_generation'),
            ("            return result(True,'accepted',y)",'final_certificates_and_acceptance')]
    for anchor,name in stages:
        if source.count(anchor)!=1:raise RuntimeError('Phase boundary differs')
        source=source.replace(anchor,"            _mark_phase('"+name+"')\n"+anchor)
    anchor='            return RoundResult(ok,reason,survivors,aggregate,time.perf_counter()-start,self.bytes,phases,refusals)'
    source=source.replace(anchor,'            self.phase_seconds=dict(_phase_times)\n'+anchor)
    namespace={'__name__':__name__,'_vector':vector,'_stream_digest':digest}
    exec(compile(source,'<SVA uint32 masks v5>','exec'),namespace)
    return namespace,hashlib.sha256(source.encode()).hexdigest()


_core,DERIVED_SHA256=_derive()
Protocol=_core['Protocol']
Q=_core['Q'];G=_core['G'];H=_core['H'];IDENTITY=_core['IDENTITY'];ec=_core['ec']
scalar_mul=_core['scalar_mul'];commitment=_core['commitment']
fields=vector.fields;field_dot=vector.dot;centered=vector.centered


def verification_observer_class():
    class ObservedProtocol(Protocol):
        def __init__(self,*args,**kwargs):
            self.public_certificates={}
            self.expected_context=None
            super().__init__(*args,**kwargs)
        def cert(self,t,rid,phase,previous,body,offline):
            result=super().cert(t,rid,phase,previous,body,offline)
            self.public_certificates[phase]=(result,t,rid,phase,previous,body)
            if phase=='ctx':
                self.expected_context=copy.deepcopy(body)
            return result
    return ObservedProtocol


def _derive_verifier():
    from sva_fl.verification import verify_public_result as old
    source=inspect.getsource(old)
    source=source.replace('from sva_fl import protocol_core as core',
                          'from sva_fl import protocol as core')
    source=source.replace("sum(a*x for a,x in zip(alpha,transcript['y']))%core.Q",
                          "core.field_dot(alpha,transcript['y'])")
    namespace={'time':__import__('time')}
    exec(compile(source,'<SVA uint32 public verifier v5>','exec'),namespace)
    return namespace['verify_public_result']


verify_public_result=_derive_verifier()

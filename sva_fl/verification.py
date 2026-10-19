"""Verification of a complete public result against a client's local context."""
import copy
import time
def verification_observer_class():
    from sva_fl.protocol_core import Protocol
    class BenchmarkProtocol(Protocol):
        def __init__(self,*args,**kwargs):
            self.public_certificates={}
            self.expected_context=None
            super().__init__(*args,**kwargs)
        def cert(self,t,rid,phase,previous,body,offline):
            certificate=super().cert(t,rid,phase,previous,body,offline)
            self.public_certificates[phase]=(certificate,t,rid,phase,previous,body)
            if phase=='ctx':
                self.expected_context=copy.deepcopy(body)
            return certificate
    return BenchmarkProtocol


def validate_public_chain(protocol, chain, expected_context):
    """Check cross-phase links after each individual certificate has verified."""
    phases = ('ctx', 'recstate', 'transcript', 'final')
    if len(chain) != len(phases):
        return False
    ctx, rec, transcript, final = (entry[-1] for entry in chain)
    expected = expected_context
    if expected is None:
        expected = getattr(protocol, 'expected_context', None)
    if not isinstance(expected, dict):
        return False
    if not {'eta', 't', 'rid', 'model'}.issubset(expected):
        return False
    if any(ctx.get(key) != value for key, value in expected.items()):
        return False
    if ctx['eta'] != protocol.eta:
        return False

    # These digests were authenticated by the certificate checks above.
    state_hashes = [entry[0]['binding']['state'] for entry in chain]
    previous = ['', *state_hashes[:3]]
    for entry, phase, prior in zip(chain, phases, previous):
        _, t, rid, named_phase, named_previous, _ = entry
        if (t, rid, named_phase, named_previous) != (
            ctx['t'], ctx['rid'], phase, prior
        ):
            return False
    if (
        rec['ctx'] != state_hashes[0]
        or transcript['ctx'] != state_hashes[0]
        or transcript['rec'] != state_hashes[1]
        or final['tr'] != state_hashes[2]
    ):
        return False
    participants, survivors, dropped = ctx['P'], rec['S'], rec['D']
    for roster in (participants, survivors, dropped):
        if not isinstance(roster, list) or len(roster) != len(set(roster)):
            return False
        if not all(isinstance(uid, int) and uid in protocol.keys for uid in roster):
            return False
    return (
        transcript['S'] == survivors
        and not set(survivors).intersection(dropped)
        and set(survivors).union(dropped) == set(participants)
    )


def verify_public_result(protocol,observer_uid,timings=None,expected_context=None):
    from sva_fl import protocol_core as core
    from sva_fl.verification import validate_public_chain
    measured={}
    before=time.perf_counter()
    try:
        phase=time.perf_counter()
        chain=[protocol.public_certificates[p] for p in ['ctx','recstate','transcript','final']]
        if not all(protocol.verify_cert(*args) for args in chain):return False
        if not validate_public_chain(protocol,chain,expected_context):return False
        measured['certificate_chain_seconds']=time.perf_counter()-phase
        transcript=protocol.public_certificates['transcript'][-1]
        final=protocol.public_certificates['final'][-1]
        survivors=transcript['S']
        phase=time.perf_counter()
        if observer_uid not in survivors:return False
        if [packet['uid'] for packet in final['commits']]!=survivors:return False
        transcript_hash=core.digest(transcript)
        measured['roster_and_transcript_hash_seconds']=time.perf_counter()-phase
        phase=time.perf_counter()
        for packet in final['commits']:
            body=packet['body']
            if not protocol.verify_message(packet,'COMMIT',body) or body['uid']!=packet['uid'] or body['tr']!=transcript_hash:return False
        measured['commitment_signature_seconds']=time.perf_counter()-phase
        phase=time.perf_counter()
        product=core.IDENTITY
        for packet in final['commits']:
            point=bytes.fromhex(packet['body']['C'])
            if point!=core.IDENTITY and not core.ec.crypto_core_ed25519_is_valid_point(point):return False
            product=core.ec.crypto_core_ed25519_add(product,point)
        lhs=core.ec.crypto_core_ed25519_sub(product,core.scalar_mul(final['rho_open'],core.H))
        measured['commitment_group_seconds']=time.perf_counter()-phase
        phase=time.perf_counter()
        alpha=core.fields(bytes.fromhex(protocol.eta),transcript,'challenge',len(transcript['y']))
        rhs=core.scalar_mul(sum(a*x for a,x in zip(alpha,transcript['y']))%core.Q,core.G)
        measured['challenge_projection_equation_seconds']=time.perf_counter()-phase
        return lhs==rhs
    except (ValueError,KeyError,TypeError,AttributeError,OverflowError):return False
    finally:
        if timings is not None:
            measured['total_seconds']=time.perf_counter()-before
            timings.update(measured)



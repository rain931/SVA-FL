"""Native ChaCha20/bounded mask streaming/binary wire v3; frozen v2 unchanged.
Single-process transport simulation; native cryptography, no network latency claims.
"""
from __future__ import annotations
import hashlib, hmac, json, math, secrets, time
from dataclasses import dataclass
from sva_fl.field_stream import ReplayableMask
import nacl.bindings as ec
from pqcrypto.kem import ml_kem_768 as kem
from pqcrypto.sign import ml_dsa_44 as dsa

Q = 2**252 + 27742317777372353535851937790883648493
IDENTITY = b'\x01' + b'\x00' * 31
G = ec.crypto_scalarmult_ed25519_base_noclamp((1).to_bytes(32, 'little'))
H = ec.crypto_core_ed25519_from_uniform(hashlib.sha256(b'SVA-FL/independent-H/v1').digest())

from sva_fl.core_wire import enc

def digest(x):
    return hashlib.sha256(enc(x)).hexdigest()

def scalar_mul(s, p):
    s %= Q
    return IDENTITY if s == 0 or p == IDENTITY else ec.crypto_scalarmult_ed25519_noclamp(s.to_bytes(32, 'little'), p)

def commitment(z, rho):
    return ec.crypto_core_ed25519_add(scalar_mul(z, G), scalar_mul(rho, H))

def fields(key, context, purpose, count):
    from sva_fl.field_stream import fields as native_fields
    return native_fields(key, context, purpose, count)

@dataclass
class RoundResult:
    accepted: bool
    reason: str
    survivors: list[int]
    aggregate: list[int] | None
    seconds: float
    bytes: int
    phases: list[str]
    recovery_refusals: int = 0

class Protocol:
    def __init__(self, n=4, degree=3, f_local=1, n_w=None, f_w=1, d_w=0, epoch='pilot-v1'):
        n_w = n if n_w is None else n_w
        if not 0 <= degree < n or n * degree % 2:
            raise ValueError('A simple D-regular graph requires D<n and even n*D')
        if not (0 <= f_local < degree and 0 <= f_w and 0 <= d_w and f_w+d_w < n_w/3 and n_w <= n):
            raise ValueError('Manuscript assumptions violated: f_w+d_w < n_w/3, f_local<D')
        self.n, self.degree, self.f_local = n, degree, f_local
        self.epoch, self.qw = epoch, math.ceil(2*n_w/3)
        self.fw, self.dw = f_w, d_w
        self.keys = {i: dsa.generate_keypair() for i in range(n)}
        self.kems = {i: kem.generate_keypair() for i in range(n)}
        self.bytes, self.signed = 0, {}
        roster = []
        for i in range(n):
            body = {'epoch': epoch, 'uid': i, 'pk': self.kems[i][0].hex()}
            record = self.signed_message(i, 'REGISTER', body)
            assert self.verify_message(record, 'REGISTER', body)
            self.wire(record, n) # upload + relay to other n-1 clients
            roster.append(record)
        self.eta = digest([epoch, roster])
        order = sorted(range(n), key=lambda i: digest(['GRAPH', self.eta, i]))
        self.neighbors = {i: set() for i in range(n)}
        for pos, i in enumerate(order):
            for k in range(1, degree//2+1):
                self.neighbors[i].update([order[(pos+k)%n], order[(pos-k)%n]])
            if degree % 2:
                self.neighbors[i].add(order[(pos+n//2)%n])
        self.edges = [(i,j) for i in range(n) for j in sorted(self.neighbors[i]) if i<j]
        self.seeds = {}
        for i, j in self.edges:
            cipher, shared = kem.encrypt(self.kems[j][0])
            body = {'eta': self.eta, 'from': i, 'to': j, 'ct': cipher.hex()}
            packet = self.signed_message(i, 'KEM', body)
            self.wire(packet, 2) # server relay
            assert self.verify_message(packet, 'KEM', body)
            assert hmac.compare_digest(shared, kem.decrypt(self.kems[j][1], cipher))
            self.seeds[i,j] = shared
        self.witnesses = sorted(range(n), key=lambda i: digest(['WITNESS', self.eta, i]))[:n_w]
        self.setup_seconds = None
        self.setup_bytes = self.bytes
        self.bytes = 0

    def wire(self, body, copies=1):
        self.bytes += len(enc(body)) * copies

    def signed_message(self, uid, kind, body):
        signature = dsa.sign(self.keys[uid][1], enc([kind, body]))
        return {'uid':uid, 'kind':kind, 'body':body, 'sig':signature.hex()}

    def verify_message(self, packet, kind, body):
        try:
            if kind == 'OPEN' and packet.get('uid') != body.get('from'):
                return False
            if 'uid' in body and packet['uid'] != body['uid']:
                return False
            if kind == 'KEM' and packet['uid'] != body['from']:
                return False
            return (packet['kind'] == kind and packet['body'] == body and
                    dsa.verify(self.keys[packet['uid']][0], enc([kind, body]), bytes.fromhex(packet['sig'])))
        except (ValueError, KeyError, TypeError):
            return False

    def cert(self, t, rid, phase, previous, body, offline):
        binding = {'eta':self.eta, 't':t, 'rid':rid, 'phase':phase, 'prev':previous, 'state':digest(body)}
        binding_hash = digest(binding)
        signatures = []
        for uid in self.witnesses:
            if uid in offline:
                continue
            index = (self.eta, t, phase) # deliberately excludes rid
            if index in self.signed and self.signed[index].get(uid) not in (None, binding_hash):
                continue
            self.signed.setdefault(index,{})[uid] = binding_hash
            packet = self.signed_message(uid,'CERT',binding)
            signatures.append(packet)
            self.wire(packet)
        cert = {'binding':binding, 'signatures':signatures}
        self.wire({'state':body,'cert':cert}, self.n)
        if not self.verify_cert(cert, t, rid, phase, previous, body):
            raise ValueError('certificate unavailable: '+phase)
        return cert

    def verify_cert(self, cert, t, rid, phase, previous, body):
        binding = {'eta':self.eta,'t':t,'rid':rid,'phase':phase,'prev':previous,'state':digest(body)}
        if cert.get('binding') != binding:
            return False
        signers = set()
        witness_set = set(self.witnesses)
        # Every signer must verify against exactly the same locally derived bytes.
        # This local value is rebuilt on EVERY certificate verification call.
        encoded_binding = enc(['CERT', binding])
        for packet in cert.get('signatures',[]):
            uid = packet.get('uid')
            if uid in signers or uid not in witness_set:
                return False
            try:
                if (packet['kind'] != 'CERT' or packet['body'] != binding or
                    not dsa.verify(self.keys[uid][0], encoded_binding, bytes.fromhex(packet['sig']))):
                    return False
            except (ValueError, KeyError, TypeError):
                return False
            signers.add(uid)
        return len(signers) >= self.qw

    def run(self, t, model_digest, update_fn, participants=None, pre_live=(), pre_upload=(), witness_offline=(), attack='benign'):
        """update_fn(uid) executes after admission; returns integer weighted update."""
        start, self.bytes = time.perf_counter(), 0
        p = list(range(self.n)) if participants is None else sorted(set(participants))
        if not set(p).issubset(self.keys):
            raise ValueError('participant not in registered roster')
        p_set = set(p)
        live = p_set-set(pre_live)
        upload_online = live-set(pre_upload)
        offline = set(witness_offline)
        phases, survivors, refusals = [], [], 0
        rid = secrets.token_hex(16)
        ctx = {'eta':self.eta,'t':t,'rid':rid,'P':p,'model':model_digest}
        def result(ok, reason, aggregate=None):
            return RoundResult(ok,reason,survivors,aggregate,time.perf_counter()-start,self.bytes,phases,refusals)
        try:
            cctx = self.cert(t,rid,'ctx','',ctx,offline); phases.append('ctx')
            ctx_hash = digest(ctx)
            receipts = {i:set() for i in p}
            for i,j in self.edges:
                if i not in live or j not in live:
                    continue
                for sender,receiver in [(i,j),(j,i)]:
                    body = {'sender':sender,'receiver':receiver,'ctx':ctx_hash}
                    tag = hmac.new(self.seeds[i,j],enc(['LIVE',body]),hashlib.sha256).hexdigest()
                    self.wire({'body':body,'tag':tag},2)
                    if hmac.compare_digest(tag,hmac.new(self.seeds[i,j],enc(['LIVE',body]),hashlib.sha256).hexdigest()):
                        receipts[receiver].add(sender)
            eligible = [i for i in p if len(receipts[i]) >= self.f_local+1 and i in upload_online]
            if not eligible:
                return result(False,'no admitted uploads')
            updates = {i:[int(x)%Q for x in update_fn(i)] for i in eligible}
            dimension = len(updates[eligible[0]])
            if any(len(x)!=dimension for x in updates.values()):
                raise ValueError('update dimensions differ')
            pair = {}
            for i,j in self.edges:
                if i in p_set and j in p_set:
                    pair[i,j] = (ReplayableMask(self.seeds[i,j],ctx_hash,'vector',dimension),fields(self.seeds[i,j],ctx_hash,'scalar',1)[0])
            masked, rho, acks, held = {},{}, {}, {i:{} for i in p}
            for i in eligible:
                vector, scalar = updates[i][:], 0
                for j in sorted(self.neighbors[i] & p_set):
                    r,beta = pair[min(i,j),max(i,j)]
                    sign = 1 if i<j else -1
                    vector = [(x+sign*y)%Q for x,y in zip(vector,r)]
                    scalar = (scalar+sign*beta)%Q
                masked[i],rho[i] = vector,scalar
                body = {'uid':i,'ctx':ctx_hash,'masked':digest(vector)}
                acks[i] = self.signed_message(i,'ACK',body)
                self.wire({'vector':vector,'ack':acks[i]})
                for j in sorted(self.neighbors[i] & p_set):
                    self.wire(acks[i]) # server forwards upload ACK
                    if self.verify_message(acks[i],'ACK',body):
                        held[j][i] = acks[i]
            survivors = eligible[:]
            if attack == 'fake_dropout':
                survivors.remove(eligible[0])
            survivor_set = set(survivors)
            dropped = sorted(p_set-survivor_set)
            recovery = [(i,j) for i in survivors for j in sorted(self.neighbors[i]&set(dropped))]
            rec = {'ctx':ctx_hash,'S':survivors,'D':dropped,'R':recovery,'acks':[acks[i] for i in survivors]}
            # Witnesses validate partition, exact edges, ACK signatures before signing.
            if survivor_set&set(dropped) or survivor_set|set(dropped)!=p_set:
                return result(False,'invalid recovery partition')
            if not all(self.verify_message(acks[i],'ACK',{'uid':i,'ctx':ctx_hash,'masked':digest(masked[i])}) for i in survivors):
                return result(False,'invalid upload ACK')
            rec_hash = digest(rec)
            crec = self.cert(t,rid,'recstate',ctx_hash,rec,offline); phases.append('recstate')
            delta, rho_open = [0]*dimension,0
            signed_responses = []
            for i,j in recovery:
                # Compute refusal from actual neighbor's stored authenticated ACK.
                if j in held[i] or len(set(held[i])&survivor_set) < self.f_local+1:
                    refusals += 1
                    continue
                r,beta = pair[min(i,j),max(i,j)]
                r = list(r) # disclose only authorized edge; retain wire-compatible values
                body = {'rec':rec_hash,'from':i,'to':j,'r':r,'beta':beta}
                packet = self.signed_message(i,'OPEN',body)
                self.wire(packet)
                if not self.verify_message(packet,'OPEN',body):
                    return result(False,'invalid recovery response')
                signed_responses.append(packet)
                sign = 1 if i<j else -1
                delta = [(a+sign*b)%Q for a,b in zip(delta,r)]
                rho_open = (rho_open+sign*beta)%Q
            if any(i not in survivors for i in eligible):
                return result(False,'uploaded online client excluded from survivor set')
            if refusals or len(signed_responses)!=len(recovery):
                return result(False,'recovery disclosure gate refused')
            y = [(sum(masked[i][k] for i in survivors)-delta[k])%Q for k in range(dimension)]
            if attack == 'aggregate_forgery':
                y[0] = (y[0]+1)%Q
            tr = {'ctx':ctx_hash,'rec':rec_hash,'S':survivors,'y':y}
            ctr = self.cert(t,rid,'transcript',rec_hash,tr,offline); phases.append('transcript')
            tr_hash = digest(tr)
            alpha = fields(bytes.fromhex(self.eta),tr,'challenge',dimension)
            commits = []
            for i in survivors:
                z = sum(a*x for a,x in zip(alpha,updates[i]))%Q
                body = {'uid':i,'tr':tr_hash,'C':commitment(z,rho[i]).hex()}
                packet = self.signed_message(i,'COMMIT',body)
                self.wire(packet); commits.append(packet)
            final = {'tr':tr_hash,'commits':commits,'rho_open':rho_open}
            if attack == 'commitment_omission':
                final['commits'] = commits[:-1]
            if attack == 'scalar_forgery':
                final['rho_open'] = (rho_open+1)%Q
            roster = [m['uid'] for m in final['commits']]
            if roster!=survivors or not all(self.verify_message(m,'COMMIT',m['body']) and m['body']['uid']==m['uid'] and m['body']['tr']==tr_hash for m in final['commits']):
                return result(False,'witness rejected commitment roster')
            cfinal = self.cert(t,rid,'final',tr_hash,final,offline); phases.append('final')
            chain = [(cctx,'ctx','',ctx),(crec,'recstate',ctx_hash,rec),(ctr,'transcript',rec_hash,tr),(cfinal,'final',tr_hash,final)]
            if attack == 'context_replay':
                chain[0] = (cctx,'ctx','',{**ctx,'rid':'other-round'})
            if attack == 'split_view':
                tr = {**tr,'y':[(y[0]+1)%Q]+y[1:]}
                chain[2] = (ctr,'transcript',rec_hash,tr)
            if not all(self.verify_cert(c,t,rid,phase,prev,body) for c,phase,prev,body in chain):
                return result(False,'client rejected certificate chain')
            product = IDENTITY
            for packet in final['commits']:
                point = bytes.fromhex(packet['body']['C'])
                if point!=IDENTITY and not ec.crypto_core_ed25519_is_valid_point(point):
                    return result(False,'invalid commitment group element')
                product = ec.crypto_core_ed25519_add(product,point)
            lhs = ec.crypto_core_ed25519_sub(product,scalar_mul(final['rho_open'],H))
            rhs = scalar_mul(sum(a*x for a,x in zip(alpha,y))%Q,G)
            if lhs!=rhs:
                return result(False,'client rejected public verification equation')
            if attack not in {'benign','fake_dropout','aggregate_forgery','commitment_omission','scalar_forgery','context_replay','split_view'}:
                raise ValueError('unknown attack')
            return result(True,'accepted',y)
        except ValueError as exc:
            return result(False,str(exc))

def centered(v):
    return [x if x<=Q//2 else x-Q for x in v]

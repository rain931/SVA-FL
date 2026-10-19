"""Run SVA-FL on local integer update vectors."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import Protocol, centered, verify_public_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--updates', type=Path, help='JSON array containing one integer vector per client')
    parser.add_argument('--clients', type=int, default=4)
    parser.add_argument('--dimension', type=int, default=16)
    parser.add_argument('--degree', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--drop-client', type=int, action='append', default=[])
    args = parser.parse_args()
    if args.seed < 0:
        parser.error('Seed must be nonnegative')
    if args.updates:
        updates = np.asarray(json.loads(args.updates.read_text(encoding='utf-8')))
        if updates.ndim != 2 or updates.dtype.kind not in 'iu' or updates.shape[1] < 1:
            parser.error('Updates must be a nonempty rectangular array of integer vectors')
        n, dimension = updates.shape
    else:
        n, dimension = args.clients, args.dimension
        if n < 4 or dimension < 1:
            parser.error('At least four clients and a positive dimension are required')
        updates = np.random.default_rng(args.seed).integers(
            -1000, 1001, size=(n, dimension), dtype=np.int64
        )
    if n < 4 or not set(args.drop_client).issubset(range(n)):
        parser.error('At least four registered clients and valid dropout IDs are required')
    degree = args.degree if args.degree is not None else min(4, n-1)
    nw = min(60, n)
    protocol = Protocol(
        n=n, degree=degree, n_w=nw, f_w=(nw-1)//3,
        epoch=f'sva-fl-{args.seed}',
    )
    result = protocol.run(
        0, 'local-model', lambda uid: updates[uid],
        pre_upload=args.drop_client,
    )
    aggregate, verified = None, False
    if result.accepted:
        aggregate = centered(result.aggregate).tolist()
        verified = all(
            verify_public_result(protocol, uid) for uid in result.survivors
        )
        if not verified:
            raise RuntimeError('A client rejected the public result')
    print(json.dumps(dict(
        accepted=result.accepted, reason=result.reason, survivors=result.survivors,
        aggregate=aggregate, all_survivors_verified=verified,
    )))
    return 0 if result.accepted and verified else 1


if __name__ == '__main__':
    raise SystemExit(main())

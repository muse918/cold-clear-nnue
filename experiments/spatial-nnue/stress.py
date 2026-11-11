#!/usr/bin/env python3
"""Geometry/accumulator edge cases, deliberately not a gameplay distribution."""
from pathlib import Path
import json
import numpy as np
from train import DTYPE

root=Path(__file__).resolve().parent
rng=np.random.default_rng(918731)
records=[]
counts={}
for i in range(2400):
    record=np.zeros(1,dtype=DTYPE)
    record['game']=100000+i;record['parent_id']=i
    record['next']=i%7;record['selected']=1
    before=np.zeros(40,np.uint16)
    kind=i%4
    if kind in (0,1):
        # I travels down an open rightmost well. A subset of its four rows
        # completes; survivor rows may be noncontiguous in the old coordinates.
        high=int(rng.integers(5,41))
        before[:high]=rng.integers(1,511,high,dtype=np.uint16)
        n=1+(i//4)%4
        for y in rng.choice(4,n,replace=False):before[y]=511
        placed=np.array([10*y+9 for y in range(4)],np.uint16)
        locked=before.copy();locked[:4]|=512
        full=np.flatnonzero(locked==1023)
        kept=np.flatnonzero(locked!=1023)
        after=np.r_[locked[kept],np.zeros(len(full),np.uint16)]
        mapping=np.r_[kept.astype(np.int8),np.full(len(full),-1,np.int8)]
        record['event']=0;record['piece']=0;record['placed']=placed
        record['clears']=len(full);record['clear_mask']=sum(1<<int(y) for y in full)
        name='clear_and_garbage' if kind==1 else 'clear'
        amount=int(rng.integers(1,9)) if kind==1 else 0
    elif kind==2:
        # Ordinary O lock at the top boundary (rows 38 and 39).
        before[:38]=rng.integers(1,1023,38,dtype=np.uint16)
        before[37]|=3
        if before[37]==1023:before[37]^=4
        after=before.copy();after[38:40]=3
        mapping=np.arange(40,dtype=np.int8)
        record['event']=0;record['piece']=1
        record['placed']=[380,381,390,391]
        name='top_boundary_lock';amount=0
    else:
        before[:]=rng.integers(1,1023,40,dtype=np.uint16)
        after=before.copy();mapping=np.arange(40,dtype=np.int8)
        record['event']=1;record['piece']=-1;record['placed']=65535
        name='buffer_garbage';amount=int(rng.integers(1,9))
    if amount:
        added=np.array([1023^(1<<int(rng.integers(10))) for _ in range(amount)],np.uint16)
        after=np.r_[added,after[:-amount]]
        mapping=np.r_[np.full(amount,-1,np.int8),mapping[:-amount]]
        record['garbage']=amount
    record['parent']=before;record['child']=after;record['row_source']=mapping
    records.append(record)
    counts[name]=counts.get(name,0)+1
output=root/'data/stress'
output.mkdir(parents=True,exist_ok=True)
np.concatenate(records).tofile(output/'records.bin')
(output/'metadata.json').write_text(json.dumps({'records':len(records),'categories':counts,
    'scope':'Adversarial row/accumulator arithmetic; includes terminal boards and combined clear+garbage; not frequency/strength data.',
    'seed':918731},indent=2))
print(json.dumps(counts))

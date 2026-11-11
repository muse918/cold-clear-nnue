#!/usr/bin/env python3
"""Held-out ranking and matched-seed closed-loop HARD-DROP surrogate games.

No live service connection. No hold/spins/timing/cancellation. These metrics
are not TETR.IO winrates. Generator's explicit simplified terminal rule applies.
"""
from __future__ import annotations
import argparse
import ctypes as C
import json
from pathlib import Path
import time
import numpy as np
import torch
from train import (DTYPE, boards_and_patches, LocalModel, DenseCNN, RowTransformer,
                   predict, ranking_metrics, export, check_compiled_encoder)

ROOT=Path(__file__).resolve().parent
U16P=C.POINTER(C.c_uint16)
U8P=C.POINTER(C.c_uint8)
FP=C.POINTER(C.c_float)
IP=C.POINTER(C.c_int)


class NativeModel:
    def __init__(self,path):
        self.lib=C.CDLL(str(ROOT/"data/libkernels.so"))
        self.lib.spatial_model_load.argtypes=[C.c_char_p]
        self.lib.spatial_model_load.restype=C.c_void_p
        self.lib.spatial_model_free.argtypes=[C.c_void_p]
        self.lib.spatial_model_predict.argtypes=[C.c_void_p,U16P,U8P,FP,C.c_size_t]
        self.lib.spatial_model_predict.restype=C.c_int
        self.lib.spatial_last_error.restype=C.c_char_p
        self.handle=self.lib.spatial_model_load(str(path.resolve()).encode())
        if not self.handle:
            raise RuntimeError(self.lib.spatial_last_error())

    def __call__(self,rows,next_pieces):
        rows=np.ascontiguousarray(rows,dtype=np.uint16)
        next_pieces=np.ascontiguousarray(next_pieces,dtype=np.uint8)
        out=np.empty(len(rows),np.float32)
        status=self.lib.spatial_model_predict(self.handle,rows.ctypes.data_as(U16P),
            next_pieces.ctypes.data_as(U8P),out.ctypes.data_as(FP),len(rows))
        if status:
            raise RuntimeError(self.lib.spatial_last_error())
        return out

    def close(self):
        if self.handle:
            self.lib.spatial_model_free(self.handle)
            self.handle=None


def torch_model(path):
    ck=torch.load(path,weights_only=False)
    name=ck["model"]
    model=LocalModel(name) if name in {"cell","cnn","attention"} else DenseCNN() if name=="dense_cnn" else RowTransformer()
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model,ck


class TorchPredictor:
    def __init__(self,path):
        self.model,self.ck=torch_model(path)
    @torch.no_grad()
    def __call__(self,rows,next_pieces):
        bits,patches=boards_and_patches(rows)
        nexts=torch.from_numpy(next_pieces.astype(np.int64))
        return self.model(bits,patches,nexts).numpy()*self.ck["target_std"]+self.ck["target_mean"]
    def close(self):
        pass


def games(policy,game_count,limit):
    moves=C.CDLL(str(ROOT/"data/libmoves.so"))
    moves.spatial_moves.argtypes=[U16P,C.c_int,C.c_int,U16P,FP,FP,IP]
    moves.spatial_moves.restype=C.c_int
    boards=np.empty((34,40),np.uint16)
    rewards=np.empty(34,np.float32)
    labels=np.empty(34,np.float32)
    clears=np.empty(34,np.int32)
    results=[]
    for game in range(game_count):
        # Piece queues and garbage schedule are independent of model decisions.
        rng=np.random.default_rng(20261111918+game)
        queue=np.concatenate([rng.permutation(7) for _ in range((limit+14)//7)])
        grng=np.random.default_rng(20261111918+game+100000)
        packets={t:(int(grng.integers(1,3)),int(grng.integers(0,10)))
                 for t in range(15,limit,15)} if game%2 else {}
        row=np.zeros(40,np.uint16)
        total_lines,total_reward,total_garbage=0,0.0,0
        steps=0; topout=False; max_height=0
        for step in range(limit):
            if step in packets:
                amount,hole=packets[step]
                row=np.r_[np.full(amount,1023^(1<<hole),np.uint16),row[:-amount]]
                total_garbage+=amount
                high=int(np.flatnonzero(row)[-1]+1) if np.any(row) else 0
                if high>20:
                    topout=True;break
            next_piece=int(queue[step+1])
            n=moves.spatial_moves(row.ctypes.data_as(U16P),int(queue[step]),next_piece,
                boards.ctypes.data_as(U16P),rewards.ctypes.data_as(FP),
                labels.ctypes.data_as(FP) if policy is None else None,clears.ctypes.data_as(IP))
            if n<=0:
                topout=True;break
            values=labels[:n] if policy is None else policy(boards[:n],np.full(n,next_piece,np.uint8))
            selected=int(np.argmax(rewards[:n]+values))
            row=boards[selected].copy()
            steps+=1
            total_lines+=int(clears[selected]);total_reward+=float(rewards[selected])
            high=int(np.flatnonzero(row)[-1]+1) if np.any(row) else 0
            max_height=max(max_height,high)
            if high>20:
                topout=True;break
        results.append({"game":game,"condition":"garbage" if game%2 else "clean",
                        "pieces":steps,"lines":total_lines,"reward":total_reward,
                        "garbage_received":total_garbage,"topout":topout,"max_height":max_height})
    def summary(rs):
        return {"games":len(rs),"mean_pieces":float(np.mean([x["pieces"] for x in rs])),
                "mean_lines":float(np.mean([x["lines"] for x in rs])),
                "capped_survival_rate":float(np.mean([not x["topout"] for x in rs]))}
    return {"overall":summary(results),"clean":summary(results[::2]),
            "garbage":summary(results[1::2]),"games":results,
            "scope":"Matched PCG64 piece/garbage seeds; no hold/spin/timing; height>20 terminal; fixed piece cap, not infinite survival."}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--models",type=Path,default=ROOT/"results/final")
    p.add_argument("--heldout",type=Path,default=ROOT/"data/heldout/records.bin")
    p.add_argument("--rollout-games",type=int,default=24)
    p.add_argument("--rollout-pieces",type=int,default=160)
    p.add_argument("--seeds",default="918,919,920")
    p.add_argument("--out",type=Path,default=ROOT/"results/final/evaluation.json")
    args=p.parse_args()
    torch.set_num_threads(2)
    records=np.fromfile(args.heldout,dtype=DTYPE)
    ids=np.arange(len(records))
    bits,patches=boards_and_patches(records["child"])
    nexts=torch.from_numpy(records["next"].astype(np.int64))
    result={"heldout_records":len(records),"heldout_games":sorted(map(int,np.unique(records["game"]))),
            "rollout_games":args.rollout_games,"rollout_piece_cap":args.rollout_pieces,"models":{}}
    for seed in map(int,args.seeds.split(",")):
        directory=args.models/f"seed{seed}"
        names=["cell","cnn","attention"]+(["dense_cnn","transformer"] if seed==918 else [])
        for name in names:
            path=directory/f"{name}.pt"
            model,ck=torch_model(path)
            prediction=predict(model,bits,patches,nexts,ids)*ck["target_std"]+ck["target_mean"]
            entry={"ranking":ranking_metrics(records,ids,prediction)}
            for parity,label in [(0,"clean"),(1,"garbage")]:
                ix=ids[records["game"]%2==parity]
                entry[f"ranking_{label}"]=ranking_metrics(records,ix,prediction[ix])
            if isinstance(model,LocalModel):
                # Re-export also supplies encoder weights for the independent native checker.
                export(model,directory/f"{name}.snn",ck["target_mean"],ck["target_std"])
                if name!="cell":entry["compiled_encoder_parity"]=check_compiled_encoder(model)
                policy=NativeModel(directory/f"{name}.snn")
                native=policy(records["child"],records["next"])
                error=np.abs(native-prediction)
                entry["native_parity"]={"max_abs":float(error.max()),"mean_abs":float(error.mean())}
                if error.max()>0.003:raise AssertionError(f"Native parity failure {name}: {error.max()}")
                native.tofile(directory/f"{name}_heldout.f32")
            else:
                policy=TorchPredictor(path)
            # Match the same queues/garbage across independently trained models.
            # Replicated surrogate rollouts still do not measure versus strength.
            if args.rollout_games:
                entry["rollout"]=games(policy,args.rollout_games,args.rollout_pieces)
            policy.close()
            result["models"][f"{name}_seed{seed}"]=entry
            args.out.parent.mkdir(parents=True,exist_ok=True)
            args.out.write_text(json.dumps(result,indent=2))
            print(json.dumps({"done":name,"seed":seed,"ranking":entry["ranking"],
                              "rollout":entry.get("rollout",{}).get("overall")}),flush=True)
    if args.rollout_games:
        result["surrogate_teacher_rollout"]=games(None,args.rollout_games,args.rollout_pieces)
    args.out.write_text(json.dumps(result,indent=2))


if __name__=="__main__":main()

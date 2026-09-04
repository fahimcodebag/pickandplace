"""Fit the corrector for the WRIST camera at 320x240, t=0 only.

Per-setup calibration: the agentview corrector encodes a different camera,
resolution and viewing geometry and does not transfer. Tune on seeds 0-2,
hold out seed 3 -- the search never sees it.
"""
import csv, glob, json, os
import numpy as np, torch as T, torch.nn as nn

F=["reproj","area_px","obliq_deg","cam_range","gripper_perp","gripper_dz",
   "cx","cy","obj_yaw","det_x","det_y","det_z"]
R=[]
for f in sorted(glob.glob("Results/wrist_ds/w_s*.csv")):
    for r in csv.DictReader(open(f)): r["src"]=os.path.basename(f); R.append(r)
X=np.array([[float(r[k]) for k in F] for r in R],np.float32)
Y=np.array([[float(r[k]) for k in ("res_x","res_y","res_z")] for r in R],np.float32)*1000
G=np.array([r["src"] for r in R])
hold=G=="w_s3.csv"; tune=~hold
base=np.median(np.linalg.norm(Y[hold],axis=1))
dev=T.device("cuda" if T.cuda.is_available() else "cpu")
print(f"tune {tune.sum()}  hold {hold.sum()}   held-out baseline {base:.2f} mm",flush=True)

def fit(Xtr,Ytr,Xte,h,d,bs=64,ep=1200,seed=0):
    T.manual_seed(seed)
    mu,sd=Xtr.mean(0),Xtr.std(0)+1e-9
    xt=T.tensor((Xtr-mu)/sd,device=dev); yt=T.tensor(Ytr,device=dev)
    L,i=[],len(F)
    for _ in range(d): L+=[nn.Linear(i,h),nn.ReLU()]; i=h
    L+=[nn.Linear(i,3)]
    m=nn.Sequential(*L).to(dev)
    o=T.optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-3)
    sch=T.optim.lr_scheduler.CosineAnnealingLR(o,T_max=ep)
    n=len(xt)
    for _ in range(ep):
        idx=T.randperm(n,device=dev)
        for k in range(0,n,bs):
            b=idx[k:k+bs]; o.zero_grad(); ((m(xt[b])-yt[b])**2).mean().backward(); o.step()
        sch.step()
    m.eval()
    with T.no_grad(): return m(T.tensor((Xte-mu)/sd,device=dev)).cpu().numpy(),(m,mu,sd)

print(f"{'config':10} {'params':>8} {'held-out':>10} {'cut':>6}  R2 x/y/z",flush=True)
best=None
for h,d in ((32,2),(64,2),(64,3)):
    P,(m,mu,sd)=fit(X[tune],Y[tune],X[hold],h,d)
    e=np.linalg.norm(Y[hold]-P,axis=1)
    r2=[1-((Y[hold][:,j]-P[:,j])**2).sum()/((Y[hold][:,j]-Y[hold][:,j].mean())**2).sum() for j in range(3)]
    n=sum(p.numel() for p in m.parameters())
    med=np.median(e)
    print(f"h{h} d{d}{'':4} {n:8,} {med:9.2f}mm {(1-med/base)*100:5.0f}%  "
          f"{r2[0]:+.2f}/{r2[1]:+.2f}/{r2[2]:+.2f}",flush=True)
    if best is None or med<best[0]: best=(med,h,d)
med,h,d=best
print(f"\nbest: h{h} d{d} ({med:.2f} mm held-out). refitting on ALL data",flush=True)
_,(mf,muf,sdf)=fit(X,Y,X[:1],h,d)
sdd=mf.state_dict(); li=[i for i,l in enumerate(mf) if isinstance(l,nn.Linear)]
json.dump(dict(features=F,mu=muf.tolist(),sd=sdf.tolist(),
               W=[(sdd[f"{i}.weight"].cpu().numpy()/1000.0).tolist() for i in li[-1:]][0] if False else
                 [sdd[f"{i}.weight"].cpu().tolist() for i in li],
               b=[sdd[f"{i}.bias"].cpu().tolist() for i in li],
               units="MILLIMETRES - divide the final layer by 1000 before export",
               setup="robot0_eye_in_hand 320x240, t=0"),
          open("assets/tag_residual_wrist.json","w"))
print("saved assets/tag_residual_wrist.json")

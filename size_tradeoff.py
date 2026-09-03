import sys, json
import numpy as np, torch as T, torch.nn as nn
exec(open("tune_residual.py").read().split('def main()')[0])
X,Y,G=load()
tune=np.isin(G,list(TUNE)); hold=np.isin(G,list(HOLD))
base=np.median(np.linalg.norm(Y[hold],axis=1))
print(f"held-out baseline {base:.2f} mm\n{'config':16} {'params':>9} {'held-out':>10} {'cut':>6}  R2 x/y/z",flush=True)
best=None
for h,d in ((32,2),(64,2),(64,3),(128,2),(128,3),(256,3)):
    P,(m,mu,sd)=fit(X[tune],Y[tune],X[hold],h,d,64,1500,1e-3,1e-3)
    med,_,r2=score(Y[hold],P); n=sum(p.numel() for p in m.parameters())
    print(f"h{h} d{d} bs64{'':6} {n:9,} {med:9.2f}mm {(1-med/base)*100:5.0f}%  "
          f"{r2[0]:+.2f}/{r2[1]:+.2f}/{r2[2]:+.2f}",flush=True)
    if h==64 and d==2: best=(h,d)
# refit the SMALL deployable config on all data
h,d=best
_,(mf,muf,sdf)=fit(X,Y,X[:1],h,d,64,1500,1e-3,1e-3)
sdd=mf.state_dict(); li=[i for i,l in enumerate(mf) if isinstance(l,nn.Linear)]
json.dump(dict(features=F,mu=muf.tolist(),sd=sdf.tolist(),
               W=[sdd[f"{i}.weight"].cpu().tolist() for i in li],
               b=[sdd[f"{i}.bias"].cpu().tolist() for i in li],
               config=dict(h=h,depth=d,bs=64,ep=1500,lr=1e-3,wd=1e-3)),
          open("assets/tag_residual_small.json","w"))
print(f"\nsaved assets/tag_residual_small.json (h{h} d{d}, {sum(p.numel() for p in mf.parameters()):,} params)")

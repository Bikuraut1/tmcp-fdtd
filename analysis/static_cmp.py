import sys, os; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tmcp_paths import *  # ROOT, DATA, R1, R2, R2C, R3, COMM, OUT, FIG, TAB
import numpy as np, sys, os
import fdtd_core as fc
from scipy.signal import argrelmin
FC=5.754e9
def summ(p):
    d=np.load(p); f=d["freqs"][:-1]; ar=fc.axial_ratio(d["Ex"][:-1],d["Ey"][:-1])
    ff=np.linspace(5.6e9,5.95e9,3501); a=np.interp(ff,f,ar); k=np.argmin(a)
    s=d["S11_dB"]; fs=d["f"]; m=(fs>5.5e9)&(fs<6.4e9); fs,s=fs[m],s[m]
    mins=[(fs[i]/1e9,s[i]) for i in argrelmin(s,order=20)[0] if s[i]<-5]
    b=fs[s<-10]
    R,L=fc.cp_components(d["Ex"][-1],d["Ey"][-1])
    Et,Ep=d["Et_fc"],d["Ep_fc"]; U=(abs(Et)**2+abs(Ep)**2)/(2*fc.eta0)
    eta=float(d["Prad_fc"]/d["Pacc_fc"]); D=10*np.log10(4*np.pi*U[0,0]/d["Prad_fc"])
    # Q from energy decay
    e=d["energy"]; t=np.arange(len(e))*float(d["dt"]); nz=e>0; te,ee=t[nz],e[nz]
    sel=(te>3e-9)&(ee>ee.max()*1e-6)&(ee<ee.max()*1e-1)
    Q=np.nan
    if sel.sum()>5:
        sl=np.polyfit(te[sel],np.log(ee[sel]),1)[0]; Q=2*np.pi*FC/(-sl)
    return dict(ARmin=a[k],fARmin=ff[k]/1e9,band=(ff[a<3][0]/1e9,ff[a<3][-1]/1e9) if (a<3).any() else None,
        S11band=(b[0]/1e9,b[-1]/1e9) if len(b) else None, mins=mins, ARfc=float(fc.axial_ratio(d["Ex"][-1],d["Ey"][-1])),
        XPD=float(20*np.log10(abs(L)/abs(R))), eta=eta, D=D, Q=Q, S11fc=float(np.interp(FC,d["f"],d["S11_dB"])))
if __name__=="__main__":
    for p in sys.argv[1:]:
        r=summ(p); print(os.path.basename(p), {k:(np.round(v,4) if isinstance(v,float) else v) for k,v in r.items()})

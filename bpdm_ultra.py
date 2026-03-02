"""
BPDM minimal-m — ULTRA-LEAN: only 8 k-points, shells 2..7.
eigh costs ~2s at dim=150, ~33s at dim=726.
8 k-points × 7 = 56 eigh calls per (m,ns) pair.
"""
import numpy as np, json, time
from scipy.linalg import eigh

def rot(t):
    c,s=np.cos(t),np.sin(t);return np.array([[c,-s],[s,c]])
def kagome_H(kx,ky,t=1.0):
    a1=np.array([1,0]);a2=np.array([0.5,np.sqrt(3)/2]);k=np.array([kx,ky])
    f12=t*(1+np.exp(-1j*np.dot(k,a1)));f13=t*(1+np.exp(-1j*np.dot(k,a2)))
    f23=t*(1+np.exp(-1j*np.dot(k,a1-a2)))
    return np.array([[0,f12,f13],[f12.conj(),0,f23],[f13.conj(),f23.conj(),0]])
def build(kx,ky,th,ns,tp=0.030):
    b1=2*np.pi*np.array([1,-1/np.sqrt(3)]);b2=2*np.pi*np.array([0,2/np.sqrt(3)])
    K=(2*b1+b2)/3;q1=rot(th/2)@K-rot(-th/2)@K
    q2=rot(-2*np.pi/3)@q1;q3=rot(2*np.pi/3)@q1;G1=q1-q3;G2=q2-q1
    offs=[(0,0),(0,1),(-1,0)];Ti=tp*np.eye(3)
    sh=[(m1,m2) for m1 in range(-ns,ns+1) for m2 in range(-ns,ns+1)]
    nG=len(sh);sm={s:i for i,s in enumerate(sh)};no=3;dim=2*no*nG
    H=np.zeros((dim,dim),dtype=complex)
    for ig,(m1,m2) in enumerate(sh):
        G=m1*G1+m2*G2
        H[no*ig:no*ig+no,no*ig:no*ig+no]=kagome_H(kx+G[0],ky+G[1])
        i2=no*nG+no*ig;H[i2:i2+no,i2:i2+no]=kagome_H(kx+G[0]+q1[0],ky+G[1]+q1[1])
    for ig,(m1,m2) in enumerate(sh):
        for j in range(3):
            d1,d2=offs[j];tgt=(m1+d1,m2+d2)
            if tgt in sm:
                ig2=sm[tgt];i1=no*ig;i2=no*nG+no*ig2
                H[i1:i1+no,i2:i2+no]+=Ti;H[i2:i2+no,i1:i1+no]+=Ti
    ev,vec=eigh(H);return ev,vec,G1,G2,q1,dim,nG

def bscores(vec,kx,ky,th,dim,nG,bidx):
    b1=2*np.pi*np.array([1,-1/np.sqrt(3)]);b2=2*np.pi*np.array([0,2/np.sqrt(3)])
    K=(2*b1+b2)/3;K1=rot(-th/2)@K;K2=rot(th/2)@K;k=np.array([kx,ky]);no=3
    sc=np.zeros(len(bidx))
    for l in range(2):
        Ke=K1 if l==0 else K2;Re=rot(-th/2) if l==0 else rot(th/2)
        dk=Re@(k-Ke);n=np.linalg.norm(dk);dh=dk/n if n>1e-12 else np.array([1.,0.])
        dm=dh[0]-1j*dh[1];dp=dh[0]+1j*dh[1]
        for ig in range(nG):
            idx=no*ig if l==0 else no*nG+no*ig
            for bi,bn in enumerate(bidx):
                p0=vec[idx,bn];p1=vec[idx+1,bn]
                sc[bi]+=np.real(p0.conj()*dm*p1+p1.conj()*dp*p0)
    return sc

def get_kpts(G1,G2):
    """8 strategic k-points: Γ, K, M, midpoints."""
    Ga=np.zeros(2);KM=(2*G1+G2)/3;MM=G1/2
    return [Ga, KM, MM,
            0.5*KM, 0.5*(KM+MM), 0.5*MM,
            0.25*KM, 0.75*KM]

def run(th_deg,ns,m):
    th=np.radians(th_deg)
    _,_,G1,G2,q1,dim,nG=build(0,0,th,ns)
    kpts=get_kpts(G1,G2)
    nw=max(m+4,int(dim*0.4));mid=dim//2
    lo=max(0,mid-nw//2);hi=min(dim,mid+nw//2);widx=list(range(lo,hi))
    all_eps,purs,pvecs=[],[],[]
    for k in kpts:
        ev,vec,_,_,_,_,_=build(k[0],k[1],th,ns)
        sc=bscores(vec,k[0],k[1],th,dim,nG,widx)
        order=np.argsort(-np.abs(sc));sel=order[:m];sgl=[widx[i] for i in sel]
        purs.append(float(np.mean(np.abs(sc[sel]))))
        all_eps.append(sorted([ev[n] for n in sgl]))
        pvecs.append(vec[:,sgl])
    ae=np.array(all_eps);W=float(np.max(ae)-np.min(ae))
    return {'W':W,'mp':float(np.mean(purs)),'minp':float(np.min(purs)),'pv':pvecs,'dim':dim}

def xov(pA,pB,m):
    nk=min(len(pA),len(pB));S=[]
    for ik in range(nk):
        UA,UB=pA[ik],pB[ik];dA,dB=UA.shape[0],UB.shape[0];d=max(dA,dB)
        if dA<d:UA=np.vstack([UA,np.zeros((d-dA,m),dtype=complex)])
        if dB<d:UB=np.vstack([UB,np.zeros((d-dB,m),dtype=complex)])
        S.append(float(np.sum(np.abs(UA.conj().T@UB)**2)))
    return np.array(S)

def main():
    t0=time.time();th=9.6
    ms=[2,4,6,8,10,14,20];shells=list(range(2,8))
    print(f"BPDM MINIMAL-m (ULTRA): Kagome θ={th}°, 8 k-pts",flush=True)
    print("="*80,flush=True)
    res={}
    for m in ms:
        print(f"\n--- m={m} ---",flush=True)
        print(f"{'ns':>3} {'dim':>5} {'W/t':>9} {'pur':>5} {'minp':>5} {'Sm':>7} {'Sn':>7} {'Δ%':>6}",flush=True)
        pp=None;pW=None;mr=[]
        for ns in shells:
            t1=time.time();r=run(th,ns,m);dt=time.time()-t1
            if pp:
                S=xov(pp,r['pv'],m);Sm=float(np.mean(S));Sn=float(np.min(S))
            else:Sm=Sn=float('nan')
            dp=abs(r['W']-pW)/max(r['W'],1e-10)*100 if pW else float('nan')
            mr.append({'ns':ns,'dim':r['dim'],'W':r['W'],'mp':r['mp'],'minp':r['minp'],
                       'Sm':Sm,'Sn':Sn,'dp':dp})
            sm=f"{Sm:.4f}" if not np.isnan(Sm) else "—"
            sn=f"{Sn:.4f}" if not np.isnan(Sn) else "—"
            d=f"{dp:.1f}" if not np.isnan(dp) else "—"
            print(f"{ns:>3} {r['dim']:>5} {r['W']:>9.4f} {r['mp']:>5.3f} {r['minp']:>5.3f} {sm:>7} {sn:>7} {d:>6}  ({dt:.0f}s)",flush=True)
            pp=r['pv'];pW=r['W']
            if dt>90:print(f"  (stop m={m})",flush=True);break
        res[m]=mr
    
    print(f"\n{'='*80}",flush=True)
    print("SUMMARY",flush=True)
    print(f"{'m':>3} {'W_min':>7} {'W_max':>7} {'ratio':>6} {'Sn_last':>8} {'last2Δ':>16} {'conv':>6}",flush=True)
    for m in ms:
        if m not in res:continue
        rs=res[m];ws=[r['W'] for r in rs];ratio=max(ws)/max(min(ws),1e-10)
        sn=[r['Sn'] for r in rs if not np.isnan(r['Sn'])]
        ch=[r['dp'] for r in rs if not np.isnan(r['dp'])]
        ls=f"{sn[-1]:.3f}" if sn else "—"
        l2=ch[-2:] if len(ch)>=2 else ch
        # Convergence: W stable (<15%) AND subspace somewhat stable (S_min > 0.3*m)
        conv="✓" if(len(l2)>=2 and all(c<15 for c in l2) and sn and sn[-1]>0.3*m) else "✗"
        print(f"{m:>3} {min(ws):>7.3f} {max(ws):>7.3f} {ratio:>6.2f}x {ls:>8} {str([f'{c:.0f}%' for c in l2]):>16} {conv:>6}",flush=True)
    
    print(f"\nTotal: {time.time()-t0:.0f}s",flush=True)
    sv={str(k):v for k,v in res.items()}
    with open('/home/darre/.openclaw/workspace/quantum-computer/numerics/bpdm_minimal_m.json','w') as f:
        json.dump(sv,f,indent=2)

if __name__=='__main__':main()

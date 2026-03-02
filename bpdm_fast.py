"""
BPDM minimal-m — FAST version.
Compute b_n(k) without forming full B matrix: O(dim) per band, not O(dim²).
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

def boundary_scores(vec, kx, ky, th, dim, nG, band_idx):
    """
    Compute b_n(k) = ⟨ψ_n|B|ψ_n⟩ for bands in band_idx.
    WITHOUT forming the full B matrix.
    
    B acts block-diagonally: for each shell ig and layer l,
    B[idx:idx+3, idx:idx+3] = d_hat[0]*σ_x + d_hat[1]*σ_y (embedded in 3×3).
    
    So b_n = Σ_{ig,l} ψ†[block] · (d̂·σ) · ψ[block]
    where block = (layer, shell) indices.
    """
    b1=2*np.pi*np.array([1,-1/np.sqrt(3)]);b2=2*np.pi*np.array([0,2/np.sqrt(3)])
    K=(2*b1+b2)/3;K1=rot(-th/2)@K;K2=rot(th/2)@K
    k=np.array([kx,ky]);no=3
    
    scores = np.zeros(len(band_idx))
    
    for layer in range(2):
        Ke=K1 if layer==0 else K2;Re=rot(-th/2) if layer==0 else rot(th/2)
        dk=Re@(k-Ke);n=np.linalg.norm(dk)
        dh=dk/n if n>1e-12 else np.array([1.,0.])
        
        # d̂·σ in 3×3: only acts on sublattices 0,1
        # (d̂·σ)_00 = 0, (d̂·σ)_01 = dx-i*dy, (d̂·σ)_10 = dx+i*dy, rest = 0
        # Actually σ_x: (0,1)=1,(1,0)=1; σ_y: (0,1)=-i,(1,0)=i
        # So d̂·σ: (0,1) = dx - i*dy, (1,0) = dx + i*dy
        dplus = dh[0] + 1j*dh[1]   # (1,0) element
        dminus = dh[0] - 1j*dh[1]  # (0,1) element
        
        for ig in range(nG):
            idx = no*ig if layer==0 else no*nG+no*ig
            # For each band n:
            # contribution = ψ*[idx+0]*dminus*ψ[idx+1] + ψ*[idx+1]*dplus*ψ[idx+0]
            # = 2*Re(ψ*[idx+0]*dminus*ψ[idx+1])  since Hermitian
            for bi, n in enumerate(band_idx):
                p0 = vec[idx, n]
                p1 = vec[idx+1, n]
                scores[bi] += np.real(p0.conj() * dminus * p1 + p1.conj() * dplus * p0)
    
    return scores


def run(th_deg, ns, m, nk=30):
    th=np.radians(th_deg)
    _,_,G1,G2,q1,dim,nG=build(0,0,th,ns)
    Ga=np.zeros(2);KM=(2*G1+G2)/3;MM=G1/2
    segs=[(Ga,KM,nk),(KM,MM,nk//2),(MM,Ga,nk//2)]
    kpts=[]
    for ka,kb,n in segs:
        for i in range(n):kpts.append(ka+(i/n)*(kb-ka))
    
    nw=max(m+4,int(dim*0.4));mid=dim//2
    lo=max(0,mid-nw//2);hi=min(dim,mid+nw//2);widx=list(range(lo,hi))
    
    all_eps,purs,pvecs=[],[],[]
    for k in kpts:
        ev,vec,_,_,_,_,_=build(k[0],k[1],th,ns)
        # Fast boundary scores
        sc=boundary_scores(vec,k[0],k[1],th,dim,nG,widx)
        # Sort by |score|
        order=np.argsort(-np.abs(sc))
        sel_local=order[:m]  # indices into widx
        sel_global=[widx[i] for i in sel_local]
        
        purs.append(float(np.mean(np.abs(sc[sel_local]))))
        all_eps.append(sorted([ev[n] for n in sel_global]))
        pvecs.append(vec[:,sel_global])
    
    ae=np.array(all_eps);W=float(np.max(ae)-np.min(ae))
    return {'W':W,'mp':float(np.mean(purs)),'minp':float(np.min(purs)),'pv':pvecs,'dim':dim}

def xoverlap(pA,pB,m):
    nk=min(len(pA),len(pB));S=[]
    for ik in range(nk):
        UA,UB=pA[ik],pB[ik];dA,dB=UA.shape[0],UB.shape[0];d=max(dA,dB)
        if dA<d:UA=np.vstack([UA,np.zeros((d-dA,m),dtype=complex)])
        if dB<d:UB=np.vstack([UB,np.zeros((d-dB,m),dtype=complex)])
        S.append(float(np.sum(np.abs(UA.conj().T@UB)**2)))
    return np.array(S)

def main():
    t0=time.time();th=9.6
    ms=[2,4,6,8,10];shells=range(2,8)
    print(f"BPDM MINIMAL-m (FAST): Kagome θ={th}°",flush=True)
    print("="*80,flush=True)
    res={}
    for m in ms:
        print(f"\n--- m={m} ---",flush=True)
        print(f"{'ns':>3} {'dim':>5} {'W/t':>9} {'pur':>5} {'minp':>5} {'Sm':>7} {'Sn':>7} {'Δ%':>6}",flush=True)
        pp=None;pW=None;mr=[]
        for ns in shells:
            t1=time.time();r=run(th,ns,m);dt=time.time()-t1
            if pp is not None:
                S=xoverlap(pp,r['pv'],m);Sm=float(np.mean(S));Sn=float(np.min(S))
            else:Sm=Sn=float('nan')
            dp=abs(r['W']-pW)/max(r['W'],1e-10)*100 if pW else float('nan')
            mr.append({'ns':ns,'dim':r['dim'],'W':r['W'],'mp':r['mp'],'minp':r['minp'],
                       'Sm':Sm,'Sn':Sn,'dp':dp})
            sm=f"{Sm:.4f}" if not np.isnan(Sm) else "—"
            sn=f"{Sn:.4f}" if not np.isnan(Sn) else "—"
            d=f"{dp:.1f}" if not np.isnan(dp) else "—"
            print(f"{ns:>3} {r['dim']:>5} {r['W']:>9.4f} {r['mp']:>5.3f} {r['minp']:>5.3f} {sm:>7} {sn:>7} {d:>6}  ({dt:.0f}s)",flush=True)
            pp=r['pv'];pW=r['W']
            if dt>120:print(f"  (stop)",flush=True);break
        res[m]=mr
    
    print(f"\n{'='*80}",flush=True)
    print("SUMMARY",flush=True)
    print(f"{'m':>3} {'W_min':>7} {'W_max':>7} {'ratio':>6} {'S_min':>7} {'last2Δ':>14} {'conv':>6}",flush=True)
    for m in ms:
        if m not in res:continue
        rs=res[m];ws=[r['W'] for r in rs];ratio=max(ws)/max(min(ws),1e-10)
        sn=[r['Sn'] for r in rs if not np.isnan(r['Sn'])]
        ch=[r['dp'] for r in rs if not np.isnan(r['dp'])]
        ls=f"{sn[-1]:.3f}" if sn else "—"
        l2=ch[-2:] if len(ch)>=2 else ch
        conv="✓" if(len(l2)>=2 and all(c<15 for c in l2) and sn and sn[-1]>0.5*m) else "✗"
        print(f"{m:>3} {min(ws):>7.3f} {max(ws):>7.3f} {ratio:>6.2f}x {ls:>7} {str([f'{c:.0f}%' for c in l2]):>14} {conv:>6}",flush=True)
    
    print(f"\nTotal: {time.time()-t0:.0f}s",flush=True)
    sv={str(k):v for k,v in res.items()}
    with open('/home/darre/.openclaw/workspace/quantum-computer/numerics/bpdm_minimal_m.json','w') as f:
        json.dump(sv,f,indent=2)

if __name__=='__main__':main()

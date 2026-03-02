"""
Coarse-grained BPDM: refinement channel Λ_N.
Instead of raw cross-shell overlap, project N+1 back onto N's Hilbert space
via partial trace over new shells, then measure overlap.

The key insight: S_min=0 because we're comparing vectors in DIFFERENT
Hilbert spaces (dim=1014 vs dim=1350). The refinement channel Λ discards
the new degrees of freedom, giving a fair comparison.

Channel A: Partial trace — discard new G-vector blocks.
The N-shell basis has (2N+1)² G-vectors per layer. The (N+1)-shell basis
has (2(N+1)+1)² G-vectors. The "shared" G-vectors are those with
|m1|≤N and |m2|≤N. We project the N+1 eigenvectors onto these shared
components and renormalise.
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

def build_with_shells(kx,ky,th,ns,tp=0.030):
    """Returns ev, vec, shell_list, G1, G2, q1, dim, nG, no."""
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
    ev,vec=eigh(H)
    return ev,vec,sh,G1,G2,q1,dim,nG,no

def refinement_channel(vec_big, shells_big, shells_small, nG_big, nG_small, no, band_idx):
    """
    Λ_N: project vec from (N+1)-shell onto N-shell subspace.
    Keep only components corresponding to G-vectors shared with smaller shell.
    Returns projected + renormalised vectors.
    """
    # Map: for each G in shells_small, find its index in shells_big
    sm_big = {s:i for i,s in enumerate(shells_big)}
    
    dim_small = 2 * no * nG_small
    n_bands = len(band_idx)
    projected = np.zeros((dim_small, n_bands), dtype=complex)
    
    for ig_s, g in enumerate(shells_small):
        if g not in sm_big:
            continue
        ig_b = sm_big[g]
        # Layer 1
        for orb in range(no):
            row_big = no * ig_b + orb
            row_small = no * ig_s + orb
            projected[row_small, :] = vec_big[row_big, band_idx]
        # Layer 2
        for orb in range(no):
            row_big = no * nG_big + no * ig_b + orb
            row_small = no * nG_small + no * ig_s + orb
            projected[row_small, :] = vec_big[row_big, band_idx]
    
    # Renormalise each column
    for j in range(n_bands):
        norm = np.linalg.norm(projected[:, j])
        if norm > 1e-12:
            projected[:, j] /= norm
    
    return projected


def bscores(vec, kx, ky, th, dim, nG, bidx):
    b1v=2*np.pi*np.array([1,-1/np.sqrt(3)]);b2v=2*np.pi*np.array([0,2/np.sqrt(3)])
    K=(2*b1v+b2v)/3;K1=rot(-th/2)@K;K2=rot(th/2)@K;k=np.array([kx,ky]);no=3
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

def get_kpts(G1, G2, nk=12):
    Ga=np.zeros(2);KM=(2*G1+G2)/3;MM=G1/2
    segs=[(Ga,KM,nk),(KM,MM,nk//2),(MM,Ga,nk//2)]
    kpts=[]
    for ka,kb,n in segs:
        for i in range(n): kpts.append(ka+(i/n)*(kb-ka))
    return kpts


def main():
    t0 = time.time()
    theta_deg = 9.6; th = np.radians(theta_deg)
    m_values = [2, 4, 6]
    shells = list(range(2, 8))
    
    print(f"COARSE-GRAINED BPDM: Kagome θ={theta_deg}°", flush=True)
    print(f"Refinement channel: partial trace over new G-vectors", flush=True)
    print("="*80, flush=True)
    
    results = {}
    
    for m in m_values:
        print(f"\n--- m={m} ---", flush=True)
        print(f"{'ns':>3} {'dim':>5} {'W/t':>9} {'pur':>5} "
              f"{'S_raw':>7} {'S_coarse':>9} {'Δ%':>6}", flush=True)
        
        prev_data = None  # (ev, vec, shells, nG, no, proj_vecs_at_k, widx, dim)
        prev_W = None
        mr = []
        
        for ns in shells:
            t1 = time.time()
            
            # Build at all k-points
            ev0, vec0, sh0, G1, G2, q1, dim, nG, no = build_with_shells(0, 0, th, ns)
            kpts = get_kpts(G1, G2, nk=12)
            
            nw = max(m+4, int(dim*0.4)); mid=dim//2
            lo=max(0,mid-nw//2); hi=min(dim,mid+nw//2)
            widx = np.array(list(range(lo, hi)))
            
            all_eps = []
            purities = []
            sel_vecs = []  # selected eigenvector columns at each k
            sel_idx_list = []  # global indices
            
            for k in kpts:
                ev, vec, sh, _, _, _, _, _, _ = build_with_shells(k[0], k[1], th, ns)
                bs = bscores(vec, k[0], k[1], th, dim, nG, widx)
                order = np.argsort(-np.abs(bs))[:m]
                gidx = widx[order]
                
                purities.append(float(np.mean(np.abs(bs[order]))))
                all_eps.append(sorted([ev[g] for g in gidx]))
                sel_vecs.append(vec[:, gidx])
                sel_idx_list.append(gidx)
            
            ae = np.array(all_eps)
            W = float(np.max(ae) - np.min(ae))
            dp = abs(W - prev_W)/max(W, 1e-10)*100 if prev_W else float('nan')
            
            # Cross-shell overlap
            S_raw_min = float('nan')
            S_coarse_min = float('nan')
            
            if prev_data is not None:
                p_sh, p_nG, p_no, p_selvecs, p_widx, p_dim = prev_data
                
                S_raw_list = []
                S_coarse_list = []
                
                for ik in range(len(kpts)):
                    UA = p_selvecs[ik]  # (dim_prev, m)
                    UB = sel_vecs[ik]   # (dim_curr, m)
                    
                    # Raw overlap (zero-padded as before)
                    dA, dB = UA.shape[0], UB.shape[0]; d = max(dA, dB)
                    UA_pad = np.vstack([UA, np.zeros((d-dA, m), dtype=complex)]) if dA<d else UA
                    UB_pad = np.vstack([UB, np.zeros((d-dB, m), dtype=complex)]) if dB<d else UB
                    S_raw = float(np.sum(np.abs(UA_pad.conj().T @ UB_pad)**2))
                    S_raw_list.append(S_raw)
                    
                    # Coarse-grained: project UB onto prev shell's Hilbert space
                    UB_proj = refinement_channel(
                        sel_vecs[ik],  # full vec at current shell... 
                        # Actually we need the full eigenvector matrix, not just selected columns
                        # Let me use sel_vecs directly since they're already (dim_curr, m)
                        # and project their components
                        sh0,  # current shells
                        p_sh,  # previous (smaller) shells
                        nG, p_nG, no,
                        list(range(m))  # columns 0..m-1 of the m-column matrix
                    )
                    # Wait — refinement_channel expects the full vec matrix indexed by band_idx
                    # Let me restructure: create a fake "full matrix" that's just the selected columns
                    pass
                
                # Redo properly: project each selected eigenvector from N+1 onto N's G-space
                S_coarse_list = []
                for ik in range(len(kpts)):
                    UA = p_selvecs[ik]  # (dim_prev, m) — lives in prev shell's space
                    
                    # Current shell's selected vecs: sel_vecs[ik] is (dim_curr, m)
                    # Project onto prev shell's space
                    UB_full = sel_vecs[ik]  # (dim_curr, m)
                    
                    # Manual projection: keep only components for shared G-vectors
                    sm_curr = {s:i for i,s in enumerate(sh0)}  # current shell map
                    
                    UB_proj = np.zeros((p_dim, m), dtype=complex)
                    for ig_p, g in enumerate(p_sh):
                        if g in sm_curr:
                            ig_c = sm_curr[g]
                            for orb in range(no):
                                # Layer 1
                                r_c = no * ig_c + orb
                                r_p = no * ig_p + orb
                                UB_proj[r_p, :] = UB_full[r_c, :]
                                # Layer 2
                                r_c2 = no * nG + no * ig_c + orb
                                r_p2 = no * p_nG + no * ig_p + orb
                                UB_proj[r_p2, :] = UB_full[r_c2, :]
                    
                    # Renormalise columns
                    for j in range(m):
                        norm = np.linalg.norm(UB_proj[:, j])
                        if norm > 1e-12:
                            UB_proj[:, j] /= norm
                    
                    # Now both UA and UB_proj live in dim_prev space
                    S_coarse = float(np.sum(np.abs(UA.conj().T @ UB_proj)**2))
                    S_coarse_list.append(S_coarse)
                    
                    # Raw (zero-padded)
                    dA, dB = UA.shape[0], UB_full.shape[0]; d=max(dA,dB)
                    UA_p = np.vstack([UA, np.zeros((d-dA,m),dtype=complex)]) if dA<d else UA
                    UB_p = np.vstack([UB_full, np.zeros((d-dB,m),dtype=complex)]) if dB<d else UB_full
                    S_raw_list.append(float(np.sum(np.abs(UA_p.conj().T @ UB_p)**2)))
                
                S_raw_min = min(S_raw_list)
                S_coarse_min = min(S_coarse_list)
            
            dt = time.time() - t1
            
            mr.append({
                'ns': ns, 'dim': dim, 'W': W,
                'mp': float(np.mean(purities)), 'minp': float(np.min(purities)),
                'S_raw': S_raw_min, 'S_coarse': S_coarse_min, 'dp': dp,
            })
            
            sr = f"{S_raw_min:.4f}" if not np.isnan(S_raw_min) else "—"
            sc = f"{S_coarse_min:.4f}" if not np.isnan(S_coarse_min) else "—"
            d = f"{dp:.1f}" if not np.isnan(dp) else "—"
            
            print(f"{ns:>3} {dim:>5} {W:>9.4f} {np.mean(purities):>5.3f} "
                  f"{sr:>7} {sc:>9} {d:>6}  ({dt:.0f}s)", flush=True)
            
            prev_data = (sh0, nG, no, sel_vecs, widx, dim)
            prev_W = W
        
        results[m] = mr
    
    # Summary
    print(f"\n{'='*80}", flush=True)
    print("KEY COMPARISON: Raw S_min vs Coarse-grained S_min", flush=True)
    print(f"{'m':>3} {'S_raw(last)':>12} {'S_coarse(last)':>15} {'Improvement':>12}", flush=True)
    for m in m_values:
        rs = results[m]
        sr = [r['S_raw'] for r in rs if not np.isnan(r['S_raw'])]
        sc = [r['S_coarse'] for r in rs if not np.isnan(r['S_coarse'])]
        if sr and sc:
            imp = sc[-1] / max(sr[-1], 1e-10)
            print(f"{m:>3} {sr[-1]:>12.6f} {sc[-1]:>15.6f} {imp:>12.1f}x", flush=True)
            if sc[-1] > 0.3 * m:
                print(f"  ★ S_coarse > {0.3*m:.1f} — COARSE-GRAINED CONVERGENCE!", flush=True)
    
    print(f"\nTotal: {time.time()-t0:.0f}s", flush=True)
    
    sv = {str(k): v for k, v in results.items()}
    with open('/home/darre/.openclaw/workspace/quantum-computer/numerics/bpdm_coarse.json', 'w') as f:
        json.dump(sv, f, indent=2)

if __name__ == '__main__':
    main()

"""
BPDM minimal-m closure test: Kagome θ=9.6°
Find the smallest m such that an m-band Dirac subspace is stable under basis expansion.
m = 2, 4, 6, 8, 10 × n_shells = 2..7
"""
import numpy as np
from scipy.linalg import eigh
import json, time

def rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

def kagome_H(kx, ky, t=1.0):
    a1 = np.array([1, 0]); a2 = np.array([0.5, np.sqrt(3)/2])
    k = np.array([kx, ky])
    f12 = t*(1+np.exp(-1j*np.dot(k, a1)))
    f13 = t*(1+np.exp(-1j*np.dot(k, a2)))
    f23 = t*(1+np.exp(-1j*np.dot(k, a1-a2)))
    return np.array([[0,f12,f13],[f12.conj(),0,f23],[f13.conj(),f23.conj(),0]])

def build_bilayer(kx, ky, theta, ns, t=1.0, tp=0.030):
    b1 = 2*np.pi*np.array([1,-1/np.sqrt(3)])
    b2 = 2*np.pi*np.array([0,2/np.sqrt(3)])
    K = (2*b1+b2)/3
    q1 = rot(+theta/2)@K - rot(-theta/2)@K
    q2 = rot(-2*np.pi/3)@q1; q3 = rot(+2*np.pi/3)@q1
    G1 = q1-q3; G2 = q2-q1
    offsets = [(0,0),(0,1),(-1,0)]
    T_i = tp*np.eye(3)
    shells = [(m1,m2) for m1 in range(-ns,ns+1) for m2 in range(-ns,ns+1)]
    nG = len(shells); sm = {s:i for i,s in enumerate(shells)}
    no = 3; dim = 2*no*nG
    H = np.zeros((dim,dim), dtype=complex)
    for ig,(m1,m2) in enumerate(shells):
        G = m1*G1+m2*G2
        H[no*ig:no*ig+no, no*ig:no*ig+no] = kagome_H(kx+G[0],ky+G[1],t)
        i2 = no*nG+no*ig
        H[i2:i2+no, i2:i2+no] = kagome_H(kx+G[0]+q1[0],ky+G[1]+q1[1],t)
    for ig,(m1,m2) in enumerate(shells):
        for j in range(3):
            d1,d2 = offsets[j]; tgt=(m1+d1,m2+d2)
            if tgt in sm:
                ig2 = sm[tgt]; i1=no*ig; i2=no*nG+no*ig2
                H[i1:i1+no,i2:i2+no] += T_i; H[i2:i2+no,i1:i1+no] += T_i
    ev, vec = eigh(H)
    return ev, vec, G1, G2, q1, dim, nG

def build_B(kx, ky, theta, dim, nG):
    b1 = 2*np.pi*np.array([1,-1/np.sqrt(3)]); b2 = 2*np.pi*np.array([0,2/np.sqrt(3)])
    K = (2*b1+b2)/3
    K1 = rot(-theta/2)@K; K2 = rot(+theta/2)@K
    sx3 = np.zeros((3,3),dtype=complex); sx3[0,1]=1; sx3[1,0]=1
    sy3 = np.zeros((3,3),dtype=complex); sy3[0,1]=-1j; sy3[1,0]=1j
    B = np.zeros((dim,dim),dtype=complex); k=np.array([kx,ky]); no=3
    for layer in range(2):
        K_ell = K1 if layer==0 else K2
        R_ell = rot(-theta/2) if layer==0 else rot(+theta/2)
        dk = R_ell@(k-K_ell); norm=np.linalg.norm(dk)
        dhat = dk/norm if norm>1e-12 else np.array([1.0,0.0])
        dsigma = dhat[0]*sx3 + dhat[1]*sy3
        for ig in range(nG):
            idx = no*ig if layer==0 else no*nG+no*ig
            B[idx:idx+no, idx:idx+no] = dsigma
    return B

def run_bpdm(theta_deg, ns, m_select, nk=40):
    """Run BPDM with m_select bands. Return W, purity, projector vecs."""
    theta = np.radians(theta_deg)
    ev0, vec0, G1, G2, q1, dim, nG = build_bilayer(0, 0, theta, ns)
    
    Gamma = np.zeros(2); K_M = (2*G1+G2)/3; M_M = G1/2
    segs = [(Gamma,K_M,nk),(K_M,M_M,nk//2),(M_M,Gamma,nk//2)]
    kpts = []
    for ka,kb,n in segs:
        for i in range(n):
            kpts.append(ka+(i/n)*(kb-ka))
    
    n_window = max(m_select+4, int(dim*0.4))
    mid = dim//2
    lo = max(0, mid-n_window//2); hi = min(dim, mid+n_window//2)
    window_idx = list(range(lo, hi))
    
    all_eps = []  # all selected eigenvalues per k
    purities = []
    proj_vecs = []  # (dim, m_select) per k
    
    for k in kpts:
        ev, vec, _, _, _, _, _ = build_bilayer(k[0], k[1], theta, ns)
        B = build_B(k[0], k[1], theta, dim, nG)
        
        scores = []
        for n in window_idx:
            psi = vec[:, n]
            bn = np.real(psi.conj() @ B @ psi)
            scores.append((abs(bn), n))
        scores.sort(reverse=True)
        
        sel = scores[:m_select]
        sel_idx = [s[1] for s in sel]
        purity = np.mean([s[0] for s in sel])
        purities.append(purity)
        
        eps = sorted([ev[n] for n in sel_idx])
        all_eps.append(eps)
        
        U = vec[:, sel_idx]
        proj_vecs.append(U)
    
    all_eps = np.array(all_eps)  # (nk, m_select)
    W = np.max(all_eps) - np.min(all_eps)
    
    return {
        'W': float(W),
        'mean_purity': float(np.mean(purities)),
        'min_purity': float(np.min(purities)),
        'proj_vecs': proj_vecs,
        'dim': dim,
    }

def cross_shell_overlap(proj_A, proj_B, m):
    """S(k) = Tr(P_A P_B) for m-dim subspace. Pad to same dim."""
    nk = min(len(proj_A), len(proj_B))
    S_list = []
    for ik in range(nk):
        UA, UB = proj_A[ik], proj_B[ik]
        dA, dB = UA.shape[0], UB.shape[0]
        d = max(dA, dB)
        if dA < d: UA = np.vstack([UA, np.zeros((d-dA, m), dtype=complex)])
        if dB < d: UB = np.vstack([UB, np.zeros((d-dB, m), dtype=complex)])
        overlap = UA.conj().T @ UB
        S_list.append(float(np.sum(np.abs(overlap)**2)))
    return np.array(S_list)

def main():
    t0 = time.time()
    theta = 9.6
    m_values = [2, 4, 6, 8, 10]
    shell_range = range(2, 8)  # 2..7
    
    print(f"BPDM MINIMAL-m CLOSURE TEST: Kagome θ={theta}°")
    print(f"m = {m_values}, n_shells = {list(shell_range)}")
    print("="*80)
    
    all_results = {}
    
    for m in m_values:
        print(f"\n--- m = {m} ---")
        print(f"{'n_sh':>4} {'dim':>5} {'W/t':>10} {'pur':>6} {'min_p':>6} {'S_mean':>7} {'S_min':>6} {'Δ%':>6}")
        
        prev_proj = None
        prev_W = None
        m_results = []
        
        for ns in shell_range:
            t1 = time.time()
            r = run_bpdm(theta, ns, m, nk=40)
            dt = time.time() - t1
            
            if prev_proj is not None:
                S = cross_shell_overlap(prev_proj, r['proj_vecs'], m)
                S_mean, S_min = float(np.mean(S)), float(np.min(S))
            else:
                S_mean = S_min = float('nan')
            
            dpct = abs(r['W']-prev_W)/max(r['W'],1e-10)*100 if prev_W else float('nan')
            
            m_results.append({
                'n_shells': ns, 'dim': r['dim'], 'W': r['W'],
                'mean_purity': r['mean_purity'], 'min_purity': r['min_purity'],
                'S_mean': S_mean, 'S_min': S_min, 'delta_pct': dpct, 'time': dt
            })
            
            sm = f"{S_mean:.3f}" if not np.isnan(S_mean) else "—"
            sn = f"{S_min:.3f}" if not np.isnan(S_min) else "—"
            dp = f"{dpct:.1f}" if not np.isnan(dpct) else "—"
            
            print(f"{ns:>4} {r['dim']:>5} {r['W']:>10.5f} {r['mean_purity']:>6.3f} "
                  f"{r['min_purity']:>6.3f} {sm:>7} {sn:>6} {dp:>6}  ({dt:.1f}s)")
            
            prev_proj = r['proj_vecs']
            prev_W = r['W']
            
            if dt > 120:
                print(f"  (stopping m={m})")
                break
        
        all_results[m] = m_results
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY: Minimal m for closure")
    print(f"{'m':>3} {'W range':>14} {'W ratio':>8} {'S_min(last)':>12} {'Converged?':>12}")
    print("-"*55)
    
    for m in m_values:
        if m not in all_results or not all_results[m]:
            continue
        rs = all_results[m]
        ws = [r['W'] for r in rs]
        s_mins = [r['S_min'] for r in rs if not np.isnan(r['S_min'])]
        ratio = max(ws)/max(min(ws),1e-10)
        last_s = s_mins[-1] if s_mins else float('nan')
        
        # Check: S_min > 0.7*m and last 2 W changes < 15%
        changes = [rs[i]['delta_pct'] for i in range(1,len(rs)) if not np.isnan(rs[i]['delta_pct'])]
        converged = (len(changes)>=2 and all(c<15 for c in changes[-2:]) 
                     and last_s > 0.5*m)
        
        s_str = f"{last_s:.3f}" if not np.isnan(last_s) else "—"
        c_str = "✓ YES" if converged else "✗ No"
        print(f"{m:>3} {min(ws):.4f}—{max(ws):.4f} {ratio:>8.2f}x {s_str:>12} {c_str:>12}")
    
    # Find minimal converging m
    for m in m_values:
        if m not in all_results: continue
        rs = all_results[m]
        changes = [rs[i]['delta_pct'] for i in range(1,len(rs)) if not np.isnan(rs[i]['delta_pct'])]
        s_mins = [r['S_min'] for r in rs if not np.isnan(r['S_min'])]
        if (len(changes)>=2 and all(c<15 for c in changes[-2:]) 
            and s_mins and s_mins[-1] > 0.5*m):
            print(f"\n★ MINIMAL CLOSURE at m={m}")
            break
    else:
        print(f"\n⚠ NO CLOSURE up to m={max(m_values)}")
    
    print(f"\nTotal: {time.time()-t0:.0f}s")
    
    # Save (exclude proj_vecs)
    save = {}
    for m, rs in all_results.items():
        save[str(m)] = rs
    with open('/home/darre/.openclaw/workspace/quantum-computer/numerics/bpdm_minimal_m.json','w') as f:
        json.dump(save, f, indent=2)

if __name__ == '__main__':
    main()

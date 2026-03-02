"""
BPDM — Boundary-Projected Dirac Manifold for Kagome.
Defines Dirac sector by spinor boundary alignment, not energy rank.
Tests convergence of W_BPDM vs n_shells at θ=9.6°.
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
    """Returns (evals, evecs, G1, G2, q1, dim, n_G, shells)"""
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


def build_boundary_operator(kx, ky, theta, dim, nG):
    """
    B(k) = Σ_ℓ Π_ℓ (d̂_ℓ(k)·σ) Π_ℓ
    
    For Kagome (3 sublattices), σ acts on a 2D subspace. We extend to 3 orbitals
    by embedding σ_x, σ_y in the upper-left 2×2 block (sublattices A,B) and 
    treating C as spectator. This selects states with A-B Dirac character.
    
    Alternative: use the full 3×3 Gell-Mann matrices. But the 2×2 embedding
    is cleaner since the Kagome Dirac cone lives in the A-B sector.
    """
    b1 = 2*np.pi*np.array([1,-1/np.sqrt(3)])
    b2 = 2*np.pi*np.array([0,2/np.sqrt(3)])
    K = (2*b1+b2)/3
    
    # Layer Dirac points
    K1 = rot(-theta/2) @ K  # layer 1
    K2 = rot(+theta/2) @ K  # layer 2
    
    no = 3  # orbitals per layer per shell
    
    # σ_x, σ_y embedded in 3×3 (act on sublattices 0,1; sublattice 2 = spectator)
    sx3 = np.zeros((3,3), dtype=complex)
    sx3[0,1] = 1; sx3[1,0] = 1
    sy3 = np.zeros((3,3), dtype=complex)
    sy3[0,1] = -1j; sy3[1,0] = 1j
    
    B = np.zeros((dim, dim), dtype=complex)
    k = np.array([kx, ky])
    
    for layer in range(2):
        K_ell = K1 if layer == 0 else K2
        R_ell = rot(-theta/2) if layer == 0 else rot(+theta/2)
        
        # Local Dirac direction
        dk = R_ell @ (k - K_ell)
        norm = np.linalg.norm(dk)
        if norm < 1e-12:
            dhat = np.array([1.0, 0.0])  # arbitrary at Dirac point
        else:
            dhat = dk / norm
        
        # d̂·σ in 3×3
        dsigma = dhat[0] * sx3 + dhat[1] * sy3  # (3,3)
        
        # Place in each shell block for this layer
        for ig in range(nG):
            if layer == 0:
                idx = no * ig
            else:
                idx = no * nG + no * ig
            B[idx:idx+no, idx:idx+no] = dsigma
    
    return B


def bpdm_bandwidth(theta_deg, ns, nk=50, n_select=2, window_frac=0.3):
    """
    Compute W_BPDM at given theta and n_shells.
    
    Returns: W_BPDM, mean_purity, projector_vecs at each k for overlap computation.
    """
    theta = np.radians(theta_deg)
    
    # Get geometry from Γ point
    ev0, vec0, G1, G2, q1, dim, nG = build_bilayer(0, 0, theta, ns)
    
    # k-path
    Gamma = np.zeros(2); K_M = (2*G1+G2)/3; M_M = G1/2
    segs = [(Gamma, K_M, nk), (K_M, M_M, nk//2), (M_M, Gamma, nk//2)]
    
    kpts = []
    for ka, kb, npts in segs:
        for i in range(npts):
            kpts.append(ka + (i/npts)*(kb-ka))
    nk_tot = len(kpts)
    
    # Central energy window: middle window_frac of bands
    n_window = max(n_select + 2, int(dim * window_frac))
    
    eps1_list = []  # lower projected eigenvalue at each k
    eps2_list = []  # upper projected eigenvalue at each k
    purity_list = []
    proj_list = []  # projector vectors for S(k) computation
    
    for ik, k in enumerate(kpts):
        kx, ky = k
        ev, vec, _, _, _, _, _ = build_bilayer(kx, ky, theta, ns)
        B = build_boundary_operator(kx, ky, theta, dim, nG)
        
        # Select central bands (by energy, broad window)
        mid = dim // 2
        lo = max(0, mid - n_window // 2)
        hi = min(dim, mid + n_window // 2)
        window_idx = list(range(lo, hi))
        
        # Compute boundary scores for windowed bands
        scores = []
        for n in window_idx:
            psi = vec[:, n]
            bn = np.real(psi.conj() @ B @ psi)
            scores.append((abs(bn), bn, n))
        
        # Select top n_select by |b_n|
        scores.sort(reverse=True)
        selected = scores[:n_select]
        
        # Purity
        purity = np.mean([s[0] for s in selected])
        purity_list.append(purity)
        
        # Build projector subspace
        sel_idx = [s[2] for s in selected]
        U = vec[:, sel_idx]  # (dim, n_select)
        proj_list.append(U)
        
        # Projected Hamiltonian
        H_full = np.diag(ev)  # already diagonal in eigenbasis
        # But we need H in the original basis... actually ev,vec gives us:
        # H_eff = U† H U where H|ψ_n⟩ = E_n|ψ_n⟩
        # Since U columns are eigenvectors, H_eff = diag(E_selected)
        # That's trivially the selected eigenvalues.
        # BUT: that's only true if n_select columns are exact eigenvectors.
        # They ARE eigenvectors of H(k), so H_eff IS diagonal.
        eps = sorted([ev[n] for n in sel_idx])
        eps1_list.append(eps[0])
        eps2_list.append(eps[-1])
    
    eps1 = np.array(eps1_list)
    eps2 = np.array(eps2_list)
    
    W_bpdm = np.max(eps2) - np.min(eps1)
    mean_purity = np.mean(purity_list)
    min_purity = np.min(purity_list)
    
    return {
        'W_bpdm': float(W_bpdm),
        'mean_purity': float(mean_purity),
        'min_purity': float(min_purity),
        'proj_list': proj_list,
        'dim': dim,
        'nk': nk_tot,
    }


def subspace_overlap(proj_A, proj_B):
    """
    Compute S(k) = Tr(P_A P_B) for each k.
    proj_A, proj_B are lists of (dim_A, n_select) and (dim_B, n_select) arrays.
    Since dims differ between shells, we can't directly compare.
    
    WORKAROUND: Embed smaller into larger. The first dim_small components
    of the larger basis correspond to the same shells (inner shells are shared).
    Pad smaller vectors with zeros.
    """
    nk = min(len(proj_A), len(proj_B))
    S_list = []
    for ik in range(nk):
        UA = proj_A[ik]
        UB = proj_B[ik]
        dA, nA = UA.shape
        dB, nB = UB.shape
        
        # Pad to same dimension
        d_max = max(dA, dB)
        if dA < d_max:
            UA = np.vstack([UA, np.zeros((d_max - dA, nA), dtype=complex)])
        if dB < d_max:
            UB = np.vstack([UB, np.zeros((d_max - dB, nB), dtype=complex)])
        
        # S = Tr(P_A P_B) = Tr(UA UA† UB UB†) = ||UA† UB||_F^2
        overlap = UA.conj().T @ UB  # (nA, nB)
        S = np.sum(np.abs(overlap)**2)
        S_list.append(S)
    
    return np.array(S_list)


def main():
    t0 = time.time()
    theta = 9.6
    
    print(f"BPDM CONVERGENCE: Kagome θ={theta}°")
    print("="*70)
    print(f"{'n_sh':>4} {'dim':>5} {'W_BPDM/t':>10} {'W(meV)':>8} {'purity':>8} {'min_pur':>8} {'S_mean':>7} {'S_min':>6} {'Δ%':>6}")
    print("-"*70)
    
    results = []
    prev_proj = None
    prev_W = None
    
    for ns in range(2, 9):
        t1 = time.time()
        r = bpdm_bandwidth(theta, ns, nk=50, n_select=2, window_frac=0.3)
        dt = time.time() - t1
        
        # Subspace overlap with previous shell
        if prev_proj is not None:
            S = subspace_overlap(prev_proj, r['proj_list'])
            S_mean = float(np.mean(S))
            S_min = float(np.min(S))
        else:
            S_mean = S_min = float('nan')
        
        delta_pct = abs(r['W_bpdm'] - prev_W) / max(r['W_bpdm'], 1e-10) * 100 if prev_W else float('nan')
        
        results.append({
            'n_shells': ns, 'dim': r['dim'],
            'W_bpdm': r['W_bpdm'], 'W_meV': r['W_bpdm'] * 100,
            'mean_purity': r['mean_purity'], 'min_purity': r['min_purity'],
            'S_mean': S_mean, 'S_min': S_min,
            'delta_pct': delta_pct, 'time': dt,
        })
        
        dp = f"{delta_pct:.1f}" if not np.isnan(delta_pct) else "—"
        sm = f"{S_mean:.3f}" if not np.isnan(S_mean) else "—"
        sn = f"{S_min:.3f}" if not np.isnan(S_min) else "—"
        
        print(f"{ns:>4} {r['dim']:>5} {r['W_bpdm']:>10.6f} {r['W_bpdm']*100:>8.2f} "
              f"{r['mean_purity']:>8.4f} {r['min_purity']:>8.4f} {sm:>7} {sn:>6} {dp:>6}  ({dt:.1f}s)")
        
        prev_proj = r['proj_list']
        prev_W = r['W_bpdm']
        
        if dt > 150:
            print(f"  (stopping: {dt:.0f}s)")
            break
    
    # Verdict
    print(f"\n{'='*70}")
    ws = [r['W_bpdm'] for r in results]
    changes = [results[i]['delta_pct'] for i in range(1, len(results)) if not np.isnan(results[i]['delta_pct'])]
    purities = [r['min_purity'] for r in results]
    s_mins = [r['S_min'] for r in results[1:] if not np.isnan(r['S_min'])]
    
    print(f"W_BPDM range: {min(ws):.6f} — {max(ws):.6f} (ratio {max(ws)/max(min(ws),1e-10):.2f}x)")
    print(f"Changes: {[f'{c:.1f}%' for c in changes]}")
    print(f"Min purity (last 3): {purities[-3:]}")
    if s_mins:
        print(f"S_min (last 3): {s_mins[-3:]}")
    
    # Convergence check
    if len(changes) >= 2:
        if all(c < 10 for c in changes[-2:]):
            print("\n✓ CONVERGING — last 2 changes < 10%")
        elif all(c < 20 for c in changes[-2:]):
            print("\n◐ APPROACHING — last 2 changes < 20%")
        else:
            print("\n⚠ NOT CONVERGED")
    
    # Purity check
    if purities[-1] < 0.3:
        print("⚠ LOW PURITY — Dirac sector poorly defined at some k-points")
    elif purities[-1] < 0.5:
        print("◐ MODERATE PURITY — some k-points have weak Dirac character")
    else:
        print("✓ GOOD PURITY — Dirac sector well-defined")
    
    # S check
    if s_mins and s_mins[-1] > 1.8:
        print("✓ SUBSPACE STABLE — S_min > 1.8")
    elif s_mins and s_mins[-1] > 1.5:
        print("◐ SUBSPACE MODERATE — S_min > 1.5")
    elif s_mins:
        print("⚠ SUBSPACE UNSTABLE — S_min < 1.5")
    
    print(f"\nTotal: {time.time()-t0:.0f}s")
    
    # Save
    save_results = [{k:v for k,v in r.items()} for r in results]
    with open('/home/darre/.openclaw/workspace/quantum-computer/numerics/bpdm_kagome.json', 'w') as f:
        json.dump(save_results, f, indent=2)


if __name__ == '__main__':
    main()

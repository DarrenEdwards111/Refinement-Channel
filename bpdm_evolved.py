"""
Evolved BPDM: learn an invariant projector via evolutionary optimization.

Score function: s_n(k) = w1*|b_n| + w2*(1-|E_n|/E_max) + w3*layer_pol + w4*ipr_inv
Select top-m by s_n(k).

Fitness = -λ_S * S_overlap - λ_P * purity + λ_R * roughness
Evaluated across shell pairs (3→4, 4→5, 5→6) to prevent overfitting.
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

def get_kpts(G1,G2):
    Ga=np.zeros(2);KM=(2*G1+G2)/3;MM=G1/2
    return [Ga, KM, MM, 0.5*KM, 0.5*(KM+MM), 0.5*MM, 0.25*KM, 0.75*KM]

# ============================================================
# FEATURE EXTRACTION
# ============================================================

def compute_features(ev, vec, kx, ky, th, dim, nG, band_idx):
    """
    For each band in band_idx, compute 4 features:
    f1: |b_n| (Dirac boundary alignment)
    f2: energy proximity to neutrality: 1 - |E_n|/E_max
    f3: layer polarization: |⟨ψ|Π_1|ψ⟩ - ⟨ψ|Π_2|ψ⟩| (low = balanced = good)
    f4: inverse IPR proxy: 1/Σ|ψ_i|^4 / dim (extended = high = good)
    """
    nb = len(band_idx)
    feats = np.zeros((nb, 4))
    no = 3
    
    # f1: boundary scores
    b1=2*np.pi*np.array([1,-1/np.sqrt(3)]);b2=2*np.pi*np.array([0,2/np.sqrt(3)])
    K=(2*b1+b2)/3;K1=rot(-th/2)@K;K2=rot(th/2)@K;k=np.array([kx,ky])
    for l in range(2):
        Ke=K1 if l==0 else K2;Re=rot(-th/2) if l==0 else rot(th/2)
        dk=Re@(k-Ke);n=np.linalg.norm(dk);dh=dk/n if n>1e-12 else np.array([1.,0.])
        dm=dh[0]-1j*dh[1];dp=dh[0]+1j*dh[1]
        for ig in range(nG):
            idx=no*ig if l==0 else no*nG+no*ig
            for bi,bn in enumerate(band_idx):
                p0=vec[idx,bn];p1=vec[idx+1,bn]
                feats[bi,0]+=np.real(p0.conj()*dm*p1+p1.conj()*dp*p0)
    feats[:,0] = np.abs(feats[:,0])
    
    # f2: energy proximity
    E_max = max(abs(ev[band_idx[0]]), abs(ev[band_idx[-1]]), 1e-10)
    feats[:,1] = 1.0 - np.abs(ev[band_idx]) / E_max
    
    # f3: layer balance (1 = balanced, 0 = fully polarized)
    half = no * nG
    for bi, bn in enumerate(band_idx):
        w1 = np.sum(np.abs(vec[:half, bn])**2)
        w2 = np.sum(np.abs(vec[half:, bn])**2)
        feats[bi,2] = 1.0 - abs(w1 - w2)  # 1=balanced
    
    # f4: inverse IPR (extensiveness)
    for bi, bn in enumerate(band_idx):
        ipr = np.sum(np.abs(vec[:, bn])**4)
        feats[bi,3] = min(1.0 / (ipr * dim), 1.0)  # normalized, extended→1
    
    return feats


def select_bands(feats, weights, m):
    """Score each band, return indices of top-m."""
    scores = feats @ weights  # (nb,)
    return np.argsort(-scores)[:m]


# ============================================================
# FITNESS EVALUATION
# ============================================================

def evaluate(weights, m, theta_deg, shell_pairs, kpts_cache, feats_cache):
    """
    Evaluate fitness of weight vector across shell pairs.
    Returns: mean_S_min, mean_purity, roughness, W_stability
    """
    th = np.radians(theta_deg)
    all_S_min = []
    all_purity = []
    all_W = []
    
    for ns_a, ns_b in shell_pairs:
        # For each shell, select bands at each k-point
        pvecs_a, pvecs_b = [], []
        eps_a, eps_b = [], []
        purs_a = []
        
        for ik in range(len(kpts_cache[ns_a])):
            ev_a, vec_a, feats_a = kpts_cache[ns_a][ik]
            sel_a = select_bands(feats_a, weights, m)
            sel_global_a = feats_cache[ns_a]['widx'][sel_a]
            pvecs_a.append(vec_a[:, sel_global_a])
            eps_a.append(sorted([ev_a[n] for n in sel_global_a]))
            purs_a.append(float(np.mean(feats_a[sel_a, 0])))  # boundary score
            
            ev_b, vec_b, feats_b = kpts_cache[ns_b][ik]
            sel_b = select_bands(feats_b, weights, m)
            sel_global_b = feats_cache[ns_b]['widx'][sel_b]
            pvecs_b.append(vec_b[:, sel_global_b])
            eps_b.append(sorted([ev_b[n] for n in sel_global_b]))
        
        # Subspace overlap
        S_vals = []
        for ik in range(len(pvecs_a)):
            UA, UB = pvecs_a[ik], pvecs_b[ik]
            dA, dB = UA.shape[0], UB.shape[0]; d = max(dA, dB)
            if dA < d: UA = np.vstack([UA, np.zeros((d-dA, m), dtype=complex)])
            if dB < d: UB = np.vstack([UB, np.zeros((d-dB, m), dtype=complex)])
            S_vals.append(float(np.sum(np.abs(UA.conj().T @ UB)**2)))
        
        all_S_min.append(min(S_vals))
        all_purity.append(np.mean(purs_a))
        
        ea = np.array(eps_a); W_a = np.max(ea) - np.min(ea)
        eb = np.array(eps_b); W_b = np.max(eb) - np.min(eb)
        all_W.append((W_a, W_b))
    
    mean_S_min = np.mean(all_S_min)
    mean_purity = np.mean(all_purity)
    
    # W stability: max % change across pairs
    W_changes = [abs(wa-wb)/max(wa,1e-10)*100 for wa,wb in all_W]
    W_stability = np.mean(W_changes)
    
    # Roughness: not computed here (would need adjacent k comparison)
    roughness = 0
    
    return mean_S_min, mean_purity, roughness, W_stability


def fitness(weights, m, theta_deg, shell_pairs, kpts_cache, feats_cache):
    """Lower is better."""
    S_min, purity, rough, W_stab = evaluate(weights, m, theta_deg, shell_pairs, kpts_cache, feats_cache)
    # Maximize S_min and purity, minimize W instability
    return -2.0 * S_min / m - 1.0 * purity + 0.01 * W_stab


# ============================================================
# PRECOMPUTE
# ============================================================

def precompute(theta_deg, shells, m):
    """Precompute eigensystems and features for all shells and k-points."""
    th = np.radians(theta_deg)
    kpts_cache = {}  # ns -> list of (ev, vec, feats)
    feats_cache = {}  # ns -> {'widx': array}
    
    for ns in shells:
        _, _, G1, G2, _, dim, nG = build(0, 0, th, ns)
        kpts = get_kpts(G1, G2)
        
        nw = max(m + 4, int(dim * 0.4)); mid = dim // 2
        lo = max(0, mid - nw // 2); hi = min(dim, mid + nw // 2)
        widx = np.array(list(range(lo, hi)))
        
        cache = []
        for k in kpts:
            ev, vec, _, _, _, _, _ = build(k[0], k[1], th, ns)
            feats = compute_features(ev, vec, k[0], k[1], th, dim, nG, widx)
            cache.append((ev, vec, feats))
        
        kpts_cache[ns] = cache
        feats_cache[ns] = {'widx': widx}
        print(f"  Precomputed ns={ns} (dim={dim})", flush=True)
    
    return kpts_cache, feats_cache


# ============================================================
# EVOLUTIONARY ALGORITHM
# ============================================================

def evolve(m, theta_deg, shell_pairs, kpts_cache, feats_cache,
           pop_size=40, n_gen=60, n_weights=4):
    """Simple (μ+λ) evolution strategy."""
    # Initialize population: random weights in [0,1]
    pop = np.random.rand(pop_size, n_weights)
    # Normalize rows
    pop = pop / (np.linalg.norm(pop, axis=1, keepdims=True) + 1e-10)
    
    best_fit = float('inf')
    best_w = None
    
    for gen in range(n_gen):
        # Evaluate
        fits = np.array([fitness(w, m, theta_deg, shell_pairs, kpts_cache, feats_cache) for w in pop])
        
        # Track best
        idx_best = np.argmin(fits)
        if fits[idx_best] < best_fit:
            best_fit = fits[idx_best]
            best_w = pop[idx_best].copy()
        
        if gen % 10 == 0:
            S, P, R, W = evaluate(best_w, m, theta_deg, shell_pairs, kpts_cache, feats_cache)
            print(f"    gen={gen:>3}: fit={best_fit:.4f} S_min={S:.4f} pur={P:.3f} Wstab={W:.1f}%  w={best_w.round(3)}", flush=True)
        
        # Selection: top half
        order = np.argsort(fits)
        parents = pop[order[:pop_size//2]]
        
        # Offspring: mutation + crossover
        children = []
        for _ in range(pop_size):
            p1 = parents[np.random.randint(len(parents))]
            p2 = parents[np.random.randint(len(parents))]
            # Crossover
            mask = np.random.rand(n_weights) > 0.5
            child = np.where(mask, p1, p2)
            # Mutation
            child += np.random.randn(n_weights) * 0.1
            child = np.clip(child, -2, 2)
            children.append(child)
        
        pop = np.array(children)
    
    return best_w, best_fit


# ============================================================
# MAIN
# ============================================================

def main():
    t0 = time.time()
    theta = 9.6
    m_values = [4, 6, 8]
    shells = [3, 4, 5, 6]
    shell_pairs = [(3,4), (4,5), (5,6)]  # evaluate across 3 pairs
    
    print(f"EVOLVED BPDM: Kagome θ={theta}°", flush=True)
    print(f"Shell pairs: {shell_pairs}", flush=True)
    print("="*80, flush=True)
    
    results = {}
    
    for m in m_values:
        print(f"\n--- m={m}: Precomputing ---", flush=True)
        kc, fc = precompute(theta, shells, m)
        
        print(f"--- m={m}: Evolving (40 pop × 60 gen) ---", flush=True)
        best_w, best_fit = evolve(m, theta, shell_pairs, kc, fc, pop_size=40, n_gen=60)
        
        # Final evaluation with best weights
        S, P, R, W = evaluate(best_w, m, theta, shell_pairs, kc, fc)
        
        print(f"\n  RESULT m={m}: S_min={S:.4f}/{m} pur={P:.3f} W_stab={W:.1f}% w={best_w.round(4)}", flush=True)
        
        # Compare to baseline (BPDM = [1,0,0,0])
        S0, P0, R0, W0 = evaluate(np.array([1,0,0,0]), m, theta, shell_pairs, kc, fc)
        print(f"  BASELINE:   S_min={S0:.4f}/{m} pur={P0:.3f} W_stab={W0:.1f}%", flush=True)
        print(f"  IMPROVEMENT: S_min {S/max(S0,1e-10):.1f}x, purity {P/max(P0,1e-10):.2f}x", flush=True)
        
        results[m] = {
            'best_w': best_w.tolist(),
            'S_min': S, 'purity': P, 'W_stability': W,
            'baseline_S': S0, 'baseline_P': P0, 'baseline_W': W0,
        }
        
        # Convergence check
        if S > 0.3 * m:
            print(f"  ★ S_min > 0.3m — SUBSPACE CONVERGING!", flush=True)
        else:
            print(f"  ✗ S_min < 0.3m — still not stable", flush=True)
    
    print(f"\n{'='*80}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'m':>3} {'S_min':>7} {'S/m':>6} {'pur':>5} {'W%':>5} {'baseline_S':>10} {'improv':>7}", flush=True)
    for m in m_values:
        r = results[m]
        imp = r['S_min'] / max(r['baseline_S'], 1e-10)
        print(f"{m:>3} {r['S_min']:>7.4f} {r['S_min']/m:>6.3f} {r['purity']:>5.3f} {r['W_stability']:>5.1f} {r['baseline_S']:>10.4f} {imp:>7.1f}x", flush=True)
    
    print(f"\nTotal: {time.time()-t0:.0f}s", flush=True)
    
    with open('/home/darre/.openclaw/workspace/quantum-computer/numerics/bpdm_evolved.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()

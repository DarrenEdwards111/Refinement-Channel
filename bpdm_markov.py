"""
Markov BPDM: Viterbi-optimal projector path across k-points.
Instead of selecting top-m independently at each k, find the globally
smoothest path of m-dimensional subspaces via dynamic programming.

F({P_i}) = Σ_i E_local(P_i; k_i) + λ Σ_i d(P_i, P_{i+1})
where d(P,Q) = m - Tr(PQ) and E_local = -α·Tr(P|B|) + β·|Ē_P|
"""
import numpy as np, json, time
from scipy.linalg import eigh
from itertools import combinations

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

def bscores_vec(vec,kx,ky,th,dim,nG,bidx):
    """Vectorized boundary scores."""
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

def get_kpath(G1, G2, nk=12):
    Ga=np.zeros(2);KM=(2*G1+G2)/3;MM=G1/2
    segs=[(Ga,KM,nk),(KM,MM,nk//2),(MM,Ga,nk//2)]
    kpts=[]
    for ka,kb,n in segs:
        for i in range(n): kpts.append(ka+(i/n)*(kb-ka))
    return kpts

# ============================================================
# CANDIDATE GENERATION
# ============================================================

def generate_candidates(ev, vec, bscores, widx, m, L=8, max_cands=30):
    """
    Generate candidate m-tuples from top-L boundary-scored bands.
    Return list of (tuple_of_global_indices, projector_vecs).
    """
    order = np.argsort(-np.abs(bscores))[:L]
    global_idx = [widx[i] for i in order]
    
    # All m-combinations of L (capped)
    combos = list(combinations(range(L), m))
    if len(combos) > max_cands:
        # Sample + always include top-m
        combos = [tuple(range(m))] + [combos[i] for i in 
                  np.random.choice(len(combos), max_cands-1, replace=False)]
    
    candidates = []
    for combo in combos:
        gidx = tuple(global_idx[c] for c in combo)
        U = vec[:, list(gidx)]  # (dim, m)
        candidates.append((gidx, U))
    
    return candidates

# ============================================================
# COST FUNCTIONS
# ============================================================

def local_cost(ev, bscores, widx, gidx, alpha=1.0, beta=0.3):
    """E_local = -α·mean(|b|) + β·|mean(E)|"""
    local_idx = [list(widx).index(g) for g in gidx]
    purity = np.mean(np.abs(bscores[local_idx]))
    mean_E = np.mean([ev[g] for g in gidx])
    return -alpha * purity + beta * abs(mean_E)

def transition_cost(U_a, U_b, m):
    """d(P,Q) = m - Tr(PQ) = m - ||U_a† U_b||_F²"""
    overlap = np.sum(np.abs(U_a.conj().T @ U_b)**2)
    return m - overlap

# ============================================================
# VITERBI
# ============================================================

def viterbi(kdata, m, lam=2.0, alpha=1.0, beta=0.3):
    """
    Dynamic programming over k-path.
    kdata[i] = (ev, vec, bscores, widx, candidates)
    candidates[j] = (gidx, U)
    
    Returns: optimal sequence of candidate indices, total cost.
    """
    nk = len(kdata)
    
    # Forward pass
    # cost[i][j] = min cost to reach candidate j at k_i
    # back[i][j] = which candidate at k_{i-1}
    n_cands = [len(kdata[i][4]) for i in range(nk)]
    
    cost = [np.full(nc, np.inf) for nc in n_cands]
    back = [np.zeros(nc, dtype=int) for nc in n_cands]
    
    # Initialize: k=0
    for j, (gidx, U) in enumerate(kdata[0][4]):
        cost[0][j] = local_cost(kdata[0][0], kdata[0][2], kdata[0][3], gidx, alpha, beta)
    
    # Forward
    for i in range(1, nk):
        ev_i, vec_i, bs_i, widx_i, cands_i = kdata[i]
        _, _, _, _, cands_prev = kdata[i-1]
        
        for j, (gidx_j, U_j) in enumerate(cands_i):
            lc = local_cost(ev_i, bs_i, widx_i, gidx_j, alpha, beta)
            best_prev = np.inf
            best_k = 0
            for k, (gidx_k, U_k) in enumerate(cands_prev):
                tc = lam * transition_cost(U_k, U_j, m)
                total = cost[i-1][k] + tc
                if total < best_prev:
                    best_prev = total
                    best_k = k
            cost[i][j] = best_prev + lc
            back[i][j] = best_k
    
    # Backtrace
    path = [0] * nk
    path[-1] = int(np.argmin(cost[-1]))
    for i in range(nk-2, -1, -1):
        path[i] = back[i+1][path[i+1]]
    
    total_cost = cost[-1][path[-1]]
    return path, total_cost


# ============================================================
# MAIN
# ============================================================

def run_markov(theta_deg, ns, m, lam=2.0, nk=12, L=8, max_cands=30):
    """Run Markov BPDM for one shell size. Return W, purity, projector vecs."""
    th = np.radians(theta_deg)
    _, _, G1, G2, _, dim, nG = build(0, 0, th, ns)
    kpts = get_kpath(G1, G2, nk)
    
    nw = max(m+4, int(dim*0.4)); mid=dim//2
    lo=max(0,mid-nw//2);hi=min(dim,mid+nw//2);widx=np.array(list(range(lo,hi)))
    
    # Precompute per k-point
    kdata = []
    for k in kpts:
        ev, vec, _, _, _, _, _ = build(k[0], k[1], th, ns)
        bs = bscores_vec(vec, k[0], k[1], th, dim, nG, widx)
        cands = generate_candidates(ev, vec, bs, widx, m, L, max_cands)
        kdata.append((ev, vec, bs, widx, cands))
    
    # Viterbi
    path, total_cost = viterbi(kdata, m, lam)
    
    # Extract results along optimal path
    all_eps = []
    purities = []
    proj_vecs = []
    
    for i, (ev, vec, bs, wi, cands) in enumerate(kdata):
        gidx, U = cands[path[i]]
        local_idx = [list(wi).index(g) for g in gidx]
        purity = np.mean(np.abs(bs[local_idx]))
        purities.append(purity)
        eps = sorted([ev[g] for g in gidx])
        all_eps.append(eps)
        proj_vecs.append(U)
    
    ae = np.array(all_eps)
    W = float(np.max(ae) - np.min(ae))
    
    # Smoothness: average transition cost along path
    smooth = []
    for i in range(len(proj_vecs)-1):
        smooth.append(transition_cost(proj_vecs[i], proj_vecs[i+1], m))
    avg_smooth = np.mean(smooth) if smooth else 0
    
    return {
        'W': W, 'mean_purity': float(np.mean(purities)),
        'min_purity': float(np.min(purities)),
        'avg_transition': float(avg_smooth),
        'total_cost': float(total_cost),
        'proj_vecs': proj_vecs, 'dim': dim,
    }


def cross_overlap(pA, pB, m):
    nk = min(len(pA), len(pB)); S = []
    for ik in range(nk):
        UA, UB = pA[ik], pB[ik]; dA, dB = UA.shape[0], UB.shape[0]; d = max(dA, dB)
        if dA < d: UA = np.vstack([UA, np.zeros((d-dA, m), dtype=complex)])
        if dB < d: UB = np.vstack([UB, np.zeros((d-dB, m), dtype=complex)])
        S.append(float(np.sum(np.abs(UA.conj().T @ UB)**2)))
    return np.array(S)


def main():
    t0 = time.time()
    theta = 9.6
    m_values = [2, 4, 6]
    shells = list(range(2, 8))
    lam = 2.0  # smoothness weight
    
    print(f"MARKOV BPDM (VITERBI): Kagome θ={theta}°, λ={lam}", flush=True)
    print("="*80, flush=True)
    
    results = {}
    
    for m in m_values:
        print(f"\n--- m={m} ---", flush=True)
        print(f"{'ns':>3} {'dim':>5} {'W/t':>9} {'pur':>5} {'minp':>5} {'d_avg':>6} "
              f"{'Sm':>7} {'Sn':>7} {'Δ%':>6}", flush=True)
        
        pp = None; pW = None; mr = []
        
        for ns in shells:
            t1 = time.time()
            r = run_markov(theta, ns, m, lam=lam, nk=12, L=8, max_cands=25)
            dt = time.time() - t1
            
            if pp is not None:
                S = cross_overlap(pp, r['proj_vecs'], m)
                Sm = float(np.mean(S)); Sn = float(np.min(S))
            else:
                Sm = Sn = float('nan')
            
            dp = abs(r['W']-pW)/max(r['W'],1e-10)*100 if pW else float('nan')
            
            mr.append({
                'ns': ns, 'dim': r['dim'], 'W': r['W'],
                'mp': r['mean_purity'], 'minp': r['min_purity'],
                'davg': r['avg_transition'], 'Sm': Sm, 'Sn': Sn, 'dp': dp,
            })
            
            sm = f"{Sm:.4f}" if not np.isnan(Sm) else "—"
            sn = f"{Sn:.4f}" if not np.isnan(Sn) else "—"
            d = f"{dp:.1f}" if not np.isnan(dp) else "—"
            
            print(f"{ns:>3} {r['dim']:>5} {r['W']:>9.4f} {r['mean_purity']:>5.3f} "
                  f"{r['min_purity']:>5.3f} {r['avg_transition']:>6.3f} "
                  f"{sm:>7} {sn:>7} {d:>6}  ({dt:.0f}s)", flush=True)
            
            pp = r['proj_vecs']; pW = r['W']
            if dt > 120:
                print(f"  (stop m={m})", flush=True); break
        
        results[m] = mr
    
    # Summary
    print(f"\n{'='*80}", flush=True)
    print("SUMMARY: Markov vs Independent BPDM", flush=True)
    print(f"{'m':>3} {'W_range':>14} {'ratio':>6} {'Sn_last':>8} {'last2Δ':>14} {'d_avg':>6}", flush=True)
    
    for m in m_values:
        rs = results[m]
        ws = [r['W'] for r in rs]; ratio = max(ws)/max(min(ws),1e-10)
        sn = [r['Sn'] for r in rs if not np.isnan(r['Sn'])]
        ch = [r['dp'] for r in rs if not np.isnan(r['dp'])]
        davg = [r['davg'] for r in rs]
        ls = f"{sn[-1]:.4f}" if sn else "—"
        l2 = ch[-2:] if len(ch) >= 2 else ch
        print(f"{m:>3} {min(ws):.3f}—{max(ws):.3f} {ratio:>6.2f}x {ls:>8} "
              f"{str([f'{c:.0f}%' for c in l2]):>14} {davg[-1]:>6.3f}", flush=True)
    
    # Key question: did Markov smoothness improve S_min?
    print(f"\nKey: Does global smoothness constraint improve S_min?", flush=True)
    for m in m_values:
        rs = results[m]
        sn = [r['Sn'] for r in rs if not np.isnan(r['Sn'])]
        if sn and sn[-1] > 0.3 * m:
            print(f"  m={m}: ★ S_min={sn[-1]:.4f} > {0.3*m:.1f} — CONVERGING", flush=True)
        elif sn:
            print(f"  m={m}: S_min={sn[-1]:.4f} < {0.3*m:.1f} — not stable", flush=True)
    
    print(f"\nTotal: {time.time()-t0:.0f}s", flush=True)
    
    sv = {str(k): v for k, v in results.items()}
    with open('/home/darre/.openclaw/workspace/quantum-computer/numerics/bpdm_markov.json', 'w') as f:
        json.dump(sv, f, indent=2)


if __name__ == '__main__':
    main()

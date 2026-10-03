"""
Rough 2D collection-efficiency model for an OPD with an embedded (transparent) grid cathode.

- Electric field: Laplace solution (no space charge), electrodes as Dirichlet nodes.
- Carriers: steady-state drift-diffusion with first-order recombination (lifetime tau),
  Scharfetter-Gummel discretisation on a vertex grid, periodic in x.
- For each carrier the adjoint problem gives the collection probability P(r) at its
  own electrode(s); wrong-type electrodes either absorb (ohmic metal, default) or block.
- Pair collection ~ P_n(r) * P_p(r); efficiency = sum(G * P_n * P_p) / sum(G).
- Generation profile is 1D (grid optics cost <= ~1 pp per FDTD group 05), taken from the
  TMM profile of the 655 nm device (g1D_L655_800_1100.txt).

Structures:
  orig : bottom cathode / BHJ / top anode (no grid)
  grid : bottom cathode / BHJ / grid cathode / BHJ / top anode (current design)
  sym  : bottom anode / BHJ / grid cathode / BHJ / top anode (proposed)
  cont : bottom anode / BHJ / continuous middle cathode / BHJ / top anode (group 07)
Grid line = Ag core (w x 20 nm) + 10 nm ZnO shell -> conducting block (w+20) x 40 nm.
"""
import os
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spl

VT = 0.02585
d, zg, t = 655e-9, 290e-9, 40e-9
VBI = 0.6   # built-in potential estimate from group 06 parameter table

def bern(x):
    x = np.asarray(x, float); out = np.empty_like(x); s = np.abs(x) < 1e-6
    out[s] = 1 - x[s]/2; out[~s] = x[~s]/np.expm1(x[~s]); return out

def build(struct, p, h, w=100e-9):
    nx = max(int(round(p/h)), 1) if struct != 'orig' else 1
    nz = int(round(d/h)) + 1
    z = np.arange(nz)*h; x = (np.arange(nx)+0.5)*h
    X, Z = np.meshgrid(x, z, indexing='ij')
    bot = np.zeros((nx,nz),bool); bot[:,0] = True
    top = np.zeros((nx,nz),bool); top[:,-1] = True
    grid = np.zeros((nx,nz),bool)
    if struct in ('grid','sym'):
        grid = (np.abs(X-p/2) <= (w+20e-9)/2) & (np.abs(Z-zg) <= t/2)
    elif struct == 'cont':
        nx = 1; X, Z = X[:1], Z[:1]; bot, top = bot[:1], top[:1]
        grid = np.abs(Z-zg) <= t/2
    if struct in ('orig','grid'):
        cath = bot | grid; anod = top
    else:
        cath = grid; anod = bot | top
    return nx, nz, z, cath, anod

def laplace(nx, nz, h, cath, anod, Vk):
    fixed = cath | anod; val = np.where(cath, Vk, 0.0)
    idx = np.arange(nx*nz).reshape(nx,nz); I,J = np.meshgrid(np.arange(nx),np.arange(nz),indexing='ij')
    k = idx.ravel(); fx = fixed.ravel()
    rows=[k]; cols=[k]; data=[np.where(fx,1.0,0.0)]; diag = np.zeros(nx*nz)
    for di,dj in [(1,0),(-1,0),(0,1),(0,-1)]:
        if nx == 1 and di != 0: continue
        In=(I+di)%nx; Jn=J+dj; ok=(Jn>=0)&(Jn<nz)&~fixed
        rows.append(idx[ok]); cols.append(idx[In[ok],Jn[ok]]); data.append(np.ones(ok.sum()))
        np.add.at(diag, idx[ok], -1.0)
    rows.append(k); cols.append(k); data.append(np.where(fx,0.0,diag))
    A = sp.csr_matrix((np.concatenate(data),(np.concatenate(rows),np.concatenate(cols))),shape=(nx*nz,)*2)
    return spl.spsolve(A, np.where(fx, val.ravel(), 0.0)).reshape(nx,nz)

def collection_prob(V, h, collect, block, mu, tau, sign, wrong='sink'):
    """sign=+1 electrons (move to higher V), -1 holes. Returns P on all nodes.
    wrong='sink': wrong-type electrode absorbs the carrier (P=0, ohmic metal contact);
    wrong='block': wrong-type electrode reflects it (ideal selective interlayer)."""
    nx, nz = V.shape; D = mu*VT; psi = sign*V/VT
    idx = np.arange(nx*nz).reshape(nx,nz); I,J = np.meshgrid(np.arange(nx),np.arange(nz),indexing='ij')
    solid = collect | block; free = ~solid
    rows=[]; cols=[]; data=[]; diag = np.full(nx*nz, -1.0/tau); rhs = np.zeros(nx*nz)
    for di,dj in [(1,0),(-1,0),(0,1),(0,-1)]:
        if nx == 1 and di != 0: continue
        In=(I+di)%nx; Jn=J+dj; inb=(Jn>=0)&(Jn<nz)
        a = free & inb
        i_ = idx[a]; jn = idx[In[a],Jn[a]]
        c = D/h**2 * bern(psi.ravel()[i_] - psi.ravel()[jn])   # rate i -> neighbour
        nb_free = free.ravel()[jn]; nb_coll = collect.ravel()[jn]
        if wrong == 'sink':
            nb_sink = block.ravel()[jn]; np.add.at(diag, i_[nb_sink], -c[nb_sink])
        # adjoint: P_i * (-sum rates - 1/tau) + sum rate * P_j = 0, P=1 at collecting nodes
        np.add.at(diag, i_[nb_free|nb_coll], -c[nb_free|nb_coll])
        rows.append(i_[nb_free]); cols.append(jn[nb_free]); data.append(c[nb_free])
        np.add.at(rhs, i_[nb_coll], -c[nb_coll])
    fi = np.where(free.ravel())[0]; m = -np.ones(nx*nz,int); m[fi] = np.arange(fi.size)
    r = np.concatenate(rows); cc = np.concatenate(cols); dd = np.concatenate(data)
    A = sp.csr_matrix((np.concatenate([dd, diag[fi]]), (np.concatenate([m[r], m[fi]]), np.concatenate([m[cc], m[fi]]))), shape=(fi.size,)*2)
    P = np.zeros(nx*nz); P[collect.ravel()] = 1.0
    P[fi] = spl.spsolve(A.tocsc(), rhs[fi])
    return P.reshape(nx,nz)

_G = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'g1D_L655_800_1100.txt'))
def gen_profile(z, kind):
    col = {'800': 1, '1100': 2}[kind]
    return np.interp(z*1e9, _G[:,0], _G[:,col])

def efficiency(struct, p, Vapp, kind, mu_n, mu_p, tau_n, tau_p, h=5e-9, wrong='sink', w=100e-9):
    nx, nz, z, cath, anod = build(struct, p, h, w)
    V = laplace(nx, nz, h, cath, anod, VBI - Vapp)
    Pn = collection_prob(V, h, cath, anod, mu_n, tau_n, +1, wrong)
    Pp = collection_prob(V, h, anod, cath, mu_p, tau_p, -1, wrong)
    G = np.broadcast_to(gen_profile(z, kind), V.shape).copy()
    G[cath | anod] = 0
    return (G*Pn*Pp).sum()/G.sum()

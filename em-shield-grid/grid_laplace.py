# 2D Laplace (uniform eps, no space charge): bottom cathode z=0 (V=0), top anode z=d (V=1),
# embedded grid line (width w, thickness t, centered at zg) held at Vg; periodic in x with pitch p.
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spl
d, zg, w, t = 655e-9, 290e-9, 100e-9, 50e-9
def solve(p, Vg, Va=1.0, h=5e-9):
    nx = int(round(p/h)); nz = int(round(d/h))+1
    x = (np.arange(nx)+0.5)*h; z = np.arange(nz)*h
    X, Z = np.meshgrid(x, z, indexing='ij')
    metal = (np.abs(X-p/2) <= w/2) & (np.abs(Z-zg) <= t/2)
    fixed = np.zeros((nx,nz),bool); val = np.zeros((nx,nz))
    fixed[:,0]=True; fixed[:,-1]=True; val[:,-1]=Va
    fixed|=metal; val[metal]=Vg
    idx = np.arange(nx*nz).reshape(nx,nz)
    rows=[];cols=[];data=[]; b=np.zeros(nx*nz)
    I,J = np.meshgrid(np.arange(nx),np.arange(nz),indexing='ij')
    k = idx.ravel(); fx = fixed.ravel()
    rows.append(k); cols.append(k); data.append(np.where(fx,1.0,-4.0))
    b[fx] = val.ravel()[fx]
    for di,dj in [(1,0),(-1,0),(0,1),(0,-1)]:
        In = (I+di)%nx; Jn = J+dj; ok = (Jn>=0)&(Jn<nz)&~fixed
        rows.append(idx[ok]); cols.append(idx[In[ok],Jn[ok]]); data.append(np.ones(ok.sum()))
    A = sp.csr_matrix((np.concatenate(data),(np.concatenate(rows),np.concatenate(cols))),shape=(nx*nz,nx*nz))
    V = spl.spsolve(A,b).reshape(nx,nz)
    E0 = Va/d
    Ez_cath = (V[:,1]-V[:,0])/h                # field at cathode surface -> induced charge
    lower = slice(0, int((zg-t/2-20e-9)/h))
    Ez_low = np.diff(V[:,lower],axis=1)/h
    return Ez_cath.mean()/E0, np.abs(Ez_low).mean()/E0, np.abs(Ez_low).min()/E0
print(f"{'pitch um':>8} | grounded grid: cathode coupling, mean|E| lower, min|E| lower | matched grid (Vg=zg/d): mean|E| lower")
for p in [0.3,0.5,1,2,4,8,16]:
    h = 5e-9 if p<=2 else 10e-9
    c0,m0,n0 = solve(p*1e-6, 0.0, h=h)
    c1,m1,n1 = solve(p*1e-6, zg/d, h=h)
    print(f"{p:8.1f} | {c0:8.3f} {m0:8.3f} {n0:8.3f} | {m1:8.3f}")

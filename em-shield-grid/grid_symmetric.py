# Laplace estimate for the "anode / BHJ / grid cathode / BHJ / anode" structure.
# Bottom electrode (z=0) and top electrode (z=d) both at V=0 (anodes), grid line at V=1 (cathode).
# Reports how much of the BHJ volume has a weak field (|E| < 20% of the nominal segment field),
# i.e. the lateral "saddle" between grid lines where carriers rely on diffusion.
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spl
d, zg, w, t = 655e-9, 290e-9, 100e-9, 50e-9
def solve(p, h=5e-9):
    nx = int(round(p/h)); nz = int(round(d/h))+1
    x = (np.arange(nx)+0.5)*h; z = np.arange(nz)*h
    X, Z = np.meshgrid(x, z, indexing='ij')
    metal = (np.abs(X-p/2) <= w/2) & (np.abs(Z-zg) <= t/2)
    fixed = np.zeros((nx,nz),bool); fixed[:,0]=fixed[:,-1]=True; fixed|=metal
    val = np.where(metal,1.0,0.0)
    idx = np.arange(nx*nz).reshape(nx,nz); I,J = np.meshgrid(np.arange(nx),np.arange(nz),indexing='ij')
    k=idx.ravel(); fx=fixed.ravel()
    rows=[k]; cols=[k]; data=[np.where(fx,1.0,-4.0)]; b=np.where(fx,val.ravel(),0.0)
    for di,dj in [(1,0),(-1,0),(0,1),(0,-1)]:
        In=(I+di)%nx; Jn=J+dj; ok=(Jn>=0)&(Jn<nz)&~fixed
        rows.append(idx[ok]); cols.append(idx[In[ok],Jn[ok]]); data.append(np.ones(ok.sum()))
    A=sp.csr_matrix((np.concatenate(data),(np.concatenate(rows),np.concatenate(cols))),shape=(nx*nz,)*2)
    V=spl.spsolve(A,b).reshape(nx,nz)
    Ex=(np.roll(V,-1,0)-np.roll(V,1,0))/(2*h); Ez=np.gradient(V,h,axis=1)
    E=np.hypot(Ex,Ez)[~metal]; Zs=Z[~metal]
    Eref=np.where(Zs<zg,1/zg,1/(d-zg))
    return (E<0.2*Eref).mean(), (E<0.5*Eref).mean()
print("pitch um | weak-field volume fraction (<20%, <50% of nominal)")
for p in [0.2,0.3,0.5,0.75,1,2,4]:
    a,b=solve(p*1e-6, 5e-9 if p<=1 else 10e-9)
    print(f"{p:8.2f} | {a:6.3f} {b:6.3f}")

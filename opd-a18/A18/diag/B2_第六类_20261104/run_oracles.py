"""Family-oracle Gaussian-mixture posterior with the actual real-mode noise.

No true spectrum or brightness enters inference. The observation is only the
direction y/||y||. Integrate lognormal gain and the unknown positive radial
coordinate. Gaussian components describe area-normalized training spectra.
This is MMSE under the fitted Gaussian mixture, NOT a universal error bound.
"""
from pathlib import Path
import sys, time
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
from bootstrap_b2 import *
from scipy.linalg import cho_factor,cho_solve,solve_triangular
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
threadpool_limits(limits=4)

def fit_prior(S,K,shrink,labels=None):
    d=S.shape[1];project=np.eye(d)-np.ones((d,d))/d
    if labels is None:
        labels=np.zeros(len(S),dtype=int) if K==1 else KMeans(n_clusters=K,random_state=18,n_init=3,max_iter=150).fit_predict(S)
    mus=[];covs=[];weights=[];sizes=[]
    for k in range(K):
        z=S[labels==k]
        if len(z)<3:continue
        mu=z.mean(0);c=np.cov(z.T)
        # Respect exact unit-area constraint; ridge only on its tangent subspace.
        c=(1-shrink)*c+shrink*np.trace(c)/(d-1)*project
        mus.append(mu);covs.append(c);weights.append(len(z)/len(S));sizes.append(len(z))
    return dict(mu=np.array(mus),cov=np.array(covs),weight=np.array(weights)/sum(weights),sizes=sizes)

def posterior(prior,Y,R,noise,gain_nodes=15,radial_nodes=48):
    """Integrals of p(a*u|k,g) * a^(n-1), a>0; Gaussian conditional E[x|a*u]."""
    n=R.shape[0];d=R.shape[1]
    U=Y/np.linalg.norm(Y,axis=1,keepdims=True)
    hg,wg=np.polynomial.hermite.hermgauss(gain_nodes)
    gains=np.exp(np.sqrt(2)*AD.ER['sigma_g']*hg);lg=np.log(wg)-.5*np.log(np.pi)
    hr,wr=np.polynomial.hermite.hermgauss(radial_nodes)
    logw=[];means=[]
    for mu,C,pi in zip(prior['mu'],prior['cov'],prior['weight']):
        rm=R@mu;rc=R@C
        for g,lp in zip(gains,lg):
            cov=g*g*rc@R.T+noise;cov=(cov+cov.T)/2
            cf=cho_factor(cov,lower=True,check_finite=False)
            ciu=cho_solve(cf,U.T,check_finite=False).T
            m=g*rm;cim=cho_solve(cf,m,check_finite=False)
            aa=np.einsum('ij,ij->i',U,ciu);bb=U@cim;cc=float(m@cim)
            disc=np.sqrt(bb*bb+4*aa*(n-1))
            # Stable positive quadratic root, even for negative bb.
            mode=np.where(bb>=0,(bb+disc)/(2*aa),2*(n-1)/(disc-bb))
            sd=1/np.sqrt(aa+(n-1)/mode**2)
            a=mode[:,None]+np.sqrt(2)*sd[:,None]*hr[None,:]
            valid=a>0;ap=np.maximum(a,np.finfo(float).tiny)
            expo=(n-1)*np.log(ap)-.5*(aa[:,None]*ap**2-2*bb[:,None]*ap+cc)
            lw=expo+hr[None,:]**2+np.log(wr)[None,:]+np.log(np.sqrt(2)*sd)[:,None]
            lw=np.where(valid,lw,-np.inf)
            lz=logsumexp(lw,axis=1)
            ea=np.sum(np.exp(lw-lz[:,None])*np.where(valid,a,0),axis=1)
            norm=-.5*(n*np.log(2*np.pi)+2*np.log(np.diag(cf[0])).sum())
            logw.append(np.log(pi)+lp+norm+lz)
            gainmat=cho_solve(cf,g*rc,check_finite=False)
            means.append(mu+(ea[:,None]*U-m)@gainmat)
    logw=np.array(logw).T
    wt=np.exp(logw-logsumexp(logw,axis=1,keepdims=True))
    X=np.einsum('nk,knd->nd',wt,np.array(means))
    assert np.all(np.isfinite(X))
    # Unit-area constraint should survive Gaussian conditionals and mixing.
    assert np.max(np.abs(X.sum(1)-1))<1e-5
    return X

def metric(X,S):
    full=np.zeros((len(X),len(BM)));full[:,BM]=np.maximum(X,0)
    return AD.LD.band_nrmse(full.T,S.T)

def invisibility(R,noise,tr,dg):
    n,d=R.shape
    u,sv,vh=np.linalg.svd(R,full_matrices=True)
    rank=int(np.sum(sv>sv[0]*1e-10));cut=int(np.sum(sv>sv[0]*.02))
    null=vh[rank:];weak=vh[cut:]
    def ratio(X,V):return float(np.sum((X@V.T)**2)/np.sum(X**2))
    trarea=tr/tr.sum(1,keepdims=True);da=dg/dg.sum(1,keepdims=True)
    mu=trarea.mean(0);c=np.cov(trarea.T)
    chol=np.linalg.cholesky(noise)
    white=solve_triangular(chol,R,lower=True)
    h=white@mu;h=h/np.linalg.norm(h)
    eff=white-np.outer(h,h@white)  # discard unknown global amplitude direction
    uu,ss,vv=np.linalg.svd(eff,full_matrices=True)
    ssfull=np.zeros(d);ssfull[:len(ss)]=ss
    vari=np.einsum('ij,jk,ik->i',vv,c,vv)
    snr=ssfull**2*np.maximum(vari,0)
    noiseweak=vv[snr<1]
    centered=da-mu
    return dict(rank=rank,rank_relative_2pct=cut,singular_values=sv.tolist(),
        null_energy_fraction=ratio(dg,null),weak_2pct_energy_fraction=ratio(dg,weak),
        centered_area_null_fraction=ratio(centered,null),
        centered_area_noise_unidentifiable_fraction=ratio(centered,noiseweak),
        noise_SNR_below_one_directions=int(np.sum(snr<1)),noise_snr_variance=snr.tolist(),
        definition='energy ratio of sums; noise criterion uses training variance, whitened R, unknown amplitude projected out; correlations can predict weak modes')

def main():
    t0=time.perf_counter();z=np.load(OUT/'allowed_data.npz')
    R=z['RN'][:,BM];noise=z['DEVCOV'];selections=[];candidates=[];nullrows=[];detail={}
    errors={};outpred={}
    for j,k in enumerate(FAMILIES):
        tr=z[f'tr{j}'][:,BM];va=z[f'va{j}'];dg=z[f'dg{j}']
        a=tr/tr.sum(1,keepdims=True)
        yv=z[f'yv{j}'].transpose(0,2,1).reshape(-1,R.shape[0])
        yd=z[f'yd{j}'].transpose(0,2,1).reshape(-1,R.shape[0])
        sv=np.tile(va,(4,1));sd=np.tile(dg,(4,1))
        best=(np.inf,None,None);single=None
        grid=[1,8,32,64]
        best_by_K={}
        for K in grid:
            labels=np.zeros(len(a),dtype=int) if K==1 else KMeans(n_clusters=K,random_state=18,n_init=3,max_iter=150).fit_predict(a)
            for shrink in (1e-4,1e-2):
                prior=fit_prior(a,K,shrink,labels)
                pred=posterior(prior,yv,R,noise)
                e=float(metric(pred,sv).mean())
                row=dict(family=k,prior_kind='spectral_kmeans',K=K,actual_K=len(prior['mu']),shrink=shrink,validation_nrmse=e,
                         min_cluster_size=min(prior['sizes']),max_cluster_size=max(prior['sizes']))
                candidates.append(row)
                best_by_K[K]=min(best_by_K.get(K,np.inf),e)
                if K==1 and (single is None or e<single[0]):single=(e,prior,shrink)
                if e<best[0]:best=(e,prior,row)
                log(f'Oracle {k}: K={K} shrink={shrink:g} validation={e:.6f}')
            # Validation-only saturation check. Do not use diagnostic errors to
            # decide which families receive a richer prior.
            if K==64 and best_by_K[64]<.95*best_by_K[32]:
                grid.extend([128,256])
                log(f'Oracle {k}: validation still improves >5%; extend K to 128,256.')
        if k=='同分布五类':
            # The existing generator supplies training-only subtype labels.
            # Preserve those known submanifolds in the prior, but DO NOT give
            # validation/diagnostic subtype labels to inference.
            s0,l0,*_=AD.D.gen5(3000,31338,AD.BAND)
            s1,l1,*_=AD.D.gen5(9000,31344,AD.BAND)
            assert np.array_equal(np.concatenate([s0,s1])[:,BM],tr)
            subtype=np.concatenate([l0,l1])
            for pertype in (1,8,32,64):
                labs=np.zeros(len(a),dtype=int);offset=0
                for typ in np.unique(subtype):
                    ix=np.flatnonzero(subtype==typ)
                    lab=np.zeros(len(ix),dtype=int) if pertype==1 else KMeans(n_clusters=pertype,random_state=18,n_init=3,max_iter=150).fit_predict(a[ix])
                    labs[ix]=offset+lab;offset+=pertype
                for shrink in (1e-4,1e-2):
                    prior=fit_prior(a,offset,shrink,labs)
                    e=float(metric(posterior(prior,yv,R,noise),sv).mean())
                    row=dict(family=k,prior_kind='training_subtype_mixture',K=offset,actual_K=len(prior['mu']),
                             shrink=shrink,validation_nrmse=e,min_cluster_size=min(prior['sizes']),max_cluster_size=max(prior['sizes']))
                    candidates.append(row)
                    if e<best[0]:best=(e,prior,row)
                    log(f'Oracle {k}: training subtype mixture K={offset} shrink={shrink:g} validation={e:.6f}')
        # All candidate choices finished before touching diagnostic oracle errors.
        vscore,prior,row=best
        pred=posterior(prior,yd,R,noise)
        check=posterior(prior,yd,R,noise,gain_nodes=25,radial_nodes=80)
        ep=metric(pred,sd);ec=metric(check,sd)
        gaussian=posterior(single[1],yd,R,noise)
        eg=metric(gaussian,sd)
        er=ep.reshape(4,len(dg));errors[f'dg{j}']=er;outpred[f'pred{j}']=pred
        null=invisibility(R,noise,tr,dg[:,BM]);detail[k]=null
        nullrows.append(dict(family=k,**{x:y for x,y in null.items() if not isinstance(y,(list,str))}))
        rs=dict(family=k,prior_kind=row['prior_kind'],selected_K=row['K'],shrink=row['shrink'],validation_nrmse=vscore,
            oracle_nrmse=float(ep.mean()),gaussian_K1_nrmse=float(eg.mean()),
            refined_quadrature_nrmse=float(ec.mean()),quadrature_max_spectrum_difference=float(np.max(np.abs(pred-check))),
            quadrature_mean_nrmse_difference=float(ec.mean()-ep.mean()),
            per_spectrum_mean_min=float(er.mean(0).min()),per_spectrum_mean_max=float(er.mean(0).max()))
        selections.append(rs)
        np.savez_compressed(OUT/f'prior_{j}.npz',**{v:prior[v] for v in ('mu','cov','weight')})
        csvwrite('oracle_candidates.csv',candidates);csvwrite('oracle_metrics.csv',selections)
        csvwrite('nullspace_metrics.csv',nullrows)
        log(f'{k} diagnostic oracle={ep.mean():.6f}, Gaussian={eg.mean():.6f}, null={null["null_energy_fraction"]:.4%}')
    np.savez_compressed(OUT/'oracle_errors.npz',**errors)
    np.savez_compressed(OUT/'oracle_predictions.npz',**outpred)
    dump('nullspace_detail.json',detail)
    dump('oracle_summary.json',dict(simulation_only=True,seconds=time.perf_counter()-t0,metrics=selections,
        prior='KMeans spectral-area Gaussian mixture, K and covariance shrinkage selected only on validation; extend 64 to 128/256 only if validation gain over 32 exceeds 5%',
        likelihood='exact simulator additive Gaussian dark+white noise; marginal lognormal gain and positive radial coordinate',
        inference_input='reading direction only; true family known; no true spectral amplitude/true power supplied',
        caveats=['MMSE only under fitted area-GMM and squared area loss; shape NRMSE is a different metric',
                 'Non-Gaussian true families, finite training library, Gaussian tails, clipping and peak renormalization prevent a strict universal lower bound',
                 'Nullspace energy alone is not irreducible error; a prior can infer correlated nullspace components']))
    log('Oracle phase complete.')

if __name__=='__main__':main()

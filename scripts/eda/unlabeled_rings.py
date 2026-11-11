import sys, os, numpy as np, tifffile, pandas as pd
from multiprocessing import Pool
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
D=sys.argv[1]
def f(fn):
    img=tifffile.imread(os.path.join(D,'train_samples',fn))[::4,::4].astype(np.float32)
    m=tifffile.imread(os.path.join(D,'train_instance_labels',fn))[::4,::4]
    sm=ndi.gaussian_filter(img,1.5)
    my=(m>=3)&(m%2==1); bg=m==0
    thr=(np.median(sm[my])+np.median(sm[bg]))/2
    dark=ndi.binary_opening(sm<thr,iterations=2)
    holes=ndi.binary_fill_holes(dark)&~dark
    L=label(holes); n_unl=n_lab=n_art=0
    for p in regionprops(L):
        if not (1500/16 <= p.area <= 1.2e6/16) or p.solidity<0.8: continue
        vals=m[L==p.label]
        frac_lab=(vals>=2).mean(); frac_art=(vals==1).mean()
        if frac_lab>0.5: n_lab+=1
        elif frac_art>0.5: n_art+=1
        else: n_unl+=1
    n_gt=len([i for i in np.unique(m) if i>=2 and i%2==0])
    return dict(file=fn,ring_cands_labeled=n_lab,ring_cands_artifact=n_art,ring_cands_unlabeled=n_unl,n_gt=n_gt)
if __name__=='__main__':
    fs=sorted(os.listdir(os.path.join(D,'train_instance_labels')))
    with Pool(8) as p: r=p.map(f,fs)
    pd.DataFrame(r).to_csv(sys.argv[2],index=False)

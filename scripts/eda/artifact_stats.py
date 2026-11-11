import sys, os, numpy as np, tifffile, pandas as pd
from multiprocessing import Pool
from skimage.measure import label, regionprops
from scipy import ndimage as ndi
D=sys.argv[1]
def f(fn):
    m=tifffile.imread(os.path.join(D,'train_instance_labels',fn))
    a=(m==1); L=label(a,connectivity=2); H,W=m.shape; out=[]
    for p in regionprops(L):
        r0,c0,r1,c1=p.bbox
        sub=L[r0:r1,c0:c1]==p.label
        filled=ndi.binary_fill_holes(sub)
        # does the artifact enclose labeled fibers?
        inside=m[r0:r1,c0:c1][filled & ~sub]
        out.append(dict(file=fn,area=p.area,border=(r0==0 or c0==0 or r1==H or c1==W),solidity=p.solidity,ecc=p.eccentricity,
                        euler=p.euler_number,encl_lab=int((inside>=2).sum()),encl_bg=int((inside==0).sum()),h=r1-r0,w=c1-c0))
    return out
if __name__=='__main__':
    fs=sorted(os.listdir(os.path.join(D,'train_instance_labels')))
    with Pool(8) as p: r=p.map(f,fs)
    pd.DataFrame([x for y in r for x in y]).to_csv(sys.argv[2],index=False)

"""Fixed eight-view inference; no fitted weights or threshold changes."""
import numpy as np
import torch
from torch.utils.data import DataLoader


def transform(x, k, flip):
    x=torch.rot90(x,k,(-2,-1))
    return x.flip(-1) if flip else x


def inverse(x, k, flip):
    if flip:x=x.flip(-1)
    return torch.rot90(x,-k,(-2,-1))


@torch.inference_mode()
def predict_dihedral(model,dataset):
    model.eval(); output=[]
    for x,g,m,_ in DataLoader(dataset,batch_size=8,num_workers=0,pin_memory=True):
        x,g,m=x.cuda(),g.cuda(),m.cuda(); total=None
        with torch.autocast('cuda',dtype=torch.bfloat16):
            for k in range(4):
                for flip in (False,True):
                    p=model(transform(x,k,flip),transform(g,k,flip),m).float().sigmoid()
                    p=inverse(p,k,flip); total=p if total is None else total+p
        output.append((total/8).cpu().numpy())
    return np.concatenate(output)

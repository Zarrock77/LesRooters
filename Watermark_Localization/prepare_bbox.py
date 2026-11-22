"""Fetch a pinned Adobe detector implementation and verified official weights.

Run on a login node (network/files only), never replace the active environment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

COMMIT='4c60b9fdb4125fca3001f0803e6890a6b55371ac'
FILES={'trustmark_bbox_Q.ckpt':'9d15428a33e15140ea16aa378416d304','trustmark_bbox_Q.yaml':'749b0d62106f8f6648e6f781c3143105'}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--kit-root',type=Path,required=True)
    args=ap.parse_args();root=args.kit_root/'models/trustmark_bbox';root.mkdir(parents=True,exist_ok=True)
    source=root/'source'
    if not source.exists():
        subprocess.run(['git','clone','--filter=blob:none','--no-checkout','https://github.com/adobe/trustmark.git',str(source)],check=True)
        subprocess.run(['git','-C',str(source),'checkout','--detach',COMMIT],check=True)
    assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==COMMIT
    records={}
    for filename,expected in FILES.items():
        path=root/filename;url='https://cai-watermark.adobe.net/watermarking/trustmark-models/'+filename
        if not path.exists():urllib.request.urlretrieve(url,path)
        data=path.read_bytes();assert hashlib.md5(data).hexdigest()==expected,filename
        records[filename]={'url':url,'md5':expected,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
        print('VERIFIED',filename,len(data),flush=True)
    (root/'provenance.json').write_text(json.dumps({'source_commit':COMMIT,'files':records},indent=2)+'\n')


if __name__=='__main__':main()

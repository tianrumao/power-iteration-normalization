"""Archive completed pilot weights/results; preserve raw-array locations and manifests."""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--shared',type=Path,required=True);p.add_argument('--selected',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--archives',type=Path,required=True);a=p.parse_args()
    assert json.loads((a.selected/'generalization-audit/complete.json').read_text())['status']=='complete'
    assert json.loads((a.selected/'generalization-audit/leakage-and-numerics.json').read_text())['status']=='complete'
    path=Path('/content/matched-pilots-20261127.zip');manifest=[]
    raw_index={'raw_arrays_embedded_in_this_bundle':False,'persistent_raw_array_directory':str(a.cache),
        'note':'Actual raw .npy arrays remain in Google Drive. The archive includes the complete generator, exact split seeds, per-array digests, model checkpoints, validation histories and per-sample evaluation results.',
        'dataset_manifests':[json.loads(p.read_text()) for p in sorted(a.shared.glob('seed-*-data.json'))]}
    roots=[('shared_lr001',a.shared)]
    if a.selected.resolve()!=a.shared.resolve():roots.append(('condition_specific',a.selected))
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for label,root in roots:
            for file in sorted(root.rglob('*')):
                if file.is_file():
                    name=label+'/'+file.relative_to(root).as_posix();content=file.read_bytes();z.writestr(name,content)
                    manifest.append({'path':name,'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()})
        raw=json.dumps(raw_index,indent=2).encode();z.writestr('RAW_DATA_INDEX.json',raw)
        manifest.append({'path':'RAW_DATA_INDEX.json','bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
        z.writestr('ARCHIVE_MANIFEST.json',json.dumps(manifest,indent=2))
    digest=sha(path);a.archives.mkdir(parents=True,exist_ok=True)
    if path.stat().st_size<=240*1024*1024:
        shutil.copyfile(path,a.archives/path.name)
        print('MATCHED_PILOT_ARCHIVE',path.name,flush=True)
    else:
        # Store only parts in Drive, avoiding the redundant archive-plus-parts storage pattern.
        parts=[]
        with path.open('rb') as f:
            index=1
            while block:=f.read(240*1024*1024):
                part=a.archives/(path.name+f'.part{index:02d}');part.write_bytes(block)
                parts.append({'name':part.name,'bytes':len(block),'sha256':hashlib.sha256(block).hexdigest()});index+=1
        (a.archives/'matched-pilots-20261127-transfer.json').write_text(json.dumps({'archive':path.name,'archive_sha256':digest,'parts':parts},indent=2))
        print('MATCHED_PILOT_ARCHIVE_PARTS',json.dumps(parts),flush=True)
    (a.archives/(path.name+'.sha256')).write_text(digest+'  '+path.name+'\n')
    print('MATCHED_PILOT_ARCHIVE_BYTES',path.stat().st_size,flush=True)
    print('MATCHED_PILOT_ARCHIVE_SHA256',digest,flush=True)
    print('MATCHED_PILOT_ARCHIVE_FILES',len(manifest),flush=True)


if __name__=='__main__':main()

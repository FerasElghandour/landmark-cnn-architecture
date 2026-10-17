"""Prepare explicit train/val/test class folders for the public landmark CNN.

No download and no implicit split. The same class-to-index map applies to all
partitions; duplicate file bytes across partitions are rejected. Supply the full
landmark dataset separately; a handful of archived examples is not a benchmark.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from PIL import Image

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--size',type=int,default=224);a=p.parse_args()
 if a.output.exists() or a.size<32:raise ValueError('New output and image size>=32 required')
 classes=sorted(d.name for d in (a.root/'train').iterdir() if d.is_dir());seen={};data={};manifest={}
 if len(classes)<2:raise ValueError('At least two classes required')
 for split in ('train','val','test'):
  if sorted(d.name for d in (a.root/split).iterdir() if d.is_dir())!=classes:raise ValueError('Identical class folders required across splits')
  images=[];labels=[];records=[]
  for label,name in enumerate(classes):
   for path in sorted((a.root/split/name).iterdir()):
    if path.suffix.lower() not in ('.png','.jpg','.jpeg'):continue
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest in seen and seen[digest]!=split:raise ValueError('Duplicate image bytes across splits')
    seen[digest]=split
    with Image.open(path) as im:images.append(np.asarray(im.convert('RGB').resize((a.size,a.size))).transpose(2,0,1))
    labels.append(label);records.append({'relative_path':str(path.relative_to(a.root)),'sha256':digest})
  if not images:raise ValueError('Empty split')
  data['x_'+split]=np.stack(images);data['y_'+split]=np.array(labels);manifest[split]=records
 a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,**data)
 a.output.with_suffix('.manifest.json').write_text(json.dumps({'classes':classes,'fixed_normalization':'uint8 /127.5 -1','size':a.size,'images':manifest},indent=2)+'\n')
 a.output.with_suffix('.config.json').write_text(json.dumps({'classes':len(classes),'dropout':.7,'class_names':classes},indent=2)+'\n')
if __name__=='__main__':main()

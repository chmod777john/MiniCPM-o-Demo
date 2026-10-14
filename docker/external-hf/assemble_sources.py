"""Freeze external independent Git sources into a Docker build context."""
import argparse,json,subprocess,shutil
from pathlib import Path
p=argparse.ArgumentParser()
for name in ('hf','accel','demo','output'): p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args(); dest=a.output/'materials'; dest.mkdir(parents=True,exist_ok=True); record={}
for name in ('hf','accel','demo'):
 root=getattr(a,name).resolve()
 sha=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
 record[name]={'path':str(root),'sha':sha}
 cmd=['git','-C',str(root),'archive',sha]
 if name=='accel': cmd+=['src','pyproject.toml','README.md']
 with (dest/('accel-runtime.tar' if name=='accel' else name+'.tar')).open('wb') as f: subprocess.run(cmd,stdout=f,check=True)
mobile=a.demo/'static/mobile'
if not (mobile/'index.html').is_file(): raise SystemExit('Run mobile build:static before assembly')
shutil.copytree(mobile,dest/'mobile',dirs_exist_ok=True)
shutil.copytree(Path(__file__).parent,a.output/'docker/external-hf',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
(dest/'source-manifest.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))

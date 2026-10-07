"""Reproduce sealed CV in a new local checkout, preserving the original artifacts."""
from pathlib import Path
from datetime import datetime
import argparse,os,shutil,subprocess,sys
from dulwich import porcelain
from dulwich.repo import Repo
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from decisionos_sre.common import read,file_hash

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir')
parser.add_argument('--check-only',action='store_true')
args=parser.parse_args();source=read(root/'outputs/round13/pretraining_source.json')
base=(root/'work/cv_reproductions').resolve()
target=Path(args.output_dir).resolve() if args.output_dir else base/datetime.now().strftime('%Y%m%d-%H%M%S')
if not target.is_relative_to(base) or target==base:raise ValueError('Output must be a new child of '+str(base))
if target.exists():raise FileExistsError('Refusing to overwrite '+str(target))
for p,sha in source['source_hashes'].items():assert file_hash(root/p)==sha
files=[root/'data/round12'/n for n in ['manifest.json','splits.json','examples.json']]+list((root/'artifacts/backbone').glob('*'))
assert all(p.is_file() for p in files)
print('Verified sealed training revision',source['revision'],'data/backbone copy bytes',sum(p.stat().st_size for p in files),'new checkout',target,flush=True)
if args.check_only:raise SystemExit(0)
# Only this newly created destination can be reset; never reset the source repository.
target.parent.mkdir(parents=True,exist_ok=True)
child=porcelain.clone(str(root),str(target),checkout=False)
porcelain.reset(child,'hard',source['revision'])
assert Repo(target).head().decode()==source['revision']
for p in files:
 dest=target/p.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
shutil.copyfile(root/'outputs/round13/pretraining_source.json',target/'outputs/round13/pretraining_source.json')
for p,sha in source['source_hashes'].items():assert file_hash(target/p)==sha
env=os.environ.copy();env.update(PYTHONPATH=str(target/'src'),PYTHONIOENCODING='utf-8',TEMP=str(root/'work/tmp'),TMP=str(root/'work/tmp'))
subprocess.run([sys.executable,'scripts/run_round13_cv.py'],cwd=target,env=env,check=True)
subprocess.run([sys.executable,str(root/'scripts/audit_round13_cv.py')],cwd=target,env=env,check=True)
print('REPRODUCED',target/'outputs/round13/cv_results.json',flush=True)

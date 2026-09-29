"""Inspect, launch, and resume LongSpark evaluation with isolated GPU workers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .plan import ROOT, PRESETS, make_plan, identity

def write_new(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(obj,f,indent=2);f.write('\n')

def environment(gpus):
    env=dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)),
        PYTHONPATH=os.pathsep.join(map(str,[ROOT/'src',ROOT/'vendor/sglang/python',ROOT/'vendor'])),
        PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', SGLANG_USE_SGL_FA3_KERNEL='1',
        DFK_GLOBAL16_FUSED_QK_NORM_ROPE='0', SGLANG_DSPARK_FUSED_QK_NORM_ROPE='0',
        DFK_GLOBAL16_QK_SGLANG_RMSNORM='1', DFK_GLOBAL16_FUSED_SILU='1',
        DFK_GLOBAL16_DRAFT_FUSED_ALL_ATTENTION='1', DFK_GLOBAL16_FAST_METADATA='1',
        DFK_GLOBAL16_MARKOV_BFLOAT16='1')
    cuda=Path(env.get('CUDA_HOME','/usr/local/cuda'))
    if not (cuda/'bin/nvcc').is_file() or not (cuda/'include/cuda.h').is_file():
        raise RuntimeError('Set CUDA_HOME to a CUDA 12.8 toolkit (nvcc and include/cuda.h required).')
    version=subprocess.check_output([str(cuda/'bin/nvcc'),'--version'],text=True)
    if 'release 12.8' not in version:raise RuntimeError('This runtime requires CUDA 12.8')
    env['CUDA_HOME']=str(cuda)
    env['PATH']=os.pathsep.join([str(cuda/'bin'),str(Path(sys.executable).parent),env['PATH']])
    import shutil
    if shutil.which('ninja',path=env['PATH']) is None:
        raise RuntimeError('ninja is required for kernel compilation; activate the configured Conda environment.')
    env['LD_LIBRARY_PATH']=str(cuda/'lib64')+os.pathsep+env.get('LD_LIBRARY_PATH','')
    return env

def require_idle(gpus):
    output=subprocess.check_output(['nvidia-smi','--id='+','.join(map(str,gpus)),
        '--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    values=[int(x.strip()) for x in output.splitlines()]
    if len(values)!=len(gpus) or any(x>64 for x in values):
        raise RuntimeError(f'Selected GPUs are occupied: {values} MiB. No external tasks will be stopped.')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preset',choices=PRESETS,required=True)
    for name in ('size','method','dataset','seed','concurrency'):
        p.add_argument('--'+name,help='Comma-separated filter; omitted means all values in the preset')
    p.add_argument('--temperature',type=float,choices=(0.,1.),default=1.)
    p.add_argument('--graph-max-batch',type=int,help='Override graph capacity (at least concurrency, at most 128)')
    p.add_argument('--models',type=Path,default=ROOT/'configs/models.local.json')
    p.add_argument('--output',type=Path,default=ROOT/'results')
    p.add_argument('--gpus',default='0,1,2,3,4',help='Physical GPU IDs; first TP GPUs are target, next is draft')
    p.add_argument('--execute',action='store_true',help='Default is a CPU-only plan preview')
    p.add_argument('--smoke',action='store_true',help='Short functional check, excluded from metric summaries')
    p.add_argument('--resume',action='store_true',help='Skip only completed, audited, configuration-matching cells')
    a=p.parse_args();jobs=make_plan(a.preset)
    for key in ('size','method','dataset','seed','concurrency'):
        value=getattr(a,key)
        if value is not None:
            requested=set(value.split(','));available={str(j[key]) for j in jobs}
            if not requested<=available:p.error(f'Unknown {key}: {requested-available}')
            jobs=[j for j in jobs if str(j[key]) in requested]
    if not jobs:p.error('Empty selection')
    for j in jobs:
        j['temperature']=a.temperature
        if a.graph_max_batch is not None:
            if not j['concurrency']<=a.graph_max_batch<=128:
                p.error('--graph-max-batch must cover concurrency and be at most 128')
            j['graph_max_batch']=a.graph_max_batch
    if a.smoke:
        for j in jobs:
            j.update(smoke=True,concurrency=2,max_new_tokens=32,warmup_seconds=0.,measurement_seconds=2.)
        # A concurrency sweep collapses to a single C2 smoke per model/seed/dataset/method.
        jobs=list({identity(j):j for j in jobs}.values())
    if not a.execute:
        print(json.dumps(dict(cells=len(jobs),jobs=jobs),indent=2));return
    from .validate import validate_repository
    validate_repository()
    models=json.loads(a.models.read_text())
    gpus=[int(x) for x in a.gpus.split(',')]
    if len(gpus)!=len(set(gpus)) or min(gpus)<0:p.error('Distinct nonnegative GPU IDs required')
    out=a.output.resolve()/('smoke' if a.smoke else 'runs')/a.preset
    import fcntl
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for j in jobs:
            count=j['target_tp']+int(j['method']!='vanilla')
            if len(gpus)<count:p.error(f'{j["method"]} TP{j["target_tp"]} needs {count} GPUs')
            selected=gpus[:count]
            j['models']={k:str((ROOT/Path(v)).resolve()) for k,v in models[j['size']].items()}
            j['physical_gpus']=selected
            dest=out/identity(j);jobfile=dest/'job.json';audit=dest/'audit.json'
            if jobfile.exists():
                if a.resume and json.loads(jobfile.read_text())==j and audit.exists() and json.loads(audit.read_text()).get('passed'):
                    result=dest/'result.json'
                    receipt=json.loads(audit.read_text())
                    if not result.exists() or hashlib.sha256(result.read_bytes()).hexdigest()!=receipt.get('result_sha256'):
                        raise RuntimeError(f'Missing or modified audited result: {dest}')
                    print('SKIP',identity(j),flush=True);continue
                raise FileExistsError(f'{dest} exists; preserve it or select a new output directory')
            # Check model identities before reserving output or using GPUs.
            from .validate import validate_models
            validate_models(j)
            require_idle(selected);env=environment(selected)
            write_new(jobfile,j)
            print('RUN',identity(j),flush=True)
            with (dest/'worker.log').open('x') as log:
                process=subprocess.Popen([sys.executable,'-m','longspark.worker',str(jobfile)],
                    env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:code=process.wait()
                except BaseException:
                    import signal
                    os.killpg(process.pid,signal.SIGTERM)
                    process.wait(timeout=30)
                    raise
            if code or not audit.exists():raise RuntimeError(f'Worker failed; inspect {dest}/worker.log')
            if not json.loads(audit.read_text()).get('passed'):raise RuntimeError(f'Audit failed: {dest}')
            time.sleep(2)
    print(f'Completed {len(jobs)} selected cells. Results: {out}')

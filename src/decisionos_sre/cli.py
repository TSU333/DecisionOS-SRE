import argparse
from pathlib import Path
from .common import read,save

def main():
    parser=argparse.ArgumentParser(description="DecisionOS-SRE auditable MVP")
    parser.add_argument("--config",default="configs/mvp.json")
    sub=parser.add_subparsers(dest="command",required=True)
    for name in ["audit-data","prepare-data","make-splits","baseline"]:
        sub.add_parser(name)
    p=sub.add_parser("train"); p.add_argument("--mode",choices=["frozen","sft"],required=True)
    for name in ["calibrate","select-policy","evaluate","benchmark","serve"]:
        p=sub.add_parser(name); p.add_argument("--artifact",default="artifacts/sft")
        if name in ["calibrate","select-policy","evaluate"]:
            p.add_argument("--device",choices=["cpu","cuda"],default="cpu")
        if name=="serve":
            p.add_argument("--port",type=int,default=8000)
    args=parser.parse_args()
    cfg=read(args.config)
    if cfg["execution_scope"]!="mvp" or cfg["execute_remediation"] or cfg["external_llm_enabled"]:
        raise ValueError("This implementation requires mvp, no remediation and no external LLM")
    cmd=args.command
    if cmd in ["audit-data","prepare-data","make-splits"]:
        from .data import audit,prepare,make_splits
        if cmd=="audit-data": audit(cfg["data_dir"])
        if cmd=="prepare-data": prepare(cfg["data_dir"])
        if cmd=="make-splits": make_splits(cfg["data_dir"],cfg["seed"])
    elif cmd=="train":
        from .training import train
        print(train(cfg,args.mode)["history"])
    elif cmd=="serve":
        import uvicorn
        from .api import create_app
        uvicorn.run(create_app(args.artifact),host="127.0.0.1",port=args.port)
    else:
        from . import pipeline
        if cmd=="baseline": result=pipeline.run_baseline(cfg["data_dir"],Path(cfg["artifact_root"])/"baseline")
        elif cmd=="calibrate": result=pipeline.calibrate(args.artifact,args.device)
        elif cmd=="select-policy": result=pipeline.policy(args.artifact,args.device)
        elif cmd=="evaluate": result=pipeline.evaluate(args.artifact,args.device)
        else: result=pipeline.benchmark(args.artifact,cfg)
        if cmd in ["evaluate","baseline"]:
            print({"root":result["root"],"fault_accuracy":result["fault"]["accuracy"],"joint":result["joint_accuracy"],"selective":result["selective"]})
        else:
            print({k:v for k,v in result.items() if k not in ["measurements","scaling_probes"]})

# AgenticOpsEval

![Overview](overview.png)

A multi-dataset benchmark for evaluating LLM agents in microservice failure diagnosis. This repository bundles the two datasets introduced in *"A Multi-Dataset Benchmark for Evaluating LLM Agents in Microservice Failure Diagnosis"*: **503 expert-labeled failure cases**, **~15.3 GB** of multimodal observability data, validated through **6,093 teams** across two 2025 national-level competitions.

Both datasets evaluate agents along three shared pillars — **Localization** (where the fault is), **Identification** (what fault type it is), and **Reason** (whether the reasoning trace is grounded in the right diagnostic evidence) — using two complementary labelling forms: per-modality **key-evidence** (AIOps2025) and typed **causal chains** (RCA100).

## Layout

```
agenticopseval/
├── README.md
├── overview.png                 benchmark overview figure (shown above)
├── AIOps2025/                   2025 CCF AIOps Challenge dataset (561 teams)
│   ├── README.md
│   ├── input.json               (400 anomaly descriptions + time windows)
│   ├── groundtruth.jsonl        (400 per-modality key-evidence labels)
│   ├── checksums.md5
│   └── 2025-06-DD.tar.gz × 18   (per-day metrics/logs/traces parquet archives)
└── RCA100/                      RCA-100 v1.1; 2025 Alibaba Cloud Tianchi AIOps dataset (5,532 teams)
    ├── README.md, LICENSE, manifest.txt, summary.json
    ├── cases/t001..t103/        (per-task metrics/logs/traces/events/alerts/topology)
    └── answer_key/              (per-case four-layer ground truth + t→case_id mapping)
```

## At a glance

| Dimension | AIOps2025 | RCA100 |
|---|---|---|
| Scale | 400 cases / 11.9 GB | 103 cases / 3.43 GB |
| System | HipsterShop + TiDB + Redis on self-hosted K8s | OpenTelemetry Demo Store on Alibaba Cloud ACK |
| Modalities | 3 (Metrics / Logs / Traces) | 6 (+ Events / Alerts / Topology) |
| Input form | open-NL anomaly description + time window | structured alert event |
| Taxonomy | 9 categories × 18 fault types × 3 entity levels | 28 root-cause types → 6 semantic groups |
| Reasoning label | per-modality key-evidence list | typed causal chain + per-step observability checkpoints |
| Scoring | LA / TA / Exp. / Eff. at 0.40 / 0.40 / 0.10 / 0.10 | Entity / Fault / Process at 0.40 / 0.30 / 0.30 |

See [`AIOps2025/README.md`](AIOps2025/README.md) and [`RCA100/README.md`](RCA100/README.md) for full schemas, ground-truth formats, and scoring details.

## Large-scale validation

| Competition | Dataset | Teams |
|---|---|---|
| 2025 CCF AIOps Challenge | AIOps2025 | 561 |
| Alibaba Cloud Tianchi 2025 AI-Native Programming Challenge | RCA100 | 5,532 |
| **Total** | | **6,093** |

To the best of our knowledge, this is the largest competition-validated dataset for microservice reasoning-process evaluation to date.

## Beyond RCA

Both datasets are reusable beyond microservice RCA: time-aligned Metrics+Logs+Traces support multimodal time-series anomaly detection; RCA100's causal-chain labels and entity-relation graphs supply ground truth for causal discovery and graph neural methods; AIOps2025's per-modality key-evidence labels are rare material for studying expert–agent reasoning-path divergence and agent self-evaluation.

## Citation

If you use AgenticOpsEval in your research, please cite the accompanying paper ([arXiv:2606.29193](https://arxiv.org/abs/2606.29193)):

```bibtex
@misc{cai2026agenticopseval,
  title        = {A Multi-Dataset Benchmark for Evaluating LLM Agents in Microservice Failure Diagnosis},
  author       = {Cai, Yuanhong and Nie, Xiaohui and Yin, Kanglin and Pei, Changhua and Sun, Yongqian and Zhang, Shenglin and Liu, Haibin and Liu, Guiyang and Wen, Xidao and Situ, Fang and Pei, Dan},
  year         = {2026},
  eprint       = {2606.29193},
  archivePrefix= {arXiv},
  primaryClass = {cs.SE},
  doi          = {10.48550/arXiv.2606.29193},
  url          = {https://arxiv.org/abs/2606.29193}
}
```

## License

- **AIOps2025**: CC BY-NC 4.0
- **RCA100**: CC BY-NC-SA 4.0 (see `RCA100/LICENSE`)

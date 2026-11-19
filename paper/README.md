# Paper: Frozen Foundation Features for One-Shot VFL on Satellite Imagery

LaTeX sources (Springer LNCS) are in `src/`; every number, table and figure comes from the JSON files in
`../paper_results/`, through `../paper_experiments/make_assets.py`.

## Build the PDF

```bash
python paper_experiments/make_assets.py paper_results paper/src/gen   # tables, figures, macros
cd paper/src && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Re-run the experiments (one GPU, Google Colab or any CUDA machine)

| Script | What it produces | Time |
|---|---|---|
| `s1_vertibench_features.py` | VertiBench download, frozen features of 8 encoder configurations, thumbnails | 75 min (L4) |
| `s2_vertibench_heads.py` | encoders, PCA/quantisation Pareto, parties, noise (`vb_*.json`) | 17 min (L4) |
| `s3_baselines.py pack / splitnn / ssl` | full-resolution SplitNN and FedOnce-style SimCLR baselines | 6-11 min per SplitNN run, 10-16 min SimCLR (A100) |
| `s4_inversion.py` | feature-inversion attack (`vb_inversion*.{json,npz}`) | 30 min (L4) |
| `s5_worldstrat.py` | WorldStrat random vs spatial splits (`ws_results.json`) | 5 min |
| `s6_timing.py` | encoder throughput (`vb_timing_<GPU>.json`) | 5 min |

`run_chain.sh A|B|C` chains them on a fresh VM (A: s5, s1, s2, s4, s6; B: baselines; C: RGB baselines + timing).
`colab_ctl.py`, `vbctl.sh` and `keepalive.sh` drive a Colab runtime from a remote shell through the `colab` CLI
(Colab Pro reclaims runtimes that have no client activity for about 30 minutes; `keepalive.sh` touches the runtime,
harvests results and stops it when the chain is done).

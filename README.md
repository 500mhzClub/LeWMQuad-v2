# LeWMQuad-v2

LeWorldModel for quadruped navigation — v2 with clean training data.

## What changed from v1

The v1 pipeline produced corrupted training data: the robot's egocentric camera
(mounted 0.10m forward of the base) would clip through 0.06m-thick walls during
collisions, rendering views *through* the geometry. This poisoned the JEPA's
latent space and made all downstream planning unreliable near walls.

### Three-layer anti-clipping fix

| Layer | v1 | v2 | Effect |
|-------|----|----|--------|
| Wall thickness | 0.06m | **0.20m** | Camera physically cannot reach through |
| Camera near-plane | Genesis default (~0.01m) | **0.08m** | Near geometry is rendered, not skipped |
| Camera forward offset | 0.10m | **0.06m** | Camera stays within the robot body envelope |
| Corridor width | 0.30–0.55m | **0.50–0.70m** | Navigable with thicker walls |
| Frame validation | None | **Camera-in-wall detector** | Replaces any remaining clipped frames |

### What did NOT change

- **LeWM architecture**: ViT-Tiny encoder, Transformer predictor with AdaLN, SIGReg regulariser
- **Training paradigm**: End-to-end, no EMA, no target encoder, single λ hyperparameter
- **PPO low-level controller**: Same ActorCritic(obs_dim=50, act_dim=12)
- **Data pipeline structure**: Stage 1 physics rollout → Stage 2 visual rendering → HDF5

## Key design decision: keep collisions in training data

We do NOT filter out collision frames. If we did, the world model would never
learn what happens near walls and would be out-of-distribution during inference
when the robot inevitably approaches obstacles. Instead, we ensure the camera
*correctly renders wall surfaces* during collisions, so the model learns
"approaching wall → wall fills the view" — the correct signal for planning.

## Pipeline

```bash
# 1. Collect physics trajectories
python scripts/1_physics_rollout.py --ckpt <ppo_ckpt> --steps 1000 --chunks 5

# 2. Render egocentric RGB (with anti-clipping protections)
python scripts/2_visual_renderer.py --raw_dir jepa_raw_data --out_dir jepa_final_dataset

# 3. Train LeWorldModel
python scripts/3_train_lewm.py --data_dir jepa_final_dataset --sigreg_lambda 0.09

# 4. Train energy head (optional, for planning)
python scripts/4_train_energy_head.py --data_dir jepa_final_dataset --checkpoint lewm_checkpoints/latest.pt
```

## Validation scripts

```bash
# Static clipping test: renders camera at various wall distances
python scripts/demo_no_clipping.py

# End-to-end data quality: runs mini rollout and counts clipped frames
python scripts/demo_data_quality.py --ckpt <ppo_ckpt>
```

## Inference Notes

### Score-space ablation

The inference script supports three score spaces:

- `mixed`: raw observation / memory anchors with projected rollout predictions
- `raw`: raw anchors with raw rollout predictions
- `proj`: projected anchors with projected rollout predictions

In short 400-step maze runs, `mixed` has been the strongest baseline so far.
The likely reason is architectural:

- Raw encoder states preserve more local place detail for anchors such as
  breadcrumbs and visited-state memory.
- Projected rollout latents are the space the predictor is actually trained to
  match at the next step.
- True `raw` scoring underperforms because raw rollout latents are not directly
  supervised by the training loss.
- True `proj` scoring underperforms because cosine geometry in projector space
  is not guaranteed to be a good planning metric even if it is good for the
  JEPA training objective.

### Novelty failure mode

The original novelty signal in `scripts/6_infer_pure_wm.py` used a sliding
recent-history window and compared full `H`-step rollout windows after
flattening them to a single vector. That caused three problems:

- The planner forgot older regions once the fixed recent-history buffer rolled
  over.
- Whole-window cosine similarity was too blunt; small pose and gait changes
  could make a familiar corridor look novel enough.
- The agent had no persistent notion of "already explored area", so it could
  return to old regions with little or no penalty.

The current novelty implementation fixes this without adding oracle geometry or
handwritten maze heuristics:

- Visited states are stored in a persistent novelty bank using reservoir
  sampling, so memory spans the full run instead of just the recent past.
- Novelty is computed per predicted step against that bank, then averaged over
  the rollout horizon. This is sharper than comparing a flattened `H*D` window.

### Coverage diagnostics

Exploration quality should not be judged by a discrete "visit every cell"
target. The inference script now exports a soft coverage diagnostic instead:

- `coverage_map.png` renders a continuous heatmap over the map showing where the
  robot has meaningfully expanded into new space.
- `summary.json` records `soft_coverage_area_m2` and
  `soft_coverage_gain_per_m`.
- Stdout logs also report live coverage growth as `cov=...m^2` and `cov+=...`.

This is diagnostic only. It does not affect the planner or add any non-visual
oracle input to the policy.

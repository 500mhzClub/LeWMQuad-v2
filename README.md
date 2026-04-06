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

## Current Training Approach

The original `seq_len=4` contiguous setup was not paper-faithful in time. It
trained on four adjacent low-level control steps, which is too short to encode
meaningful maze progress. The paper's "4-step" setup is four observations over
four **blocks of 5 actions**.

The current recommended regime therefore keeps `seq_len=4`, but changes the
timescale:

- `temporal_stride=5`: sample observations every 5 raw control steps
- `action_block_size=5`: aggregate each 5-step command block into one macro-action
- `window_stride=5`: use overlapping macro-windows instead of only non-overlapping starts
- `macro_action_repeat=5` at inference: hold each planner command for 5 low-level control cycles

Why it changed:

- It matches the paper's temporal abstraction much more closely than contiguous
  4-step training.
- It gives the predictor transitions that actually matter for maze navigation:
  corridor progress, wall approach, turns, and local basin exits.
- `window_stride=5` uses far more valid macro-trajectories than `window_stride=20`,
  so the model sees many more distinct starts instead of repeating the same sparse
  subset.
- `--use_proprio` is now the recommended setting for navigation. Proprio is
  runtime-valid on the robot and helps short-horizon progress, stuck detection,
  and dead-reckoning.

One implementation detail matters in practice: overlap training (`window_stride=5`)
increases HDF5 pressure significantly, so the current stable settings are:

- `--num_workers 4`
- `--prefetch_factor 2`

Those settings are baked into the commands below.

## Recommended Pipeline

### 1. Optional: collect fresh physics trajectories

```bash
python scripts/1_physics_rollout.py --ckpt <ppo_ckpt> --steps 1000 --chunks 5
```

### 2. Optional: render egocentric RGB with anti-clipping protections

```bash
python scripts/2_visual_renderer.py --raw_dir jepa_raw_data --out_dir jepa_final_dataset
```

### 3. Train the base LeWorldModel

Current recommended run:

```bash
python scripts/3_train_lewm.py \
  --data_dir jepa_final_v3 \
  --device cuda \
  --epochs 30 \
  --batch_size 128 \
  --seq_len 4 \
  --temporal_stride 5 \
  --action_block_size 5 \
  --window_stride 5 \
  --num_workers 4 \
  --prefetch_factor 2 \
  --sigreg_lambda 0.09 \
  --use_proprio \
  --out_dir lewm_checkpoints_keyframe_exec_stride5_overlap_v2 \
  --log_dir lewm_logs_keyframe_exec_stride5_overlap_v2
```

This is the main world-model run to build on. Older runs such as contiguous
`seq_len=4` or stride-5 with `window_stride=20` should be treated as baselines,
not the preferred checkpoint.

### 4. Train the planning heads

The planning stack uses four learned components on top of frozen LeWM latents:

- `LatentEnergyHead`: safety / mobility consequence energy
- `GoalEnergyHead`: beacon identity-conditioned attraction
- `ProgressEnergyHead`: short-horizon improvement toward the correct goal
- `ExplorationBonus`: RND-style novelty bonus

All of them must be trained with the **same temporal abstraction** as the world
model checkpoint they consume.

Recommended command:

```bash
python scripts/4_train_energy_head.py \
  --data_dir jepa_final_v3 \
  --checkpoint lewm_checkpoints_keyframe_exec_stride5_overlap_v2/epoch_30.pt \
  --device cuda \
  --seq_len 4 \
  --temporal_stride 5 \
  --action_block_size 5 \
  --window_stride 5 \
  --num_workers 4 \
  --prefetch_factor 2 \
  --use_proprio \
  --out_dir energy_head_checkpoints_keyframe_exec_stride5_overlap_v2 \
  --log_dir energy_head_logs_keyframe_exec_stride5_overlap_v2 \
  --safety_mode consequence \
  --epochs 10 \
  --goal_epochs 10 \
  --progress_epochs 5 \
  --exploration_epochs 5 \
  --goal_weight 1.0 \
  --progress_weight 1.0 \
  --exploration_weight 0.1
```

This writes a combined scorer checkpoint to:

- `energy_head_checkpoints_keyframe_exec_stride5_overlap_v2/trajectory_scorer.pt`

### 5. Run world-model inference

Inference must use the same macro-timescale as training:

- planner horizon in latent steps
- `macro_action_repeat=5` so one planner step corresponds to 5 low-level control steps

Recommended command:

```bash
python scripts/6_infer_pure_wm.py \
  --ppo_ckpt models/ppo/ckpt_20000.pt \
  --wm_ckpt lewm_checkpoints_keyframe_exec_stride5_overlap_v2/epoch_30.pt \
  --scorer_ckpt energy_head_checkpoints_keyframe_exec_stride5_overlap_v2/trajectory_scorer.pt \
  --target_beacon red \
  --seed 42 \
  --steps 800 \
  --macro_action_repeat 5 \
  --score_space mixed \
  --out_dir inference_runs/keyframe_exec_stride5_overlap_v2_s42
```

`steps=800` is intentional: with `macro_action_repeat=5`, that corresponds to
about `4000` low-level control steps.

### 6. Multi-seed evaluation

```bash
for seed in 42 43 44; do
  python scripts/6_infer_pure_wm.py \
    --ppo_ckpt models/ppo/ckpt_20000.pt \
    --wm_ckpt lewm_checkpoints_keyframe_exec_stride5_overlap_v2/epoch_30.pt \
    --scorer_ckpt energy_head_checkpoints_keyframe_exec_stride5_overlap_v2/trajectory_scorer.pt \
    --target_beacon red \
    --seed "$seed" \
    --steps 800 \
    --macro_action_repeat 5 \
    --score_space mixed \
    --out_dir "inference_runs/keyframe_exec_stride5_overlap_v2_s${seed}"
done
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

### Why the planner changed

The current inference stack is not a plain "goal cosine + novelty" controller.
It is a short-horizon keyframe executive built to work with temporally-abstract
LeWM rollouts.

What it does:

- keeps a persistent prototype bank for frontier density / novelty
- builds an immutable keyframe graph from runtime-valid signals
- scores short rollouts by:
  - safety energy
  - goal progress
  - learned progress bonus
  - frontier density
  - route progress
- repeats each planner command for 5 low-level control cycles

This is why training and inference must agree on macro-timescale. Training on
stride-5 macro-transitions and then planning at raw 1-step control frequency
would miscalibrate the predictor.

### Novelty / frontier memory

The current planner no longer depends on a tiny recent-history window.

- Visited states are stored in a persistent prototype bank.
- Frontier pressure comes from latent density against that bank.
- Keyframe routing gives the executive a short-hop topological memory without
  using simulator geometry in the control loop.

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

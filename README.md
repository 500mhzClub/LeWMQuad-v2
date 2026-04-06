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
- `--use_proprio` is only recommended for mixed-state navigation experiments.
  Do not use it for pure-perception exploration: the latent will encode body
  state as well as vision, so turning or circling can appear "novel" even when
  the scene has barely changed.

One implementation detail matters in practice: overlap training (`window_stride=5`)
increases HDF5 pressure significantly, so the current stable settings are:

- `--num_workers 4`
- `--prefetch_factor 2`

Those settings are baked into the commands below.

### Known caveat: macro-action fidelity

The current stride-5 training setup is a useful macro-action abstraction, but it
is still an approximation.

- The dataset loader aggregates each 5-step block into a single macro-action by
  averaging the logged commands over that block.
- The rollout logs store the nominal `scaled_cmds`, not the latency-buffered
  `active_cmds` that the PPO policy actually consumed after the 2-step command
  delay in the physics rollout.

Implication:

- The model is learning "what usually happens under the average command over
  this 5-step block," not the exact executed low-level command sequence.
- This is acceptable when commands are smooth or piecewise constant, which is
  usually true in the current OU and structured-pattern data.
- It is a real weakness if within-block commands vary rapidly, because different
  executed sequences can collapse to similar averaged macro-actions.

What this means for the current project:

- The existing data is still good enough for the current training and first
  inference evaluation.
- If the stride-5 overlap model still fails in a way that suggests action-model
  mismatch, the next fix is **not** to abandon the architecture. The next fix is
  to improve command-block fidelity.

Important correction:

- New rollout collection is **not required** just to recover the executed command
  stream.
- The physics rollout applies a deterministic 2-step latency buffer to the saved
  `scaled_cmds`, so `active_cmds` can be reconstructed offline from the existing
  trajectories.
- The real current limitation is the predictor interface: one model step still
  expects a single `3D` command vector, so the 5-step action block must be
  summarized somehow unless the model is extended.

### Recommended upgrade path for action-block fidelity

This can be done with the data already on disk.

1. Reconstruct executed `active_cmds` offline from the saved `scaled_cmds`.
   The latency buffer is deterministic and zero-initialized, so each
   environment's executed command sequence can be replayed exactly from the
   saved command stream.
2. Replace `mean(scaled_cmd_block)` with `mean(active_cmd_block)` as the first
   drop-in improvement.
   This keeps the predictor interface unchanged (`cmd_dim=3`) while making the
   macro-action summary match what PPO actually consumed.
   The repo now exposes this as `--command_representation mean_active`.
3. If that is still not enough, move from a summarized macro-action to an
   explicit 5-step action block.
   The cleanest version is to flatten the block to `5 x 3 = 15` dimensions or
   add a small action-block encoder, then feed that representation into the
   predictor instead of a single averaged command.
   The repo now exposes the flattened-block path as
   `--command_representation active_block`.
4. Retrain the base world model and planning heads with the improved action
   representation.
   This does **not** require new raw rollouts or new rendered vision. Existing
   rollout/HDF5 data is sufficient.

Practical recommendation:

- First complete the current stride-5 overlap run and evaluate it.
- If inference still looks limited by action abstraction, implement step 2
  first.
- Only then escalate to step 3, which is a real model-interface change rather
  than a dataset fix.

### Deferred patch plan: full action-block fidelity

This is the concrete follow-up if the current `mean(command_block)` run still
looks action-abstraction-limited. It is intentionally deferred until the
current overlap model has been evaluated.

Goal:

- keep the current stride-5 temporal abstraction
- stop collapsing each 5-step block to a single averaged command
- feed the predictor a representation of the full executed action block instead

Why this is the next patch:

- the paper's planning stack uses `action_block=5` and optimizes full action
  blocks rather than a single averaged command
- our current setup already matches the paper in temporal spacing
- the main remaining mismatch is command-block representation

Implementation plan:

1. Reconstruct `active_cmds` offline from the saved `scaled_cmds`.
   Do this inside the dataset/label extraction path so the exact executed
   command stream is available without recollecting rollouts.
   Files to change:
   - `lewm/data/streaming_dataset.py`
   - optionally `scripts/2_visual_renderer.py` if you want to cache the
     reconstructed blocks into HDF5

2. Add a drop-in improved baseline: `mean(active_cmd_block)`.
   Keep the predictor interface unchanged and simply replace
   `mean(scaled_cmd_block)` with `mean(active_cmd_block)`.
   This is the lowest-risk correction and should be tried before changing model
   dimensions.
   Files to change:
   - `lewm/data/streaming_dataset.py`
   - `scripts/4_train_energy_head.py` if any head datasets assume the old block
     summary

3. Add explicit full-block inputs to the predictor.
   Replace one `3D` macro-action per model step with either:
   - a flattened `15D` block (`5 x 3`), or
   - a small action-block encoder that maps `(5, 3) -> D_action`
   The action-block encoder is cleaner if you expect to change block size
   later; the flattened `15D` version is simpler to patch first.
   Files to change:
   - `lewm/models/predictor.py`
   - `lewm/models/lewm.py`
   - `scripts/3_train_lewm.py`
   - `scripts/4_train_energy_head.py`
   - `scripts/6_infer_pure_wm.py`

4. Update inference to plan over full action blocks.
   The planner should sample one action block per model step and execute it as
   five low-level commands, rather than sampling one command and repeating it.
   This is the inference-side match to step 3.
   Files to change:
   - `scripts/6_infer_pure_wm.py`

5. Retrain the stack in order.
   Required order:
   - retrain base LeWM
   - retrain planning heads
   - rerun inference

Suggested rollout of the patch:

- Phase A: implement only `mean(active_cmd_block)` and rerun training
- Phase B: if still needed, implement explicit `5 x 3` block inputs
- Phase C: only after that consider recollecting data, and only if you want to
  cache executed actions directly rather than reconstruct them

Concrete run sequence for Phase A:

1. Patch the dataset to reconstruct `active_cmds` and replace
   `mean(scaled_cmd_block)` with `mean(active_cmd_block)`.
2. Retrain LeWM into fresh dirs:

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
  --command_representation mean_active \
  --use_proprio \
  --out_dir lewm_checkpoints_keyframe_exec_stride5_active_mean \
  --log_dir lewm_logs_keyframe_exec_stride5_active_mean
```

3. Retrain heads into fresh dirs:

```bash
python scripts/4_train_energy_head.py \
  --data_dir jepa_final_v3_los \
  --checkpoint lewm_checkpoints_keyframe_exec_stride5_active_mean/epoch_30.pt \
  --device cuda \
  --seq_len 4 \
  --temporal_stride 5 \
  --action_block_size 5 \
  --window_stride 5 \
  --num_workers 4 \
  --prefetch_factor 2 \
  --out_dir energy_head_checkpoints_keyframe_exec_stride5_active_mean \
  --log_dir energy_head_logs_keyframe_exec_stride5_active_mean \
  --safety_mode consequence \
  --epochs 10 \
  --goal_epochs 10 \
  --progress_epochs 5 \
  --exploration_epochs 5 \
  --goal_weight 1.0 \
  --progress_weight 1.0 \
  --exploration_weight 0.1
```

4. Evaluate:

```bash
python scripts/6_infer_pure_wm.py \
  --ppo_ckpt models/ppo/ckpt_20000.pt \
  --wm_ckpt lewm_checkpoints_keyframe_exec_stride5_active_mean/epoch_30.pt \
  --scorer_ckpt energy_head_checkpoints_keyframe_exec_stride5_active_mean/trajectory_scorer.pt \
  --target_beacon red \
  --seed 42 \
  --steps 800 \
  --macro_action_repeat 5 \
  --score_space mixed \
  --out_dir inference_runs/keyframe_exec_stride5_active_mean_s42
```

Concrete run sequence for Phase B:

1. Patch the predictor to accept a full `5 x 3` action block per model step.
2. Retrain LeWM in fresh dirs, for example:

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
  --command_representation active_block \
  --use_proprio \
  --out_dir lewm_checkpoints_keyframe_exec_stride5_block15 \
  --log_dir lewm_logs_keyframe_exec_stride5_block15
```

3. Retrain heads and rerun inference in matching fresh dirs.

Decision rule:

- If the current averaged-block model solves the maze reliably enough, keep it.
- If it fails in a way that looks like wrong within-block action semantics,
  apply Phase A first.
- Only move to Phase B if Phase A still leaves a clear action-fidelity gap.

## Recommended Pipeline

### 1. Optional: collect fresh physics trajectories

```bash
python scripts/1_physics_rollout.py --ckpt <ppo_ckpt> --steps 1000 --chunks 5
```

### 2. Optional: render egocentric RGB with anti-clipping protections

```bash
python scripts/2_visual_renderer.py --raw_dir jepa_raw_data --out_dir jepa_final_dataset
```

For the current navigation stack, there is one important label update:

- beacon visibility / identity / range for the planning heads should be
  obstacle-aware, not just FOV+range
- `scripts/2_visual_renderer.py` now recomputes those geometry-derived labels by
  default when building HDF5 from raw rollouts
- if you already have a rendered HDF5 dataset, you can reuse its existing vision
  and rewrite only the labels into a fresh output directory:

```bash
python scripts/2_visual_renderer.py \
  --raw_dir jepa_raw_data_v3 \
  --reuse_vision_from jepa_final_v3 \
  --out_dir jepa_final_v3_los \
  --workers 1
```

This does **not** require a new physics rollout. It uses the existing raw
rollout state to recompute LOS-aware labels and copies the already-rendered
vision from `jepa_final_v3`.

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

The head-training script now auto-detects encoder config from the LeWM
checkpoint and will fail fast on explicit mismatches. In particular:

- `latent_dim`, `image_size`, `patch_size`, and `use_proprio` are resolved from
  the checkpoint
- scorer checkpoints now carry temporal / encoder metadata
- inference validates that scorer metadata matches the loaded world model and
  `macro_action_repeat`

Important: the base world model does **not** use beacon labels, so it can still
be trained on `jepa_final_v3`. The planning heads should use the LOS-corrected
dataset (for example `jepa_final_v3_los`) so the goal/progress supervision does
not reward through-wall visibility.

Recommended command:

```bash
python scripts/4_train_energy_head.py \
  --data_dir jepa_final_v3_los \
  --checkpoint lewm_checkpoints_keyframe_exec_stride5_overlap_v2/epoch_30.pt \
  --device cuda \
  --seq_len 4 \
  --temporal_stride 5 \
  --action_block_size 5 \
  --window_stride 5 \
  --num_workers 4 \
  --prefetch_factor 2 \
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

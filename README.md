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

For pure-perception navigation, the current recommended regime is:

- `seq_len=4`
- `temporal_stride=5`
- `action_block_size=5`
- `window_stride=5`
- `command_representation=active_block`
- no `--use_proprio`
- `macro_action_repeat=5` and `mpc_execute=5` at inference

This matches the paper much more closely than the old contiguous setup:

- each model step spans a full block of five low-level commands
- the predictor consumes the executed command block instead of a single repeated command
- overlap training (`window_stride=5`) provides many more valid macro-trajectories than sparse non-overlapping windows
- pure-vision latents keep exploration tied to scene change instead of body-state change

Stable data-loader settings on the current 224px HDF5 pipeline are still:

- `--num_workers 4`
- `--prefetch_factor 2`

### Command-Block Support

The rollout logs still store nominal `scaled_cmds`, but the physics rollout uses
a deterministic zero-initialized 2-step latency buffer before PPO consumes those
commands. The repo now reconstructs the executed `active_cmds` offline from the
saved stream, so new rollout collection is **not** required to improve
action-block fidelity.

Three command representations are supported:

- `mean_scaled`: legacy baseline; average the stored nominal commands in each block
- `mean_active`: improved baseline; average the reconstructed executed commands in each block
- `active_block`: paper-aligned path; flatten the full executed `5 x 3` command block to `15D`

Recommendation:

- use `active_block` for the main paper-aligned pure-perception run
- keep `mean_active` only as a lower-risk ablation or compatibility baseline
- avoid `mean_scaled` except for reproducing older checkpoints

## Recommended Pipeline

### 1. Optional: collect fresh physics trajectories

```bash
python3 scripts/1_physics_rollout.py --ckpt <ppo_ckpt> --steps 1000 --chunks 5
```

### 2. Optional: render egocentric RGB with anti-clipping protections

```bash
python3 scripts/2_visual_renderer.py --raw_dir jepa_raw_data --out_dir jepa_final_dataset
```

For the current navigation stack, there is one important label update:

- beacon visibility / identity / range for the planning heads should be
  obstacle-aware, not just FOV+range
- `scripts/2_visual_renderer.py` now recomputes those geometry-derived labels by
  default when building HDF5 from raw rollouts
- if you already have a rendered HDF5 dataset, you can reuse its existing vision
  and rewrite only the labels into a fresh output directory:

```bash
python3 scripts/2_visual_renderer.py \
  --raw_dir jepa_raw_data_v3 \
  --reuse_vision_from jepa_final_v3 \
  --out_dir jepa_final_v3_los \
  --workers 1
```

This does **not** require a new physics rollout. It uses the existing raw
rollout state to recompute LOS-aware labels and copies the already-rendered
vision from `jepa_final_v3`.

- In `--reuse_vision_from` mode, `--workers` does not materially accelerate the
  label rebuild. The script recomputes labels in-process and skips multi-worker
  rendering because vision is being copied, not regenerated.
- If the output directory already exists and you want to overwrite it in place,
  add `--force`.

### 3. Train the base LeWorldModel

Current recommended pure-perception run:

```bash
python3 scripts/3_train_lewm.py \
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
  --out_dir lewm_checkpoints_keyframe_exec_stride5_block15_vision \
  --log_dir lewm_logs_keyframe_exec_stride5_block15_vision
```

This is the main world-model run to build on:

- no `--use_proprio`: the recommended navigation stack is now pure vision
- `--command_representation active_block`: the predictor consumes the executed
  five-step command block, not an averaged surrogate
- checkpoints save command metadata (`cmd_dim`, representation, latency, block
  size), and downstream scripts read that metadata automatically

Older mixed-state or averaged-command runs should be treated as baselines or
compatibility checkpoints, not the preferred starting point.

### 4. Train the planning heads

The current recommended planning stack for pure-perception exploration uses two
learned components on top of frozen LeWM latents:

- `LatentEnergyHead`: dense proximity-style safety energy
- `ExplorationBonus`: RND-style novelty bonus

Goal and progress heads remain available as optional experiments, but they are
not part of the active default runtime path.

The planning heads must use the **same temporal abstraction and command
representation** as the world-model checkpoint they consume.

The head-training script now auto-detects encoder config from the LeWM
checkpoint and will fail fast on explicit mismatches. In particular:

- `latent_dim`, `image_size`, `patch_size`, `use_proprio`, `cmd_dim`,
  command representation, and command latency are resolved from the checkpoint
- scorer checkpoints now carry temporal / encoder metadata
- inference validates that scorer metadata matches the loaded world model

Important: the base world model does **not** use beacon labels, so it can still
be trained on `jepa_final_v3`. The planning heads should use the LOS-corrected
dataset (`jepa_final_v3_los`) so safety / visibility labels do not reward
through-wall shortcuts.

Recommended command:

```bash
python3 scripts/4_train_energy_head.py \
  --data_dir jepa_final_v3_los \
  --checkpoint lewm_checkpoints_keyframe_exec_stride5_block15_vision/epoch_30.pt \
  --device cuda \
  --seq_len 4 \
  --temporal_stride 5 \
  --action_block_size 5 \
  --window_stride 5 \
  --num_workers 4 \
  --prefetch_factor 2 \
  --out_dir energy_head_checkpoints_keyframe_exec_stride5_block15_vision \
  --log_dir energy_head_logs_keyframe_exec_stride5_block15_vision \
  --safety_mode proximity \
  --epochs 10 \
  --skip_goal \
  --skip_progress \
  --exploration_epochs 5 \
  --exploration_weight 1.0
```

This writes a combined scorer checkpoint to:

- `energy_head_checkpoints_keyframe_exec_stride5_block15_vision/trajectory_scorer.pt`

### 5. Run world-model inference

Inference must use the same macro-timescale as training:

- planner horizon in latent steps
- `macro_action_repeat=5` so one planner step corresponds to one executed
  five-command block
- `mpc_execute=5` to match the paper-style cadence of executing the optimized
  block sequence before replanning

Recommended command:

```bash
python3 scripts/6_infer_pure_wm.py \
  --ppo_ckpt models/ppo/ckpt_20000.pt \
  --wm_ckpt lewm_checkpoints_keyframe_exec_stride5_block15_vision/epoch_30.pt \
  --scorer_ckpt energy_head_checkpoints_keyframe_exec_stride5_block15_vision/trajectory_scorer.pt \
  --seed 42 \
  --steps 200 \
  --macro_action_repeat 5 \
  --mpc_execute 5 \
  --cem_iters 10 \
  --rnd_online_lr 1e-3 \
  --out_dir inference_runs/perception_only_stride5_block15_vision_s42
```

Notes:

- Omit `--target_beacon` unless you know that beacon exists for the chosen
  random seed. If you pass an invalid beacon identity, the script will error.
- The active planner now uses learned safety + learned novelty + action penalty.
  Beacon similarity is still logged and checked for success, but it is not used
  as a planner cost.
- The script now rejects proprio-enabled world models by default. To run an old
  mixed-state checkpoint for comparison, pass `--allow_mixed_latent_wm`.

### 6. Multi-seed evaluation

```bash
for seed in 42 43 44; do
  python3 scripts/6_infer_pure_wm.py \
    --ppo_ckpt models/ppo/ckpt_20000.pt \
    --wm_ckpt lewm_checkpoints_keyframe_exec_stride5_block15_vision/epoch_30.pt \
    --scorer_ckpt energy_head_checkpoints_keyframe_exec_stride5_block15_vision/trajectory_scorer.pt \
    --seed "$seed" \
    --steps 200 \
    --macro_action_repeat 5 \
    --mpc_execute 5 \
    --cem_iters 10 \
    --rnd_online_lr 1e-3 \
    --out_dir "inference_runs/perception_only_stride5_block15_vision_s${seed}"
done
```

## Validation scripts

```bash
# Static clipping test: renders camera at various wall distances
python3 scripts/demo_no_clipping.py

# End-to-end data quality: runs mini rollout and counts clipped frames
python3 scripts/demo_data_quality.py --ckpt <ppo_ckpt>
```

## Inference Notes

### Active objective

The active inference path is intentionally narrow:

- learned safety cost
- learned novelty bonus
- small action penalty

The planner no longer uses goal energy, learned progress, keyframe routing,
prototype-bank frontier density, or other route-level heuristics in the control
loop. Beacon similarity is still encoded and logged, but it is treated as an
evaluation / success signal rather than a planning reward.

### Pure-perception guard

Pure-perception only really starts once the world-model latent is pure vision.

- If a checkpoint was trained with `--use_proprio`, the latent still contains
  body-state information and novelty can be satisfied by circling or posture
  change even when the view is not meaningfully changing.
- `scripts/6_infer_pure_wm.py` now rejects proprio-enabled world models by
  default.
- To run those older checkpoints as mixed-state ablations, pass
  `--allow_mixed_latent_wm`.

### Macro-action fidelity

The repo now supports paper-aligned executed command blocks end-to-end.

- The raw rollout logs still store nominal `scaled_cmds`.
- The training dataset reconstructs the actually executed `active_cmds` using
  the deterministic 2-step command-latency buffer from rollout.
- `active_block` flattens the full executed `5 x 3` block to a 15D macro-action.
- During inference, `active_block` checkpoints cause the planner to sample full
  15D blocks and execute those low-level commands sequentially.

Legacy command modes remain available for comparison:

- `mean_scaled`: average nominal commands in each block
- `mean_active`: average reconstructed executed commands in each block

### Compatibility flags

Some older CLI flags remain for backwards compatibility and experiment logging,
but do not affect the active planner:

- `--score_space`
- revisit / forward-bonus controls
- goal-activation controls tied to the old goal-seeking runtime

The script records these values in `summary.json` as ignored settings so older
run configurations remain auditable.

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

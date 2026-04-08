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
- `macro_action_repeat=5` and `mpc_execute=1` at inference

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
python3 scripts/1_physics_rollout.py \
  --ckpt <ppo_ckpt> \
  --steps 1000 \
  --chunks 200 \
  --scene_distribution mixed
```

Important:

- `--chunks 5` is only a smoke-test scale
- for any serious generalization claim, collect many more chunk seeds so the
  dataset contains many more distinct scenes, not just more frames from the
  same few layouts
- `--scene_distribution mixed` now injects enclosed mazes into training data so
  rollout collection better matches the deployment task instead of relying only
  on the older local composite scenes
- if you want a pure deployment-like curriculum, use
  `--scene_distribution enclosed`
- `--command_policy mixed` is now the recommended collector mode for serious
  maze training: it preserves some open-loop diversity but also injects
  privileged closed-loop maze-teacher trajectories that actually traverse to
  beacons and dead ends
- rollout collection now uses topology-aware respawns by default (`--respawn_strategy auto`),
  so enclosed-maze episodes start near the designated maze start region instead
  of silently respawning outside the perimeter or at unrelated locations

### 2. Optional: render egocentric RGB with anti-clipping protections

```bash
python3 scripts/2_visual_renderer.py --raw_dir jepa_raw_data --out_dir jepa_final_dataset
```

On detected ROCm hosts, the renderer caps render workers to 1 by default
because Genesis may still touch the HIP allocator even with `--sim_backend
vulkan`. Separately, any non-CPU worker-startup failure now triggers a serial
CPU retry for that chunk instead of aborting the entire render. Use
`--unsafe_backend_parallelism` only on machines where multi-worker Vulkan is
already known to be stable.

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
- base LeWM training now performs held-out scene evaluation by default, so the
  core world model is no longer judged only by in-training loss

Older mixed-state or averaged-command runs should be treated as baselines or
compatibility checkpoints, not the preferred starting point.

### 4. Train the planning heads

The current recommended planning stack for pure-perception exploration uses two
learned components on top of frozen LeWM latents:

- `LatentEnergyHead`: dense safety / mobility energy
- `ExplorationBonus`: novelty bonus
- `CoverageGainHead`: short sequence-level local frontier gain
- `EscapeFrontierHead`: longer-horizon "leave the basin and open frontier" value

The direct goal term at inference is still the rollout-space breadcrumb
similarity (`--goal_cost_mode terminal_cosine`). Goal and progress heads remain
available as optional experiments, but they are not required by the active
runtime path.

The planning heads must use the **same temporal abstraction and command
representation** as the world-model checkpoint they consume.

The head-training script now auto-detects encoder config from the LeWM
checkpoint and will fail fast on explicit mismatches. In particular:

- `latent_dim`, `image_size`, `patch_size`, `use_proprio`, `cmd_dim`,
  command representation, and command latency are resolved from the checkpoint
- scorer checkpoints now carry temporal / encoder metadata
- inference validates that scorer metadata matches the loaded world model
- held-out evaluation now defaults to `scene`-level splitting rather than
  per-env episode splitting so train/eval do not share the same chunk geometry

Important: the base world model does **not** use beacon labels, so it can still
be trained on `jepa_final_v3`. The planning heads should use the LOS-corrected
dataset (`jepa_final_v3_los`) so safety / visibility labels do not reward
through-wall shortcuts.

Recommended command:

```bash
python3 scripts/4_train_energy_head.py \
  --data_dir jepa_final_v3_los \
  --raw_data_dir jepa_raw_data \
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
  --coverage_gain_weight 0.5 \
  --escape_frontier_weight 1.0 \
  --eval_split_group scene \
  --exploration_weight 1.0
```

This writes a combined scorer checkpoint to:

- `energy_head_checkpoints_keyframe_exec_stride5_block15_vision/trajectory_scorer.pt`

### 5. Run world-model inference

Inference must use the same macro-timescale as training:

- planner horizon in latent steps
- `macro_action_repeat=5` so one planner step corresponds to one executed
  five-command block
- the planner now defaults to replanning every macro step (`mpc_execute=1`)
  because long open-loop execution was too brittle in collision-rich mazes
- the planner can condition on recent latent history, which better matches the
  training-time predictor context than single-frame planning
- the planner can search over a library of supported command-block primitives
  instead of unconstrained continuous 15D command blocks
- when a target beacon exists, the planner now includes a rollout-space goal
  term by default (`--goal_cost_mode terminal_cosine`) rather than treating
  target similarity as evaluation-only
- the planner can also use a perception-only keyframe graph when local progress
  stalls, which gives it longer-horizon memory without adding proprio or a
  privileged map

Recommended command:

```bash
python3 scripts/6_infer_pure_wm.py \
  --ppo_ckpt models/ppo/ckpt_20000.pt \
  --wm_ckpt lewm_checkpoints_keyframe_exec_stride5_block15_vision/epoch_30.pt \
  --scorer_ckpt energy_head_checkpoints_keyframe_exec_stride5_block15_vision/trajectory_scorer.pt \
  --seed 42 \
  --steps 200 \
  --macro_action_repeat 5 \
  --mpc_execute 1 \
  --cem_iters 10 \
  --planner_action_space primitives \
  --memory_router_mode keyframe \
  --route_progress_weight 7.0 \
  --rnd_online_lr 1e-3 \
  --out_dir inference_runs/perception_only_stride5_block15_vision_s42
```

Notes:

- Omit `--target_beacon` unless you know that beacon exists for the chosen
  random seed. If you pass an invalid beacon identity, the script will error.
- The active planner now uses goal attraction + learned safety + learned
  novelty + sequence-level frontier value + action penalty. If you want pure
  exploration, set
  `--goal_cost_mode off`.
- `--goal_cost_mode terminal_cosine` is the default because it compares rollout
  projected latents directly to the breadcrumb latent in the planner scoring
  space. `--goal_cost_mode head` is available only when the scorer checkpoint
  includes a trained goal head.
- `--memory_router_mode keyframe` is the recommended default for unseen mazes.
  It activates a persistent perception-only keyframe graph and temporarily
  routes toward remembered frontier waypoints when local planning plateaus.
- `--history_context_len` defaults to the world-model `max_seq_len`, so the
  planner rollout conditions on recent latent history unless you explicitly set
  it to `1`
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
    --mpc_execute 1 \
    --cem_iters 10 \
    --planner_action_space primitives \
    --rnd_online_lr 1e-3 \
    --out_dir "inference_runs/perception_only_stride5_block15_vision_s${seed}"
done

python3 scripts/7_aggregate_inference_runs.py \
  --glob "inference_runs/perception_only_stride5_block15_vision_s*" \
  --out inference_runs/perception_only_stride5_block15_vision_aggregate.json
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

The active inference path is now:

- learned safety cost
- learned novelty bonus
- direct rollout-space goal term
- learned sequence-level frontier / escape value
- perception-only keyframe routing when local progress stalls
- small action penalty

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

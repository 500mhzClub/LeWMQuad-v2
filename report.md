# Technical Report: Exploration / Planning Debugging on `LeWMQuad-v2`

Date: 2026-04-07

## Scope

This report summarizes the sequence of planner and scorer changes tried while debugging the persistent local-looping failure in `scripts/6_infer_pure_wm.py`.

The core symptom was stable across many variants:

- the robot remained trapped in a small quadrant of the maze
- it repeatedly executed slightly different local orbits
- coverage improved only marginally
- collision counts often remained high

The work fell into four phases:

1. Fix obvious latent-space mismatches in the exploration signal.
2. Replace RND-style novelty with visited-memory comparisons.
3. Improve the visited-memory comparison with snippet and place-aware embeddings.
4. Audit the planner to determine whether the remaining failure was due to bad heads, bad rollout prediction, or a flat objective.

The final conclusion is that the current scalar-head route is largely exhausted. The energy heads are accurate enough offline, but the planner still sees near-tied candidates inside a local basin and executes them into tiny wall-scraping motions. The remaining problem is not “novelty is broken” in the simple sense.

## Experimental Setup

Most runs shared this base setup:

- World model: `lewm_checkpoints_keyframe_exec_stride5_block15_vision/epoch_30.pt`
- Command representation: `active_block`, `cmd_dim=15`
- Temporal abstraction: `seq_len=4`, `stride=5`, `block=5`
- Seed: `42`
- `macro_action_repeat=5`
- `plan_horizon=5`
- `mpc_execute=5`
- `cem_iters=10`
- Safety mode: `consequence`

Unless noted otherwise, all online inference remained latent-only at runtime. Some later heads used privileged XY pose offline for supervision during training.

## Code Changes Implemented

The following code paths were extended during this investigation:

- `scripts/6_infer_pure_wm.py`
  - fixed exploration-space mismatch bugs
  - added `visited_nn` novelty
  - added margin, tail-window, revisit penalty, and safety gate variants
  - added snippet-level nearest-neighbor novelty
  - added place-head and displacement-head loading/inference
  - added planner auditing and predicted-vs-actual latent error logging
- `scripts/4_train_energy_head.py`
  - added rollout-space exploration training
  - added place-head training
  - fixed place-head triplet construction bug
  - added raw-pose joins for pose-supervised place training
  - added held-out evaluation for safety and place
  - added displacement-head training and held-out evaluation
- `lewm/models/energy_head.py`
  - added `PlaceSnippetHead`
  - added `DisplacementHead`

## Chronology of What Was Tried

### 1. Consequence safety + RND gain after latent-space alignment

#### Why

The first hypothesis was that novelty was failing because online RND updates were being applied in encoder observation space while planning and the retrained exploration head operated in rollout projected latent space.

#### Change

Online RND updates were changed to use the teacher-forced predicted latent for the executed command instead of the encoder-view observation latent.

#### Result

Run: `perception_only_stride5_block15_vision_s42_gain_rolloutv6_consequence_gainfix`

- Coverage: `1.513 m^2`
- Path length: `19.81 m`
- Console collision count: `201`
- Qualitative result: still trapped in the same quadrant and looping

Interpretation:

- safety improved relative to more naive variants
- the latent-space consistency bug was real
- but fixing it did not solve the local-looping failure

### 2. Full rollout-space RND update path

#### Why

After the first alignment fix, the remaining concern was that “memory” might still be tied to the wrong latent source or otherwise still be too weakly aligned with planning.

#### Change

The executed novelty update path was made consistently rollout-space / teacher-forced.

#### Result

Run: `perception_only_stride5_block15_vision_s42_gain_rolloutv6_consequence_gainfix_rndrollout`

- Coverage: `1.761 m^2`
- Path length: `15.10 m`
- Console collision count: `339`
- Qualitative result: still circling in the same basin

Interpretation:

- this confirmed that plain RND error was the wrong comparison
- it was still rewarding local latent/view variation instead of true escape

### 3. Replace RND gain with nearest-neighbor visited-bank novelty

#### Why

At this point the diagnosis was:

- RND gain was not asking “have I been here before?”
- it was only asking “does this endpoint still confuse the predictor?”

#### Change

Added `visited_nn`:

- maintain a bank of executed rollout latents
- score candidate terminal states by k-NN distance to that bank

#### Result

Run: `perception_only_stride5_block15_vision_s42_visitednn_k8`

- Coverage: `0.840 m^2`
- Path length: `6.57 m`
- Console collision count: `729`
- Qualitative result: no large retread loop, but severe corner-farming / getting stuck

Interpretation:

- the failure mode changed
- novelty stopped paying for large retracing loops
- but the planner still found local latent variation inside a corner and got trapped there

### 4. Margin threshold + tail-window persistence

#### Why

The `visited_nn` score was still rewarding small local differences. The next hypothesis was that repeated corner behavior sat below a stable novelty scale and could be zeroed out with a margin. Also, requiring persistence across multiple rollout tail states might suppress endpoint blips.

#### Change

Added:

- `visited_nn_margin`
- `visited_nn_tail_steps`

Novelty was computed over the rollout tail rather than just the final state.

#### Result

Run: `perception_only_stride5_block15_vision_s42_visitednn_k8_m015_t3`

- Coverage: `1.659 m^2`
- Path length: `14.01 m`
- Console collision count: `428`
- Qualitative result: novelty farming decreased, exploration improved, but looping persisted

Interpretation:

- this was a real improvement
- late repeated basin states often got `x = 0`
- however, once novelty shut off, the planner simply reverted to a local “safe” orbit

### 5. Safety gate on exploration bonus

#### Why

Some novelty bursts were clearly unsafe. The next idea was to suppress exploration where predicted safety was too poor.

#### Change

Added a sigmoid safety gate on the exploration bonus:

- threshold `0.18`
- sharpness `12.0`

#### Result

Run: `perception_only_stride5_block15_vision_s42_visitednn_k8_m015_t3_gate018_k12`

- Coverage: `1.294 m^2`
- Path length: `18.17 m`
- Console collision count: `275`
- Qualitative result: safer, but still looped locally

Interpretation:

- reduced collisions substantially
- did not create a real escape mechanism

### 6. Explicit revisit penalty

#### Why

If novelty had shut off, repeated local states were receiving no positive reward, but also no explicit “do not stay here” pressure beyond a nearly flat safety term.

#### Change

Added a signed revisit penalty below the visited margin.

#### Result

Run: `perception_only_stride5_block15_vision_s42_visitednn_k8_m015_t3_r8_gate018_k12`

- Coverage: `1.466 m^2`
- Path length: `15.64 m`
- Console collision count: `216`
- Qualitative result: best safety so far among the `visited_nn` family, but still trapped in the same broad basin

Interpretation:

- revisit penalties helped safety
- but once all candidates in the basin had similar revisit penalties, the term became nearly a constant offset and stopped affecting ranking

### 7. Snippet-level rather than state-level nearest-neighbor novelty

#### Why

Even the tail-window version was still comparing individual states unless the snippet representation itself was changed. The next hypothesis was that a multi-step snippet metric might better capture “same place / same basin” than state k-NN.

#### Change

When `visited_nn_tail_steps > 1`, the planner compared rollout snippets against executed snippets, not just isolated tail states.

#### Result

Run: `perception_only_stride5_block15_vision_s42_visitednn_snippet_k8_m015_t3_r8_gate018_k12`

- Coverage: `1.484 m^2`
- Path length: `18.37 m`
- Console collision count: `171`
- Qualitative result: modest improvement again, but still strong local looping

Interpretation:

- snippet comparison was directionally better
- but raw latent MSE in snippet space was still not sufficiently place-aware

### 8. First learned place-head attempt (temporal proxy)

#### Why

The raw latent/snippet geometry was clearly not enough. The next step was a learned place/snippet embedding head trained from rollout snippets.

#### Initial issue

The first implementation had a bug:

- `place_positive_radius` and `place_negative_gap` were compared against raw `obs_step`
- snippet starts were spaced by `temporal_stride=5`
- this produced zero valid positive triplets

Symptom:

- place loss stayed exactly `0.0`

#### Fix

The trainer was patched to interpret radii in snippet-start units and fail loudly if no valid triplets were built.

#### Result after fix

Temporal-proxy place-head training became valid, but online behavior still failed to escape.

Key runtime example:

Run: `perception_only_stride5_block15_vision_s42_visitednn_place_fix_bankfix`

- Coverage: `1.294 m^2`
- Path length: `15.47 m`
- Console collision count: `422`
- Qualitative result: revisit penalty stayed near constant and the system still looped

Interpretation:

- the temporal-proxy place head was not enough
- stronger supervision was needed

### 9. Pose-supervised place head

#### Why

The next move was to stop relying on temporal overlap as a proxy and supervise the place head with true same-episode XY proximity, while still keeping latent-only inference.

#### Change

Joined rendered HDF5 chunks with raw rollout chunks so pose labels from `jepa_raw_data_v3` could supervise the place head.

#### Held-out metrics

From `rolloutv8_place_pose` training:

- Held-out safety:
  - `rmse=0.1850`
  - `mae=0.0986`
  - `pearson=0.808`
  - `collision_auc=0.908`
  - `pred(coll)=0.569`
  - `pred(no-coll)=0.076`
- Held-out place:
  - `queries=4096`
  - `R@1=0.974`
  - `R@5=0.999`
  - `d_pos=0.053`
  - `d_neg=0.516`

These are strong offline metrics.

#### Runtime result

Run: `perception_only_stride5_block15_vision_s42_visitednn_place_pose`

- Coverage: `1.39 m^2`
- Path length from summary not captured separately in this report
- Console collision count: `333`
- Qualitative result: still looping

Interpretation:

- the place head was not the main bottleneck anymore
- despite strong held-out retrieval, the planner still failed online

### 10. Myopia / longer-horizon test

#### Why

One possibility was that the planner simply did not see far enough ahead to commit to temporarily worse sequences that would eventually escape the basin.

#### Change

A much less myopic configuration was tried:

- larger horizon
- smaller `mpc_execute`
- more CEM iterations

#### Result

This was much slower and appeared further outside the world model’s training regime. Early metrics were poor, so this line was not pursued further.

Interpretation:

- even if myopia contributes, the chosen ablation was too expensive and likely too OOD to be the main path forward

### 11. Planner audit without prediction-error logging

#### Why

At this point the question was no longer “which novelty head is best?” It was:

- are the top-ranked candidates meaningfully different?
- is the planner picking a clearly better option that reality then invalidates?
- or are all candidates nearly tied inside the basin?

#### Change

Added `--audit_plan` to log:

- state before replan
- selected plan
- final-iteration top-K candidates
- actual post-execution outcome after the first command

#### Result

Run: `perception_only_stride5_block15_vision_s42_visitednn_place_pose_audit`

Summary derived from `plan_audit.jsonl`:

- mean top-5 cost spread: `0.010997`
- mean top-5 safety spread: `0.011369`
- mean top-5 revisit spread: `0.000106`
- mean actual first-step displacement: `0.01094 m`
- mean coverage gain per replan: `0.000562 m^2`
- collision rate: `0.80`

Interpretation:

- inside the basin, the planner sees near-tied candidates
- revisit penalty is effectively constant across top-K
- actual executed motion is tiny
- the planner is not meaningfully separating “good escape” from “local scrape” candidates

### 12. Predicted-vs-actual latent error audit

#### Why

The next question was whether rollout prediction itself collapsed near collisions. If collision steps were much harder to predict, the world model could still be the primary bottleneck.

#### Change

Added predicted-vs-actual latent error logging:

- raw MSE / L2 / cosine
- projected MSE / L2 / cosine
- reported separately for collision and non-collision audited steps

#### Result

Run: `perception_only_stride5_block15_vision_s42_visitednn_place_pose_audit_prederr`

Prediction error summary:

- Mean projected prediction error:
  - `proj_mse=0.1700`
  - `proj_cosine=0.898`
- Collision steps:
  - `count=32`
  - `proj_mse=0.1699`
  - `proj_cosine=0.894`
- Non-collision steps:
  - `count=8`
  - `proj_mse=0.1706`
  - `proj_cosine=0.914`

Interpretation:

- collision prediction error was not meaningfully worse than non-collision prediction error
- this argues against a simple “rollout model catastrophically fails only at contact” explanation

### 13. Pose-supervised displacement head

#### Why

Since novelty saturated and revisit became almost constant inside a basin, the next idea was to add a non-heuristic positive motion term: predict actual XY displacement over a short future horizon from latents, then reward plans that are predicted to move.

#### Change

Added `DisplacementHead`:

- trained from true pose-supervised XY deltas
- latent-only at inference

#### Held-out metrics

From `rolloutv9_disp` training:

- Held-out safety:
  - `rmse=0.1850`
  - `mae=0.0993`
  - `pearson=0.808`
  - `collision_auc=0.909`
- Held-out place:
  - `R@1=0.973`
  - `R@5=0.998`
  - `d_pos=0.053`
  - `d_neg=0.518`
- Held-out displacement:
  - `n=18738`
  - `rmse=0.0451 m`
  - `mae=0.0381 m`
  - `pearson=0.401`
  - `pred mean=0.066 m`
  - `target mean=0.068 m`

Interpretation:

- the displacement head learned the scale reasonably well
- but its ranking power was only moderate

### 14. Displacement-head online audit

#### Why

Before integrating displacement strongly into planning, it was necessary to check whether it actually separated top-K candidates inside the local basin.

#### Result

Run: `perception_only_stride5_block15_vision_s42_disp_audit`

Audit-derived statistics:

- mean selected predicted displacement: `0.05525 m`
- mean actual first-step displacement: `0.01565 m`
- correlation predicted vs actual displacement: `0.3746`
- mean top-5 predicted-displacement spread: `0.000460 m`
- mean top-5 cost spread: `0.01576`
- mean top-5 safety spread: `0.01630`
- mean top-5 revisit spread: `0.000140`
- mean coverage gain per replan: `0.00194 m^2`
- collision rate: `0.425`

Interpretation:

- the displacement head is somewhat informative offline
- but inside the top-5 candidate set it is almost constant
- a reasonable planner weight will not materially change ranking
- using a very large weight to force an effect would likely be unstable and not principled

## What We Learned

### 1. The energy heads are not the main bottleneck anymore

The held-out metrics are strong enough that “bad heads” is no longer the primary explanation.

The strongest evidence:

- safety `collision_auc ≈ 0.91`
- place `R@1 ≈ 0.97`, `R@5 ≈ 1.0`

This is much stronger than the online behavior would suggest.

### 2. The problem is now planner ranking inside a local basin

Across the audits:

- top-K cost spread is tiny
- top-K safety spread is tiny
- top-K revisit spread is effectively zero
- top-K predicted-displacement spread is also effectively zero

Inside the local loop, all candidates are nearly tied.

### 3. Revisit penalties became constant offsets

The revisit mechanism was useful for suppressing large retreading and some novelty farming, but once the robot was inside a basin:

- `x` went to zero
- `r` stayed nearly constant across candidates

At that point the revisit term no longer changes the argmin and does not help escape.

### 4. Runtime behavior is low-motion even when commands differ

The planner produces different command sequences, but actual executed first-step motion remains small:

- around `1.1 cm` in the place-audit run
- around `1.6 cm` in the displacement-audit run

This means the planner-world-model-policy stack near contacts is effectively low-motion and locally self-consistent enough to keep orbiting.

### 5. Prediction error is not obviously worse on collisions

The collision vs non-collision prediction-error audit does not show a sharp collapse near contact.

That does not prove the rollout model is “good enough” in all relevant senses, but it does mean the failure is not simply explained by a special collision-prediction breakdown.

## Pure-Perception Status

The project crossed two different purity boundaries:

### Runtime inference purity

`rolloutv8_place_pose` and `rolloutv9_disp` still preserve pure-perception inference in the narrow runtime sense:

- at inference time the planner uses only image-derived latents and learned heads
- it does not use live XY pose

### Training-time purity

These same runs are no longer pure-perception in the stricter method-level sense because offline supervision used privileged pose:

- place head: `pose_xy_same_episode`
- displacement head: `pose_xy_delta_same_episode`

An explicit frontier-map planner would also break pure-perception at inference because it would require online XY state.

## Why Frontier Routing Was Not Pursued

An explicit frontier-map route was considered because it is likely to work better as an exploration objective than scalar novelty. However, it was not the right next step here because:

- it would require online XY pose
- that breaks pure-perception inference
- it moves the system into an explicit navigation heuristic/controller regime

Given the user’s constraint and the desire to avoid “heuristic hell,” this path was not advanced as the main recommendation.

## Recommended Next Steps

### Primary recommendation: train a sequence-level escape / coverage-gain head

The current heads are all local scalar terms:

- per-state safety
- per-state or per-snippet novelty
- per-plan endpoint displacement

The observed failure is sequence-level:

- the planner needs a signal for “this whole candidate rollout will actually get me out of the basin”
- not “this endpoint is a bit novel” or “this endpoint might move slightly”

The next clean experiment is therefore:

1. Train a `CoverageGainHead` or `EscapeHead`.
   - Inputs:
     - current latent
     - candidate rollout snippet or rollout tail
   - Targets:
     - realized future coverage gain over a longer horizon, or
     - binary escape label such as “did this sequence leave the current local basin?”
   - Use pose only offline to build the labels.

2. Evaluate it on held-out data before using it online.
   - For coverage gain:
     - correlation
     - RMSE
     - ranking agreement across candidate snippets
   - For escape:
     - AUC
     - precision/recall
     - ranking quality among candidates from the same local state

3. Add it to `6_infer_pure_wm.py` in audit-only mode first.
   - score the top-K planner candidates
   - check whether it meaningfully spreads candidates inside the currently observed basin

4. Only integrate it into the planner if it clearly separates top-K candidates.
   - start with a small weight
   - keep the strong safety head in the loop

### Secondary recommendation: stop investing in more RND / visited-NN / displacement tuning

These branches have been tested thoroughly enough to support a stop decision:

- plain RND gain
- rollout-space RND updates
- visited-bank nearest neighbors
- margin thresholds
- tail-step novelty
- safety gates
- revisit penalties
- snippet k-NN
- place-aware snippet k-NN
- displacement-head reward

All of them changed behavior, but none solved the core escape failure.

### What not to do next

The following are not recommended as the next main line of work:

- more RND variants
- more visited-margin or kNN hyperparameter sweeps
- stronger displacement weights without new evidence
- longer-horizon OOD planning as the primary path
- porting the heuristic stuck-recovery controller from `5_infer_new_maze.py`
- explicit frontier routing if pure-perception inference is meant to be preserved

## Final Conclusion

The project is no longer blocked by obviously broken exploration heads.

The strongest current evidence is:

- the safety head is accurate enough offline
- the place head is accurate enough offline
- collision prediction error is not dramatically worse than non-collision prediction error
- but online, top-K candidates inside a local basin are almost indistinguishable under the current scalar objectives

Therefore the next step should be a sequence-level value target that predicts realized escape or coverage gain, not another local novelty or displacement scalar.

## Appendix: Representative Online Results

The table below summarizes the main run family. Coverage and path length come from `summary.json`. For earlier runs where `summary.json` did not retain collision count, the console value is used.

| Run | Main idea | Coverage `m^2` | Path `m` | Collisions | Outcome |
| --- | --- | ---: | ---: | ---: | --- |
| `gain_rolloutv6_consequence_gainfix` | align online RND update to rollout-space teacher-forced latent | 1.513 | 19.81 | 201 | loops in same quadrant |
| `gain_rolloutv6_consequence_gainfix_rndrollout` | full rollout-space RND update path | 1.761 | 15.10 | 339 | still loops |
| `visitednn_k8` | banked k-NN novelty | 0.840 | 6.57 | 729 | corner farming / trapped |
| `visitednn_k8_m015_t3` | margin + tail persistence | 1.659 | 14.01 | 428 | better, still loops |
| `visitednn_k8_m015_t3_gate018_k12` | novelty safety gate | 1.294 | 18.17 | 275 | safer, still loops |
| `visitednn_k8_m015_t3_r8_gate018_k12` | revisit penalty | 1.466 | 15.64 | 216 | best safety in family, still loops |
| `visitednn_snippet_k8_m015_t3_r8_gate018_k12` | snippet-level visited k-NN | 1.484 | 18.37 | 171 | still loops |
| `visitednn_place_fix_bankfix` | temporal-proxy place head + bank fix | 1.294 | 15.47 | 422 | revisit nearly constant, still loops |
| `visitednn_place_pose` | pose-supervised place head | 1.39 | not emphasized | 333 | still loops |
| `visitednn_place_pose_audit` | audit-only short run | 0.388 | 2.30 | 157 over 200 steps | clear low-motion basin trap |
| `disp_audit` | displacement-head audit | 0.724 | 3.77 | 84 over 200 steps | displacement not discriminative enough |

## Appendix: Key Audit Numbers

### Place-audit run

- mean top-5 cost spread: `0.010997`
- mean top-5 safety spread: `0.011369`
- mean top-5 revisit spread: `0.000106`
- mean actual first-step displacement: `0.01094 m`
- mean coverage gain: `0.000562 m^2`
- collision rate: `0.80`

### Prediction-error audit

- collision `proj_mse`: `0.1699`
- non-collision `proj_mse`: `0.1706`
- collision `proj_cosine`: `0.894`
- non-collision `proj_cosine`: `0.914`

### Displacement-audit run

- mean selected predicted displacement: `0.05525 m`
- mean actual first-step displacement: `0.01565 m`
- predicted-vs-actual displacement correlation: `0.3746`
- mean top-5 predicted-displacement spread: `0.000460 m`
- mean top-5 cost spread: `0.01576`
- mean top-5 safety spread: `0.01630`
- mean top-5 revisit spread: `0.000140`

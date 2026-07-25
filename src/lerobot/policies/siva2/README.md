# SIVA2: Steerable, memory-aware SIVA

SIVA2 is a separate experimental policy. It does not replace or modify SIVA;
use `--policy.type=siva` for the original architecture and
`--policy.type=siva2` for this one.

## Architecture

SIVA2 keeps Florence-2 and the efficient structured residual-flow motor core,
then incorporates the most actionable ideas from π*0.6/RECAP and π0.7:

```text
current cameras -> scene slots ----------------------+
camera history -> temporal attention -> memory slots |
task/subtask tokens -> goal slots                    |
future/subgoal images -> subgoal slots               +-> typed fusion
quality/speed/mistake/advantage/control/source ------+       |
                                                             +-> value/progress head
                                                             +-> primitive router
                                                             +-> strategy router
                                                                  |
                                             factored prior + residual flow -> actions
```

The high-level semantic policy and subgoal world model are intentional external
interfaces: SIVA2 consumes their subtask tokens and images but keeps them off the
fast control path. This allows those slower components to evolve independently.

## Data contract

All extra fields are optional, so demonstration-only SIVA/XVLA datasets still
run. Autonomous and intervention data should supply explicit labels:

| Batch key | Meaning |
| --- | --- |
| `observation.subtask` | Current semantic subtask; the shared tokenizer creates token fields |
| `observation.subgoal.*` | Optional visual feature(s) showing a desired near-future state |
| `siva2.quality` | Execution quality from 1 to 5 |
| `siva2.speed` | Episode length or another consistent duration measure |
| `siva2.mistake` | Whether the current segment contains a mistake |
| `siva2.advantage` | RECAP-style improvement indicator, normally -1/0/1 |
| `siva2.control_mode` | 0 unknown, 1 joint, 2 end-effector |
| `siva2.data_source` | Project-defined source ID, such as demo/autonomous/intervention |
| `siva2.success` | Binary value target |
| `siva2.progress` | Progress target in [0, 1] |
| `siva2.time_to_completion` | Remaining steps/time value target |
| `siva2.intervention` | Binary correction/intervention target |

Missing quality, advantage, and mistake labels are treated as high-quality,
positive-advantage, mistake-free demonstrations by default. Disable
`assume_demonstration_if_unlabeled` before mixing unlabeled autonomous rollouts.

SIVA2's value head trains from supplied outcome labels. Producing advantage
labels from a reference policy/value checkpoint is a separate offline data step;
the policy does not pretend to infer valid counterfactual advantages from one
behavior-cloning mini-batch.

## Initialize from XVLA

```bash
uv run lerobot-convert-xvla-to-siva2 \
  --source="outputs/train/my-xvla/checkpoints/last/pretrained_model" \
  --output-dir="outputs/converted/my-siva2-init"
```

Only `model.vlm.*` tensors transfer. Typed slots, temporal memory, factored
priors, residual flow, and the value model begin newly initialized. Processor
JSON files and their referenced normalization state files are copied together.

## First controlled comparison

Train XVLA, SIVA, and SIVA2 on the same split and optimizer budget. Initially
leave optional subgoals and RL labels absent so the test isolates typed slots and
memory. Then add, one at a time: episode metadata, value targets, advantage
conditioning, real future-frame subgoals, and autonomous/intervention data.

Measure physical success versus demonstrations, intervention rate, completion
time, action jerk, rollout latency, memory ablations, and route utilization.
Architecture alone is not evidence of improvement; these ablations are the
tests that can validate or reject each SIVA2 hypothesis.

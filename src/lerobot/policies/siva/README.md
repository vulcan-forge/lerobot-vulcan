# SIVA: Slot-Intent Vision-Language-Action

SIVA is an experimental successor path for the repository's working XVLA policy.
It keeps XVLA's Florence-2 representation and preprocessing boundary so the new
action architecture can be evaluated without changing the dataset underneath it.
It is a research hypothesis, not yet a claim that it outperforms XVLA or π0.5.

## Why this direction

XVLA has two excellent properties worth preserving: a pretrained vision-language
representation and embodiment conditioning that makes heterogeneous robot data
trainable in one model. Its action transformer, however, repeatedly processes the
full multimodal token sequence, starts each chunk from generic Gaussian noise, and
normally uses ten refinement evaluations.

π0.5 adds a valuable hierarchy: semantic subtask prediction supplies long-horizon
intent before the low-level action expert runs. Its strongest generalization result
also depends on heterogeneous co-training (robot trajectories, web knowledge,
semantic labels, detections, and verbal supervision), not only on a different
low-level decoder. SIVA therefore separates two problems:

1. **Represent the scene once.** Learned cross-attention queries compress all
   Florence image/language tokens into a fixed set of task-conditioned slots.
2. **Choose a motion intent.** Each action chunk is assigned to a learned coarse
   trajectory mode. An observation router learns the assignment as a free
   self-supervised label derived from actions.
3. **Refine locally.** A compact action transformer cross-attends to the frozen
   slots and learns a residual flow from the selected motion prior to the target.
   The initial transport path should be shorter and less tangled than isotropic
   noise-to-action flow matching.
4. **Share without erasing embodiment.** Motion modes are shared across robots;
   low-rank domain offsets absorb kinematic conventions. This is more parameter
   efficient than an independent action model and more structured than asking
   soft prompts to absorb every difference.

The motion library uses linearly interpolated control points in this first version.
This gives smooth, inspectable priors and a useful fallback while the zero-initialized
residual flow learns. It can later be replaced by splines, latent video actions, or
another structured prior without changing the context encoder.

## Expected advantages and failure modes

| Hypothesis | Measurement | Main risk |
| --- | --- | --- |
| Fixed context slots reduce compute | train step time, peak VRAM, rollout latency | too few slots discard contact details |
| Motion priors improve sample efficiency | success vs. number of demonstrations/steps | dead or collapsed modes |
| Three-step residual flow matches ten-step XVLA | success and action jerk at equal latency | coarse priors miss rare motions |
| Low-rank domain offsets transfer between robots | leave-one-embodiment-out fine-tuning | offsets learn dataset IDs rather than kinematics |
| Native padding masks improve short-episode training | boundary loss and NaN rate | none expected; still verify preprocessing |

The router balance loss addresses dead modes, token dropout tests multi-view
robustness, and every auxiliary loss has a configuration weight for ablation.

## First experiment

Use the same dataset split, Florence weights, image preprocessing, optimizer budget,
and action chunk as the strongest XVLA run. Compare:

- XVLA at its current ten denoising steps.
- SIVA with 1, 2, and 3 flow steps.
- SIVA without structured priors (`prior_loss_weight=0`, source replaced by noise;
  a small code ablation is still needed).
- SIVA with 4, 8, and 16 motion modes and 8, 16, and 32 context slots.

Track validation action error, physical task success, endpoint error, temporal jerk,
router utilization, step time, and peak VRAM. The decisive plot is physical success
against the number of training demonstrations—not final loss alone.

## What is implemented now

- A registered `siva` LeRobot policy/config.
- XVLA-compatible tokenization, image normalization, domain IDs, and action spaces.
- Task-conditioned slot compression.
- Shared control-point motion priors with low-rank embodiment adapters.
- Temporally correlated structured source noise.
- Residual flow action decoding and deterministic three-step inference.
- Flow, endpoint, prior reconstruction, router, balance, smoothness, and padded-action losses.

Initialize from a working local XVLA checkpoint with:

```bash
uv run lerobot-convert-xvla-to-siva \
  --source="outputs/train/my-xvla/checkpoints/last/pretrained_model" \
  --output-dir="outputs/converted/my-siva-init"
```

The converter copies `model.vlm.*`, preserves the processor files, and writes
`siva_initialization.json`. The new context/action modules are absent by design and
are initialized when the converted checkpoint is first loaded. Florence is frozen
by default; pass `--train-vlm` only when the data and compute budget justify it.

## Research lineage

- [X-VLA](https://openreview.net/forum?id=kt51kZH4aG): soft-prompted cross-embodiment
  training and the Florence representation boundary used here.
- [π0.5](https://arxiv.org/abs/2504.16054): semantic/low-level hierarchy and
  heterogeneous co-training.
- [Latent Action Guided Flow Matching](https://arxiv.org/abs/2606.23420): evidence
  that observation-conditioned structured source distributions can shorten the
  action flow path. SIVA's control-point library, slot bottleneck, and low-rank
  cross-embodiment adapters are separate experimental choices.

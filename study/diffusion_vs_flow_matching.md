# Diffusion and flow matching for action generation

This note explains two ways of generating a continuous robot action chunk from a
current observation and instruction. Let the condition be
\(c=(o,\ell)\), with visual/robot observation \(o\) and language instruction
\(\ell\), and let a demonstrated action chunk be
\(A\in\mathbb R^{H\times d}\). Both methods learn a conditional distribution
over action chunks, rather than mapping every condition to one averaged action.
The model architecture and the generative objective are related but distinct:
the same transformer can be trained to predict noise or a velocity field.

## Conditional action diffusion

During training, a noise level \(k\) is sampled and Gaussian noise is added to
the demonstrated chunk:

\[
A_k=\sqrt{\bar\alpha_k}A_0+\sqrt{1-\bar\alpha_k}\,\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
\]

A denoiser receives the noisy action, its noise level and the observation
condition. In the common noise-prediction parameterization it is trained with
\[
\mathcal L_{\mathrm{diff}}=
\mathbb E_{A_0,k,\epsilon,c}
\left[\|\epsilon-\epsilon_\theta(A_k,k,c)\|^2\right].
\]
The predicted noise can be converted to an estimate of the clean chunk or to a
reverse transition. At inference, sampling starts with a Gaussian action chunk
and iteratively applies reverse denoising updates. The original DDPM reverse
chain injects noise at intermediate steps. DDIM changes the sampling path while
using the same training objective; its \(\eta=0\) setting is deterministic for
a fixed initial noise and condition. Thus, “diffusion” does not imply that every
sampler must add fresh noise at every step.
[DDPM](https://arxiv.org/abs/2006.11239),
[DDIM](https://arxiv.org/abs/2010.02502).

For robot control, [Diffusion Policy](https://arxiv.org/abs/2303.04137) applies
conditional denoising to sequences of actions, with visual conditioning and a
receding-horizon controller. Its denoiser can represent multiple plausible
action modes and generates a chunk by repeated denoising. It is an important
visuomotor policy example, but it is not itself a large language-conditioned
VLA. [Octo](https://arxiv.org/abs/2405.12213) combines a transformer backbone
and readout tokens with a lightweight DDPM action head: the backbone embeds
observations, task text or goal image, and observation history; the head then
denoises continuous action chunks. [CogACT](https://arxiv.org/abs/2411.19650)
instead describes a VLM-conditioned, specialized diffusion action module. These
examples differ in inputs, training data, model size and action parameterization;
the shared term “diffusion” alone does not make their architectures
interchangeable.

## Flow matching

Flow matching trains a time-conditioned vector field that transports a simple
noise distribution to the data distribution along a chosen probability path.
For a paired noise sample \(\epsilon\) and action chunk \(A\), one simple
linear path is

\[
A^\tau=\tau\epsilon+(1-\tau)A,\qquad
u^\tau=\frac{d A^\tau}{d\tau}=\epsilon-A,
\quad \tau\in[0,1].
\]

The network is conditioned on \((A^\tau,\tau,c)\) and trained to regress the
path's target velocity:

\[
\mathcal L_{\mathrm{FM}}=
\mathbb E_{A,\epsilon,\tau,c}
\left[\|v_\theta(A^\tau,\tau,c)-u^\tau\|^2\right].
\]

In this convention, \(\tau=1\) is noise and \(\tau=0\) is data. Inference
starts from noise and numerically integrates the learned vector field toward
data. An ordinary differential equation solver follows a fixed path once the
condition and initial noise are fixed. Other probability paths and stochastic
flow formulations are possible; the integration rule and step count are
implementation choices, not properties fixed by the words “flow matching.”
The general flow-matching objective regresses a vector field along a selected
path, and those paths can include diffusion paths as well as alternatives such
as straight optimal-transport paths.
[Flow Matching](https://arxiv.org/abs/2210.02747).

The selected checkpoint, \(\pi_0\), is a VLA that uses a pretrained vision-
language model and a transformer action expert to predict a flow velocity.
In the pinned upstream `pi0_libero` implementation examined for this
reproduction, training-time action interpolation follows the equation above,
with \(\tau\sim\mathrm{Beta}(1.5,1)\) scaled to \([0.001,0.999]\). Sampling
initializes a noisy action chunk and uses ten explicit Euler steps from
\(\tau=1\) to \(0\). For LIBERO, the returned chunk has shape \(50\times7\);
the controller executes five actions and then observes again. These are
implementation facts for this pinned checkpoint/configuration, not universal
settings for all \(\pi_0\) variants or all flow-matching VLAs. The source
walkthrough and exact pin are recorded in
[`repro/PIPELINE.md`](../repro/PIPELINE.md).
[\(\pi_0\) paper](https://arxiv.org/abs/2410.24164).

## Comparison

| Aspect | Action diffusion | Flow matching |
|---|---|---|
| Learned target | Often noise/score at a selected corruption level; other parameterizations are possible. | A time-dependent vector field for a chosen probability path. |
| Training example | Corrupt a demonstrated action chunk using a noise schedule and predict the injected noise. | Sample a point on a noise-to-action path and regress its conditional target velocity. |
| Inference | Repeated reverse denoising transitions from noise; DDPM can add reverse-step noise, and DDIM can use a deterministic path. | Integrate a learned vector field from a noise sample toward an action chunk; the solver can be deterministic or stochastic depending on formulation. |
| Role of path/schedule | The forward corruption schedule determines the noisy action distribution at each level and the matching reverse process. | The designer selects the probability path and target field; a path may itself be diffusion-like. |
| Typical action architecture | Observation encoder plus a conditional denoiser, such as a temporal U-Net or transformer. | Observation/language encoder plus a time-conditioned velocity network; in \(\pi_0\), the action expert is transformer-based. |
| Sampling cost | Usually multiple denoiser evaluations per chunk. | Usually multiple vector-field evaluations per chunk. Fewer steps are not guaranteed without a suitable path, solver and model. |
| Multimodality | Can model multiple action modes through its conditional generative distribution. | Can also model multiple modes through the initial-noise-to-output mapping. A deterministic solver at fixed initial noise does not make the learned conditional distribution single-mode. |

The practical distinction is the prediction target and the sampling dynamics,
not “random actions” versus “deterministic actions.” Both methods learn a
conditional generative model and can express multiple plausible behaviors.
Diffusion commonly learns a denoising/score objective and samples through
reverse transitions; flow matching learns a velocity field and samples by
integrating it. But flow matching can use a diffusion probability path, and a
diffusion model can have a deterministic DDIM sampler. Fair speed comparisons
must report the network, solver, number of function evaluations, action chunk
size, hardware and control/replanning rate; step counts alone are not a
hardware-independent speed result.

## What this means for the LIBERO reproduction

The reproduced \(\pi_0\) uses flow matching for continuous action chunks. This
does not give it a scene graph or persistent object memory. In the examined
LIBERO interface, the policy receives two current RGB images, robot state and
the prompt; simulator poses are evaluation records, not policy inputs. It
replans after executing a short action prefix. Therefore, a graph proposal
could condition the action expert with explicit object identities/relations,
or support a high-level target-selection step before action generation. The
graph would need to be grounded in observations available to the policy and
updated when objects move. Feeding simulator-only poses would create a
privileged-information comparison, not evidence that a deployable perception
module improved the VLA.

The current rollout study does not compare diffusion against flow matching,
change the sampler, or retrain the policy. The discussion here explains the
selected model and places it in context; it makes no new empirical claim about
which objective is more accurate or faster for LIBERO.

## Contrast with autoregressive action tokens

Diffusion and flow matching generate continuous action chunks through repeated
updates in a continuous action space. An autoregressive VLA instead predicts a
sequence of discrete action tokens conditioned on previous tokens. This
requires an action tokenizer: simple per-dimension bins can create long,
highly correlated token sequences, while [FAST](https://arxiv.org/abs/2501.09747)
uses a discrete cosine transform and compression to represent action chunks
with fewer tokens. FAST is a tokenization method used by an autoregressive
policy; it is neither a diffusion sampler nor a flow-matching objective. The
paper reports a Pi0-FAST variant and compares its training/performance tradeoffs
with diffusion VLAs, but the present LIBERO rerun uses the separate continuous
flow-matching `pi0_libero` checkpoint.

## Observed policy-call timing in the fresh Spatial run

The 2026-10-30 run recorded 2,746 client-side elapsed times around `client.infer`.
These include the client/server call path and policy inference; they are not isolated
GPU kernel timings and do not compare diffusion and flow-matching models. The run
used the pinned `pi0_libero` flow-matching policy on one RTX PRO 6000 Blackwell
runtime. Values pool calls within each condition; p95 uses nearest-rank.

| Condition | Episodes | Calls | Mean (ms) | Median (ms) | p95 (ms) |
|---|---:|---:|---:|---:|---:|
| E0 | 50 | 1,115 | 92.67 | 92.90 | 95.98 |
| S1 | 27 | 730 | 92.98 | 93.11 | 95.91 |
| T1 | 8 | 352 | 93.04 | 93.11 | 95.62 |
| T1-ctrl | 8 | 134 | 93.27 | 93.21 | 96.26 |
| S2 | 6 | 264 | 93.37 | 93.27 | 95.95 |
| S2-ctrl | 6 | 151 | 92.98 | 93.10 | 95.65 |

These timings describe this inference client, network path, hardware and run. They do
not show that flow matching is faster than diffusion. The machine-readable aggregation
is [`latency_summary.json`](../../artifacts/part2/2026-10-30-rerun/latency_summary.json);
raw per-call data is in each archived episode `metrics.json` and `trajectory.npz`.

## Primary references

- Ho, Jain and Abbeel. [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239), 2020.
- Song, Meng and Ermon. [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502), 2020.
- Lipman et al. [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747), 2023.
- Chi et al. [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137), RSS 2023 / extended version 2024.
- Octo Model Team et al. [Octo: An Open-Source Generalist Robot Policy](https://arxiv.org/abs/2405.12213), 2024.
- Li et al. [CogACT: A Foundational Vision-Language-Action Model for Synergizing Cognition and Action in Robotic Manipulation](https://arxiv.org/abs/2411.19650), 2024.
- Black et al. [\(\pi_0\): A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164), RSS 2025.
- Pertsch et al. [FAST: Efficient Action Tokenization for Vision-Language-Action Models](https://arxiv.org/abs/2501.09747), 2025.

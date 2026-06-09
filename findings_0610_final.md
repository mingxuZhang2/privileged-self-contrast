# J2 Final Report: Privileged Self-Contrast — Negative Result

## June 10, 2026

---

## Summary

Three rounds of experiments on ProcessBench (3,400 expert-annotated math reasoning chains) show that **reference-conditioned distributional self-contrast does not outperform entropy baseline** for step-level error localization with a frozen 3B model. The direction is stopped.

---

## Timeline

| Round | What we tested | Result | Status |
|---|---|---|---|
| **Round 1** (leaky) | z = steps[:label] + error marker | Step AUROC 0.794, F1 42.3 | **INVALID** — label leakage, marker-driven |
| **Round 2** (non-leaky v1) | z = final answer (mixed GSM8K+MATH, MATH lacked refs) | Step AUROC 0.560 | Diluted by no-ref samples |
| **Round 3** (clean) | z = gold solution, all refs matched, 6 controls | Step AUROC 0.601 | **Does not beat entropy (0.617)** |
| **Round 3b** (rescue) | Residual-TI + entropy combination | GSM8K +0.4pp, MATH -0.7pp | **No improvement** |

---

## Clean Experiment Results (Round 3)

### Step-Level Error Localization (core metric)

| Method | GSM8K | MATH | ALL |
|---|---|---|---|
| **Entropy baseline (no z, no reference)** | **0.651** | **0.605** | **0.617** |
| Gold solution z (correct ref) | 0.635 | 0.589 | 0.601 |
| Textual alignment (string similarity) | 0.615 | 0.580 | 0.589 |
| Other example's solution z (wrong ref) | 0.590 | 0.543 | 0.555 |
| Shuffled z (irrelevant) | 0.571 | 0.566 | 0.567 |
| Final answer only z | 0.533 | 0.502 | 0.510 |
| Wrong answer z | 0.514 | 0.485 | 0.492 |

### Random Boundary Marker Control (Round 2)

Confirmed the leaky results were marker-driven:
- TI argmax follows RANDOM marker: **37.6%**
- TI argmax follows REAL error: **9.6%**
- TI argmax follows NEITHER: 52.7%

### Residual-TI + Entropy Combination (Round 3b)

Tuned alpha on GSM8K, tested on MATH:
- Best GSM8K: entropy + 0.3*residual = 0.655 (+0.4pp over entropy alone)
- MATH: same combo = 0.598 (**-0.7pp**, worse than entropy)
- Conclusion: residual signal does not generalize

---

## Why It Failed

1. **3B model cannot align reference solution to candidate steps.** Gold solution is placed in the prompt, but the model doesn't reliably map reference step k to candidate step k. The distributional shift is diffuse, not localized at the error.

2. **Gold reference causes non-error distributional perturbation.** Even on correct steps, wording/notation/derivation path differences between reference and candidate cause TI to spike. This noise drowns the error signal.

3. **Error steps already have high entropy.** Mathematical errors tend to occur at steps where the model is uncertain — long algebraic manipulations, conceptual jumps, numeric calculations. Student entropy captures this directly without any reference.

4. **Final answer alone is too weak.** A 3B model cannot reverse-engineer which step is wrong from just knowing the final answer should be different. Step AUROC 0.510 ≈ random.

---

## What Survived

1. **TokenIncompatibility >> InfoGain/KL** as an error detection signal (confirmed even under leakage). KL measures "z changed the distribution" (information gain); TI measures "z disagrees with the actual token produced" (error signal). This distinction is valid regardless of z quality.

2. **The leakage detection methodology is solid.** Random-boundary-marker control + shuffled-z control + other-example controls form a clean ablation framework for detecting invalid privileged information.

3. **Entropy is a strong and underrated error localizer.** Step-level entropy with a frozen 3B model gives 0.617 step AUROC on ProcessBench — no training, no reference, 1 forward pass. This itself could be a useful baseline for future PRM research.

---

## Questions for Discussion

1. **Is this a 3B model limitation or a fundamental method limitation?** Would a 72B model produce enough distributional shift from gold references to beat entropy? The hypothesis: larger models better "understand" the reference and produce more targeted disagreement at error steps. But we have no evidence yet.

2. **Is ProcessBench the right benchmark for this method?** ProcessBench provides pre-segmented steps from model-generated solutions. The errors are diverse (calculation, logic, concept, completeness). Maybe PSC works better on a narrower error type (e.g., pure arithmetic errors where the reference directly contradicts)?

3. **Should we pivot to using OPSD signals for training (original OPD/OPSD use case) rather than evaluation?** The original OPSD papers use distributional contrast as a training signal, not an evaluation signal. Our attempt to repurpose it for evaluation didn't work at 3B scale. Maybe the right move is to go back to the training use case but in a new domain (molecules, code, etc.).

4. **Is M2 (molecular direction) worth pursuing given J2's failure?** M2's core mechanism is the same — distributional shift from privileged information. If 3B models can't produce useful distributional shifts from gold math solutions, will they produce useful shifts from 3D pocket geometry? The counter-argument: molecular generation is a very different task where 3D structural information might create much stronger distributional shifts than text references.

5. **What other OPSD applications have genuine potential?** Beyond judge/eval and molecular design, are there domains where:
   - The privileged info creates a LARGE distributional shift (not marginal)
   - No simpler baseline (entropy, text similarity) can capture the same signal
   - The application doesn't require the method to beat strong supervised approaches

---

## Repo State

- `findings_0609_leaky_deprecated.md` — Round 1 results (INVALID, label leakage)
- `findings_0610_final.md` — This file (clean negative result)
- `exp_clean.py` — Clean experiment code (Round 3)
- `exp_nonleaky.py` — Non-leaky experiment code (Round 2)
- `results/clean/clean_summary.json` — Round 3 full results
- `results/nonleaky/nonleaky_summary.json` — Round 2 results

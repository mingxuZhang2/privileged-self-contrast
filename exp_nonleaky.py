"""
Non-Leaky Privileged Self-Contrast Experiments
================================================
Fix the label leakage issue: z must NOT contain gold error location.

Experiment A: final-answer-only z
  z = "The correct final answer is: {reference_answer}"
  No step info, no error boundary, no label.

Experiment B: independent gold-solution z (GSM8K/MATH only)
  z = reference solution from original dataset (not the candidate solution)

Experiment C: random-boundary marker control
  z = steps[:random_k] + "[error marker]" where k != gold label
  Tests whether TI follows the marker or the actual error.

Experiment D: trivial baseline for leaky z
  Just parse the marker position from z. Should be near-perfect.
  Proves the old results were inflated.
"""

import argparse
import json
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_solution_text(steps):
    parts = []
    boundaries = []
    pos = 0
    for i, step in enumerate(steps):
        step_text = step.strip()
        if i > 0:
            step_text = "\n" + step_text
            pos += 1
        boundaries.append((pos, pos + len(step_text.lstrip("\n"))))
        parts.append(step_text)
        pos += len(step_text.lstrip("\n"))
    return "".join(parts), boundaries


def compute_contrast(model, tokenizer, problem, steps, z_text, device):
    solution_text, char_boundaries = build_solution_text(steps)

    base_prompt = f"Problem: {problem}\n\nSolution:\n"
    teacher_prompt = f"Problem: {problem}\n\n{z_text}\n\nSolution:\n"

    full_student = base_prompt + solution_text
    full_teacher = teacher_prompt + solution_text

    student_enc = tokenizer(full_student, return_tensors="pt", truncation=True, max_length=2048).to(device)
    teacher_enc = tokenizer(full_teacher, return_tensors="pt", truncation=True, max_length=2048).to(device)

    base_prefix_len = tokenizer(base_prompt, return_tensors="pt").input_ids.shape[1]
    teacher_prefix_len = tokenizer(teacher_prompt, return_tensors="pt").input_ids.shape[1]

    solution_enc = tokenizer(solution_text, add_special_tokens=False, return_tensors="pt")
    n_solution_tokens = solution_enc.input_ids.shape[1]

    n_response = min(
        student_enc.input_ids.shape[1] - base_prefix_len,
        teacher_enc.input_ids.shape[1] - teacher_prefix_len,
        n_solution_tokens
    )
    if n_response <= 1:
        return None

    with torch.no_grad():
        s_out = model(**student_enc)
        t_out = model(**teacher_enc)

    s_logits = s_out.logits[0, base_prefix_len - 1: base_prefix_len - 1 + n_response]
    t_logits = t_out.logits[0, teacher_prefix_len - 1: teacher_prefix_len - 1 + n_response]

    s_probs = F.softmax(s_logits, dim=-1)
    t_probs = F.softmax(t_logits, dim=-1)
    s_log_probs = F.log_softmax(s_logits, dim=-1)
    t_log_probs = F.log_softmax(t_logits, dim=-1)

    info_gain = (t_probs * (t_log_probs - s_log_probs)).sum(dim=-1).cpu().float().numpy()

    actual_ids = student_enc.input_ids[0, base_prefix_len: base_prefix_len + n_response]
    s_actual = s_log_probs[range(n_response), actual_ids].cpu().float().numpy()
    t_actual = t_log_probs[range(n_response), actual_ids].cpu().float().numpy()
    token_incompat = s_actual - t_actual

    entropy = -(s_probs * s_log_probs).sum(dim=-1).cpu().float().numpy()

    # Map tokens to steps
    offsets = tokenizer(solution_text, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping'][:n_response]
    step_to_tokens = [[] for _ in range(len(steps))]
    for tok_idx, (start, end) in enumerate(offsets):
        for step_idx, (s_start, s_end) in enumerate(char_boundaries):
            if start >= s_start and start < s_end:
                step_to_tokens[step_idx].append(tok_idx)
                break

    step_ti_max = []
    step_ti_mean = []
    step_ig = []
    step_ent = []
    for tok_indices in step_to_tokens:
        if tok_indices:
            ti_vals = token_incompat[tok_indices]
            step_ti_max.append(float(np.max(ti_vals)))
            step_ti_mean.append(float(np.mean(ti_vals)))
            step_ig.append(float(np.mean(info_gain[tok_indices])))
            step_ent.append(float(np.mean(entropy[tok_indices])))
        else:
            step_ti_max.append(0.0)
            step_ti_mean.append(0.0)
            step_ig.append(0.0)
            step_ent.append(0.0)

    return {
        'step_ti_max': step_ti_max,
        'step_ti_mean': step_ti_mean,
        'step_ig': step_ig,
        'step_entropy': step_ent,
        'n_tokens': n_response,
        'truncated': n_response < n_solution_tokens,
    }


def compute_auroc(scores, labels):
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    auroc = 0
    for i in range(len(labels)):
        if labels[i] == 1:
            for j in range(len(labels)):
                if labels[j] == 0:
                    if scores[i] > scores[j]:
                        auroc += 1
                    elif scores[i] == scores[j]:
                        auroc += 0.5
    return auroc / (n_pos * n_neg)


def get_gsm8k_reference_answers():
    """Load GSM8K reference solutions from the original dataset."""
    ref = {}
    path = "gsm8k_test.jsonl"
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                ref[d['question'][:100]] = {
                    'answer': d['answer'],
                    'final_answer': d['final_answer']
                }
    return ref


def run(args):
    print(f"Loading model: {args.model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    device = model.device
    print(f"Model loaded on {device}", flush=True)

    all_data = []
    for fname in args.data_files:
        with open(fname) as f:
            data = json.load(f)
        source = Path(fname).stem
        for d in data:
            d['source'] = source
        all_data.extend(data)
        print(f"Loaded {fname}: {len(data)} examples", flush=True)

    if args.n_samples > 0:
        all_data = all_data[:args.n_samples]

    gsm8k_refs = get_gsm8k_reference_answers()
    random.seed(42)

    results = []
    for idx, ex in enumerate(all_data):
        problem = ex['problem']
        steps = ex['steps']
        label = ex['label']
        source = ex['source']
        n_steps = len(steps)

        if not steps or n_steps < 2:
            continue

        result = {
            'idx': idx, 'source': source, 'label': label,
            'n_steps': n_steps, 'has_error': label != -1,
        }

        # === Experiment A: Final-answer-only z ===
        # Try to get the reference final answer
        ref = gsm8k_refs.get(problem[:100])
        if ref:
            final_ans_z = f"Reference: The correct final answer to this problem is: {ref['final_answer']}"
        else:
            final_ans_z = "Reference: No specific final answer reference available."

        res_a = compute_contrast(model, tokenizer, problem, steps, final_ans_z, device)
        if res_a is None:
            continue
        result['exp_a_final_answer_z'] = res_a

        # === Experiment A2: Full reference solution z (GSM8K only) ===
        if ref and ref['answer']:
            gold_sol_z = f"Reference solution:\n{ref['answer']}"
            res_a2 = compute_contrast(model, tokenizer, problem, steps, gold_sol_z, device)
            if res_a2:
                result['exp_b_gold_solution_z'] = res_a2

        # === Experiment C: Random-boundary marker (leakage control) ===
        if label != -1 and n_steps >= 3:
            candidates = [k for k in range(n_steps) if k != label]
            random_k = random.choice(candidates)
            random_z = "\n".join(steps[:random_k]) + "\n[The solution has an error starting from this point.]"
            res_c = compute_contrast(model, tokenizer, problem, steps, random_z, device)
            if res_c:
                result['exp_c_random_boundary'] = res_c
                result['exp_c_random_k'] = random_k

        # === Old leaky z (for comparison, clearly labeled) ===
        if label == -1:
            leaky_z = "\n".join(steps)
        else:
            leaky_z = "\n".join(steps[:label]) + "\n[The solution has an error starting from this point.]"
        res_leaky = compute_contrast(model, tokenizer, problem, steps, leaky_z, device)
        if res_leaky:
            result['leaky_oracle_z'] = res_leaky

        # === Shuffled z (control) ===
        shuffled_z = "Reference: This problem has no simple closed-form solution. The answer is 42."
        res_shuf = compute_contrast(model, tokenizer, problem, steps, shuffled_z, device)
        if res_shuf:
            result['shuffled_z'] = res_shuf

        # === No z (entropy-only baseline) ===
        result['entropy_baseline'] = {
            'step_entropy': res_a['step_entropy'],
        }

        results.append(result)

        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  [{idx+1}/{len(all_data)}] source={source} label={label}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "nonleaky_results.json"), 'w') as f:
        json.dump(results, f)
    print(f"\nSaved {len(results)} results", flush=True)

    analyze(results, args.output_dir)


def analyze(results, output_dir):
    has_error = [r for r in results if r['has_error']]
    no_error = [r for r in results if not r['has_error']]

    print(f"\n{'='*80}", flush=True)
    print("NON-LEAKY EXPERIMENT ANALYSIS", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"Total: {len(results)} | Errors: {len(has_error)} | Correct: {len(no_error)}")

    # === Compare all z variants on step-level localization ===
    z_variants = [
        ('leaky_oracle_z', 'Leaky oracle z (gold boundary)'),
        ('exp_a_final_answer_z', 'Exp A: Final answer only z'),
        ('exp_b_gold_solution_z', 'Exp B: Gold solution z'),
        ('exp_c_random_boundary', 'Exp C: Random boundary marker'),
        ('shuffled_z', 'Shuffled z (control)'),
    ]

    print(f"\n{'='*80}")
    print("STEP-LEVEL ERROR LOCALIZATION (step_ti_max, erroneous samples)")
    print(f"{'='*80}")
    print(f"{'Z variant':45s} {'N':>5s} {'Exact':>7s} {'±1':>7s} {'StepAUROC':>10s}")
    print("-" * 80)

    summary = {}
    for z_key, z_label in z_variants:
        exact = 0
        within1 = 0
        step_aurocs = []
        total = 0

        for r in has_error:
            if z_key not in r:
                continue
            lab = r['label']
            scores = r[z_key]['step_ti_max']
            ns = r['n_steps']
            if ns < 2 or lab >= ns:
                continue
            total += 1
            pred = int(np.argmax(scores))
            if pred == lab:
                exact += 1
            if abs(pred - lab) <= 1:
                within1 += 1
            step_labels = [1 if i == lab else 0 for i in range(ns)]
            step_aurocs.append(compute_auroc(scores, step_labels))

        if total > 0:
            em = exact / total * 100
            w1 = within1 / total * 100
            sa = np.mean(step_aurocs)
            print(f"  {z_label:43s} {total:5d} {em:7.1f} {w1:7.1f} {sa:10.4f}")
            summary[z_key] = {'exact': em, 'within1': w1, 'step_auroc': sa, 'n': total}

    # === Experiment C: Does TI follow the marker or the real error? ===
    print(f"\n{'='*80}")
    print("EXP C: RANDOM BOUNDARY MARKER CONTROL")
    print(f"{'='*80}")

    follows_marker = 0
    follows_real = 0
    follows_neither = 0
    total_c = 0

    for r in has_error:
        if 'exp_c_random_boundary' not in r:
            continue
        lab = r['label']
        random_k = r['exp_c_random_k']
        scores = r['exp_c_random_boundary']['step_ti_max']
        ns = r['n_steps']
        if ns < 2 or lab >= ns:
            continue
        total_c += 1
        pred = int(np.argmax(scores))
        if pred == random_k:
            follows_marker += 1
        elif pred == lab:
            follows_real += 1
        else:
            follows_neither += 1

    if total_c > 0:
        print(f"  Total samples: {total_c}")
        print(f"  TI argmax follows RANDOM marker: {follows_marker/total_c*100:.1f}%")
        print(f"  TI argmax follows REAL error:    {follows_real/total_c*100:.1f}%")
        print(f"  TI argmax follows NEITHER:       {follows_neither/total_c*100:.1f}%")
        print()
        if follows_marker > follows_real:
            print("  >> WARNING: TI mostly follows the marker, not the real error!")
            print("  >> The leaky z results are primarily marker-driven.")
        else:
            print("  >> GOOD: TI follows real error more than random marker.")

    # === Sequence-level AUROC comparison ===
    print(f"\n{'='*80}")
    print("SEQUENCE-LEVEL AUROC (detecting erroneous vs correct solutions)")
    print(f"{'='*80}")

    for z_key, z_label in z_variants:
        scores_seq = []
        labels_seq = []
        for r in results:
            if z_key not in r:
                continue
            scores_seq.append(max(r[z_key]['step_ti_max']))
            labels_seq.append(1 if r['has_error'] else 0)
        if scores_seq:
            auroc = compute_auroc(scores_seq, labels_seq)
            print(f"  {z_label:43s} AUROC={auroc:.4f} (n={len(scores_seq)})")

    # Entropy baseline
    ent_scores = [max(r['exp_a_final_answer_z']['step_entropy']) for r in results if 'exp_a_final_answer_z' in r]
    ent_labels = [1 if r['has_error'] else 0 for r in results if 'exp_a_final_answer_z' in r]
    auroc_ent = compute_auroc(ent_scores, ent_labels)
    print(f"  {'Entropy baseline (no z)':43s} AUROC={auroc_ent:.4f}")

    # === By source breakdown for Exp A ===
    print(f"\n{'='*80}")
    print("EXP A (FINAL ANSWER z) BREAKDOWN BY SOURCE")
    print(f"{'='*80}")

    sources = set(r['source'] for r in results)
    for src in sorted(sources):
        src_err = [r for r in results if r['source'] == src and r['has_error'] and 'exp_a_final_answer_z' in r]
        aurocs = []
        exact = 0
        for r in src_err:
            lab = r['label']
            scores = r['exp_a_final_answer_z']['step_ti_max']
            if r['n_steps'] < 2 or lab >= r['n_steps']:
                continue
            if int(np.argmax(scores)) == lab:
                exact += 1
            step_labels = [1 if i == lab else 0 for i in range(r['n_steps'])]
            aurocs.append(compute_auroc(scores, step_labels))
        n = len(aurocs)
        if n > 0:
            print(f"  {src:15s}: n={n:4d}, Exact={exact/n*100:.1f}%, StepAUROC={np.mean(aurocs):.4f}")

    # === KILL TEST for non-leaky z ===
    print(f"\n{'='*80}")
    print("KILL TEST: Non-leaky z")
    print(f"{'='*80}")

    if 'exp_a_final_answer_z' in summary:
        sa = summary['exp_a_final_answer_z']['step_auroc']
        sa_shuf = summary.get('shuffled_z', {}).get('step_auroc', 0.5)
        delta = sa - sa_shuf
        print(f"  Exp A (final answer z) step AUROC: {sa:.4f}")
        print(f"  Shuffled z step AUROC:             {sa_shuf:.4f}")
        print(f"  Delta:                             {delta:.4f}")
        if delta > 0.08:
            print(f"  >> PASS: final-answer z provides +{delta*100:.1f}pp over shuffled")
        elif delta > 0.03:
            print(f"  >> MARGINAL: +{delta*100:.1f}pp, needs more investigation")
        else:
            print(f"  >> FAIL: final-answer z does not meaningfully outperform shuffled")

    # Save summary
    with open(os.path.join(output_dir, "nonleaky_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nAnalysis complete.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="~/data/models_dl/Qwen2.5-3B-Instruct")
    parser.add_argument("--data_files", nargs="+", default=[
        "data/processbench/gsm8k.json",
        "data/processbench/math.json",
    ])
    parser.add_argument("--output_dir", type=str, default="results/nonleaky")
    parser.add_argument("--n_samples", type=int, default=0)
    args = parser.parse_args()
    args.model_path = os.path.expanduser(args.model_path)
    run(args)

"""
ProcessBench Experiment: Step-Level Error Localization via Privileged Self-Contrast
====================================================================================
Given a problem + step-by-step solution (from ProcessBench), compute token-level
self-contrast scores and aggregate to step level. Compare with human-annotated
earliest error step.

Metrics:
  - Step-level AUROC for error detection
  - Earliest-error-step accuracy (does max-score step = annotated error step?)
  - Correct vs incorrect solution discrimination
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_solution_text(steps):
    """Join steps into a single solution string, tracking step boundaries."""
    parts = []
    boundaries = []  # (start_char, end_char) for each step
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


def compute_self_contrast_steps(model, tokenizer, problem, steps, z_text, device):
    """
    Compute token-level self-contrast, then aggregate to step level.

    Returns dict with:
      - step_info_gain: [n_steps] mean InfoGain per step
      - step_token_incompat: [n_steps] max TokenIncompat per step
      - step_mean_ti: [n_steps] mean TokenIncompat per step
      - token_info_gain: raw token-level
      - token_incompat: raw token-level
      - token_entropy: raw token-level entropy
      - step_to_token_map: which tokens belong to which step
    """
    solution_text, char_boundaries = build_solution_text(steps)

    base_prompt = f"Problem: {problem}\n\nSolution:\n"
    teacher_prompt = f"Problem: {problem}\n\nCorrect solution for reference:\n{z_text}\n\nSolution:\n"

    full_student = base_prompt + solution_text
    full_teacher = teacher_prompt + solution_text

    student_enc = tokenizer(full_student, return_tensors="pt", truncation=True, max_length=2048).to(device)
    teacher_enc = tokenizer(full_teacher, return_tensors="pt", truncation=True, max_length=2048).to(device)

    base_prefix_enc = tokenizer(base_prompt, return_tensors="pt")
    teacher_prefix_enc = tokenizer(teacher_prompt, return_tensors="pt")
    base_prefix_len = base_prefix_enc.input_ids.shape[1]
    teacher_prefix_len = teacher_prefix_enc.input_ids.shape[1]

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

    # Map tokens to steps via character offsets
    token_offsets = tokenizer(solution_text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = token_offsets['offset_mapping'][:n_response]

    step_to_tokens = [[] for _ in range(len(steps))]
    for tok_idx, (start, end) in enumerate(offsets):
        for step_idx, (s_start, s_end) in enumerate(char_boundaries):
            if start >= s_start and start < s_end:
                step_to_tokens[step_idx].append(tok_idx)
                break

    step_ig = []
    step_ti_max = []
    step_ti_mean = []
    step_ent = []
    for tok_indices in step_to_tokens:
        if tok_indices:
            ig_vals = info_gain[tok_indices]
            ti_vals = token_incompat[tok_indices]
            ent_vals = entropy[tok_indices]
            step_ig.append(float(np.mean(ig_vals)))
            step_ti_max.append(float(np.max(ti_vals)))
            step_ti_mean.append(float(np.mean(ti_vals)))
            step_ent.append(float(np.mean(ent_vals)))
        else:
            step_ig.append(0.0)
            step_ti_max.append(0.0)
            step_ti_mean.append(0.0)
            step_ent.append(0.0)

    return {
        'step_info_gain': step_ig,
        'step_ti_max': step_ti_max,
        'step_ti_mean': step_ti_mean,
        'step_entropy': step_ent,
        'step_to_tokens': [[int(x) for x in toks] for toks in step_to_tokens],
        'n_tokens': n_response,
    }


def compute_auroc(scores, labels):
    """Binary AUROC. labels=1 means positive (error step)."""
    pairs = list(zip(scores, labels))
    n_pos = sum(l for _, l in pairs)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    auroc = 0
    for i in range(len(pairs)):
        if pairs[i][1] == 1:
            for j in range(len(pairs)):
                if pairs[j][1] == 0:
                    if pairs[i][0] > pairs[j][0]:
                        auroc += 1
                    elif pairs[i][0] == pairs[j][0]:
                        auroc += 0.5
    return auroc / (n_pos * n_neg)


def run_experiment(args):
    print(f"Loading model: {args.model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    device = model.device
    print(f"Model loaded on {device}", flush=True)

    # Load ProcessBench data
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
    print(f"Total: {len(all_data)} examples", flush=True)

    results = []
    for idx, ex in enumerate(all_data):
        problem = ex['problem']
        steps = ex['steps']
        label = ex['label']  # -1 = all correct, else 0-based error step index
        source = ex['source']

        if not steps or len(steps) < 2:
            continue

        # z variants:
        # 1) answer_z: just the error indication (partial correct steps + error marker)
        if label == -1:
            answer_z = "\n".join(steps)
        else:
            answer_z = "\n".join(steps[:label]) + "\n[The solution has an error starting from this point.]"

        # 2) full_z: the entire solution text as z (like "seeing the answer sheet")
        full_z = "\n".join(steps)

        # 3) shuffled_z: irrelevant reference
        shuffled_z = "This problem has no simple closed-form solution. The answer is 42."

        # Compute with answer z (partial correct info)
        res_answer = compute_self_contrast_steps(model, tokenizer, problem, steps, answer_z, device)
        if res_answer is None:
            continue

        # Compute with full solution z
        res_full = compute_self_contrast_steps(model, tokenizer, problem, steps, full_z, device)

        # Compute with shuffled z
        res_shuf = compute_self_contrast_steps(model, tokenizer, problem, steps, shuffled_z, device)

        result = {
            'idx': idx,
            'source': source,
            'label': label,
            'n_steps': len(steps),
            'has_error': label != -1,
            'answer_z': {
                'step_ig': res_answer['step_info_gain'],
                'step_ti_max': res_answer['step_ti_max'],
                'step_ti_mean': res_answer['step_ti_mean'],
                'step_entropy': res_answer['step_entropy'],
            },
            'true_z': {
                'step_ig': res_answer['step_info_gain'],
                'step_ti_max': res_answer['step_ti_max'],
                'step_ti_mean': res_answer['step_ti_mean'],
                'step_entropy': res_answer['step_entropy'],
            },
        }

        if res_full is not None:
            result['full_z'] = {
                'step_ig': res_full['step_info_gain'],
                'step_ti_max': res_full['step_ti_max'],
                'step_ti_mean': res_full['step_ti_mean'],
                'step_entropy': res_full['step_entropy'],
            }

        if res_shuf is not None:
            result['shuf_z'] = {
                'step_ig': res_shuf['step_info_gain'],
                'step_ti_max': res_shuf['step_ti_max'],
                'step_ti_mean': res_shuf['step_ti_mean'],
                'step_entropy': res_shuf['step_entropy'],
            }

        results.append(result)

        if (idx + 1) % 50 == 0 or idx == 0:
            n_done = len(results)
            print(f"  [{idx+1}/{len(all_data)}] processed={n_done}, "
                  f"source={source}, label={label}, n_steps={len(steps)}", flush=True)

    # Save raw results
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "processbench_results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f)
    print(f"\nSaved {len(results)} results to {out_path}", flush=True)

    analyze(results, args.output_dir)


def analyze(results, output_dir):
    """Comprehensive analysis of ProcessBench results."""
    print(f"\n{'='*70}", flush=True)
    print("PROCESSBENCH ANALYSIS", flush=True)
    print(f"{'='*70}", flush=True)

    has_error = [r for r in results if r['has_error']]
    no_error = [r for r in results if not r['has_error']]
    print(f"Total: {len(results)} | Has error: {len(has_error)} | No error: {len(no_error)}")

    # --- 1. Sequence-level: can we tell erroneous from correct solutions? ---
    print(f"\n{'='*70}")
    print("1. SEQUENCE-LEVEL DISCRIMINATION (erroneous vs correct solutions)")
    print(f"{'='*70}")

    for score_name in ['step_ti_max', 'step_ti_mean', 'step_ig', 'step_entropy']:
        all_scores = []
        all_labels = []
        for r in results:
            vals = r['true_z'][score_name]
            all_scores.append(max(vals))  # max step score as sequence summary
            all_labels.append(1 if r['has_error'] else 0)

        auroc = compute_auroc(all_scores, all_labels)
        mean_err = np.mean([s for s, l in zip(all_scores, all_labels) if l == 1])
        mean_ok = np.mean([s for s, l in zip(all_scores, all_labels) if l == 0])
        print(f"  {score_name:15s}: AUROC={auroc:.4f}  mean(error)={mean_err:.4f}  mean(correct)={mean_ok:.4f}")

    # Shuffled z baseline
    shuf_scores = []
    shuf_labels = []
    for r in results:
        if 'shuf_z' not in r:
            continue
        shuf_scores.append(max(r['shuf_z']['step_ti_max']))
        shuf_labels.append(1 if r['has_error'] else 0)
    if shuf_scores:
        auroc_shuf = compute_auroc(shuf_scores, shuf_labels)
        print(f"  {'shuf_ti_max':15s}: AUROC={auroc_shuf:.4f}  (shuffled z baseline)")

    # Entropy baseline
    ent_scores = []
    ent_labels = []
    for r in results:
        ent_scores.append(max(r['true_z']['step_entropy']))
        ent_labels.append(1 if r['has_error'] else 0)
    auroc_ent = compute_auroc(ent_scores, ent_labels)
    print(f"  {'entropy':15s}: AUROC={auroc_ent:.4f}  (entropy baseline)")

    # --- 2. Step-level: can we identify the error step? ---
    print(f"\n{'='*70}")
    print("2. STEP-LEVEL ERROR LOCALIZATION (within erroneous solutions)")
    print(f"{'='*70}")

    for score_name in ['step_ti_max', 'step_ti_mean', 'step_ig']:
        # Earliest error step accuracy: is argmax(score) == label?
        exact_match = 0
        within_1 = 0
        step_aurocs = []
        total_valid = 0

        for r in has_error:
            label = r['label']
            scores = r['true_z'][score_name]
            n_steps = r['n_steps']

            if n_steps < 2 or label >= n_steps:
                continue

            total_valid += 1
            pred_step = int(np.argmax(scores))
            if pred_step == label:
                exact_match += 1
            if abs(pred_step - label) <= 1:
                within_1 += 1

            # Step-level AUROC: is the error step ranked higher than non-error steps?
            step_labels = [1 if i == label else 0 for i in range(n_steps)]
            step_auroc = compute_auroc(scores, step_labels)
            step_aurocs.append(step_auroc)

        if total_valid > 0:
            print(f"\n  {score_name} (n={total_valid}):")
            print(f"    Exact match (argmax = error step): {exact_match/total_valid*100:.1f}%")
            print(f"    Within ±1 step:                    {within_1/total_valid*100:.1f}%")
            print(f"    Mean step-level AUROC:             {np.mean(step_aurocs):.4f}")

        # Same for shuffled z
        exact_shuf = 0
        aurocs_shuf = []
        total_shuf = 0
        for r in has_error:
            if 'shuf_z' not in r:
                continue
            label = r['label']
            scores = r['shuf_z'][score_name]
            n_steps = r['n_steps']
            if n_steps < 2 or label >= n_steps:
                continue
            total_shuf += 1
            if int(np.argmax(scores)) == label:
                exact_shuf += 1
            step_labels = [1 if i == label else 0 for i in range(n_steps)]
            aurocs_shuf.append(compute_auroc(scores, step_labels))

        if total_shuf > 0:
            print(f"    [Shuffled z] Exact match: {exact_shuf/total_shuf*100:.1f}%, "
                  f"Mean AUROC: {np.mean(aurocs_shuf):.4f}")

    # --- 2b. Full solution z vs answer z ---
    print(f"\n{'='*70}")
    print("2b. FULL SOLUTION z vs ANSWER-ONLY z (step-level localization)")
    print(f"{'='*70}")

    for z_key, z_label in [('answer_z', 'Answer z'), ('full_z', 'Full sol z'), ('shuf_z', 'Shuffled z')]:
        exact = 0
        within1 = 0
        s_aurocs = []
        total = 0
        for r in has_error:
            if z_key not in r:
                continue
            label = r['label']
            scores = r[z_key]['step_ti_max']
            n_steps = r['n_steps']
            if n_steps < 2 or label >= n_steps:
                continue
            total += 1
            pred = int(np.argmax(scores))
            if pred == label:
                exact += 1
            if abs(pred - label) <= 1:
                within1 += 1
            step_labels = [1 if i == label else 0 for i in range(n_steps)]
            s_aurocs.append(compute_auroc(scores, step_labels))
        if total > 0:
            print(f"  {z_label:15s} (n={total}): Exact={exact/total*100:.1f}%, ±1={within1/total*100:.1f}%, StepAUROC={np.mean(s_aurocs):.4f}")

    # --- 3. Breakdown by source ---
    print(f"\n{'='*70}")
    print("3. BREAKDOWN BY SOURCE")
    print(f"{'='*70}")

    sources = set(r['source'] for r in results)
    for source in sorted(sources):
        src_results = [r for r in results if r['source'] == source]
        src_errors = [r for r in src_results if r['has_error']]

        seq_scores = [max(r['true_z']['step_ti_max']) for r in src_results]
        seq_labels = [1 if r['has_error'] else 0 for r in src_results]
        auroc = compute_auroc(seq_scores, seq_labels)

        step_aurocs = []
        for r in src_errors:
            if r['n_steps'] < 2 or r['label'] >= r['n_steps']:
                continue
            labels = [1 if i == r['label'] else 0 for i in range(r['n_steps'])]
            step_aurocs.append(compute_auroc(r['true_z']['step_ti_max'], labels))

        mean_step = np.mean(step_aurocs) if step_aurocs else 0
        print(f"  {source:15s}: n={len(src_results):4d}, errors={len(src_errors):4d}, "
              f"seq_AUROC={auroc:.4f}, step_AUROC={mean_step:.4f}")

    # --- 4. Kill test ---
    print(f"\n{'='*70}")
    print("4. KILL TEST SUMMARY")
    print(f"{'='*70}")

    # Get best true z AUROC and shuffled z AUROC
    all_true = [max(r['true_z']['step_ti_max']) for r in results]
    all_true_labels = [1 if r['has_error'] else 0 for r in results]
    auroc_true = compute_auroc(all_true, all_true_labels)

    all_shuf2 = []
    all_shuf2_labels = []
    for r in results:
        if 'shuf_z' in r:
            all_shuf2.append(max(r['shuf_z']['step_ti_max']))
            all_shuf2_labels.append(1 if r['has_error'] else 0)
    auroc_shuf2 = compute_auroc(all_shuf2, all_shuf2_labels) if all_shuf2 else 0

    delta = auroc_true - auroc_shuf2
    delta_ent = auroc_true - auroc_ent

    print(f"  Sequence-level AUROC (true z):      {auroc_true:.4f}")
    print(f"  Sequence-level AUROC (shuffled z):   {auroc_shuf2:.4f}")
    print(f"  Sequence-level AUROC (entropy):      {auroc_ent:.4f}")
    print(f"  Delta (true - shuffled):             {delta:.4f} ({'PASS' if delta > 0.08 else 'FAIL'})")
    print(f"  Delta (true - entropy):              {delta_ent:.4f} ({'PASS' if delta_ent > 0.05 else 'FAIL'})")

    # Save analysis
    analysis = {
        'n_total': len(results),
        'n_error': len(has_error),
        'n_correct': len(no_error),
        'auroc_true_z': auroc_true,
        'auroc_shuffled_z': auroc_shuf2,
        'auroc_entropy': auroc_ent,
        'delta_true_vs_shuffled': delta,
        'delta_true_vs_entropy': delta_ent,
    }
    with open(os.path.join(output_dir, "analysis_summary.json"), 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"\nAnalysis complete.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="~/data/models_dl/Qwen2.5-3B-Instruct")
    parser.add_argument("--data_files", nargs="+", default=[
        "data/processbench/gsm8k.json",
        "data/processbench/math.json",
    ])
    parser.add_argument("--output_dir", type=str, default="results/processbench")
    parser.add_argument("--n_samples", type=int, default=0)
    args = parser.parse_args()
    args.model_path = os.path.expanduser(args.model_path)
    run_experiment(args)

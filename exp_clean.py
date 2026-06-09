"""
Clean Non-Leaky Experiment v2
==============================
Fixes all issues identified by GPT Pro:
1. Token slicing: use manual prefix_ids + solution_ids concatenation
2. Reference matching: normalized full-question matching with match rate tracking
3. Separate GSM8K and MATH reporting (no mixed has_ref/no_ref)
4. MATH references loaded from hendrycks_math dataset
5. Truncation tracking
6. All stats saved to JSON
7. Extra controls: wrong-answer z, another-example z, textual alignment baseline
"""

import argparse
import json
import os
import random
import re
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from difflib import SequenceMatcher


def norm_text(s):
    return " ".join(str(s).replace("’", "'").replace("‘", "'").split())


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


def compute_contrast_safe(model, tokenizer, problem, steps, z_text, device):
    """
    Safe token-level contrast with manual ID concatenation.
    Avoids BPE boundary merge issues.
    """
    solution_text, char_boundaries = build_solution_text(steps)

    base_prefix = f"Problem: {problem}\n\nSolution:\n"
    teacher_prefix = f"Problem: {problem}\n\n{z_text}\n\nSolution:\n"

    # Manual tokenization: prefix_ids + solution_ids
    base_prefix_ids = tokenizer(base_prefix, add_special_tokens=True).input_ids
    teacher_prefix_ids = tokenizer(teacher_prefix, add_special_tokens=True).input_ids
    solution_ids = tokenizer(solution_text, add_special_tokens=False).input_ids

    max_len = 2048
    base_total = base_prefix_ids + solution_ids
    teacher_total = teacher_prefix_ids + solution_ids

    base_prefix_len = len(base_prefix_ids)
    teacher_prefix_len = len(teacher_prefix_ids)

    # Truncate solution if needed
    truncated = False
    if len(base_total) > max_len:
        n_avail = max_len - base_prefix_len
        solution_ids_used = solution_ids[:n_avail]
        truncated = True
    else:
        solution_ids_used = solution_ids

    if len(teacher_total) > max_len:
        n_avail_t = max_len - teacher_prefix_len
        solution_ids_teacher = solution_ids[:n_avail_t]
        truncated = True
    else:
        solution_ids_teacher = solution_ids

    n_response = min(len(solution_ids_used), len(solution_ids_teacher))
    if n_response <= 1:
        return None

    # Build input tensors
    s_ids = torch.tensor([base_prefix_ids + solution_ids_used[:n_response]], device=device)
    t_ids = torch.tensor([teacher_prefix_ids + solution_ids_teacher[:n_response]], device=device)

    with torch.no_grad():
        s_out = model(input_ids=s_ids)
        t_out = model(input_ids=t_ids)

    # Logits for response positions
    s_logits = s_out.logits[0, base_prefix_len - 1: base_prefix_len - 1 + n_response]
    t_logits = t_out.logits[0, teacher_prefix_len - 1: teacher_prefix_len - 1 + n_response]

    s_probs = F.softmax(s_logits, dim=-1)
    t_probs = F.softmax(t_logits, dim=-1)
    s_log_probs = F.log_softmax(s_logits, dim=-1)
    t_log_probs = F.log_softmax(t_logits, dim=-1)

    info_gain = (t_probs * (t_log_probs - s_log_probs)).sum(dim=-1).cpu().float().numpy()

    actual_ids = torch.tensor(solution_ids_used[:n_response], device=device)
    s_actual = s_log_probs[range(n_response), actual_ids].cpu().float().numpy()
    t_actual = t_log_probs[range(n_response), actual_ids].cpu().float().numpy()
    token_incompat = s_actual - t_actual

    entropy = -(s_probs * s_log_probs).sum(dim=-1).cpu().float().numpy()

    # Map tokens to steps via character offsets
    offsets = tokenizer(solution_text, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping'][:n_response]
    step_to_tokens = [[] for _ in range(len(steps))]
    for tok_idx, (start, end) in enumerate(offsets):
        for step_idx, (s_start, s_end) in enumerate(char_boundaries):
            if start >= s_start and start < s_end:
                step_to_tokens[step_idx].append(tok_idx)
                break

    step_ti_max = []
    step_ti_mean = []
    step_ti_top3 = []
    step_ig = []
    step_ent = []
    for tok_indices in step_to_tokens:
        if tok_indices:
            ti_vals = token_incompat[tok_indices]
            pos_ti = ti_vals[ti_vals > 0]
            step_ti_max.append(float(np.max(ti_vals)))
            step_ti_mean.append(float(np.mean(ti_vals)))
            top3 = sorted(ti_vals, reverse=True)[:3]
            step_ti_top3.append(float(np.mean(top3)))
            step_ig.append(float(np.mean(info_gain[tok_indices])))
            step_ent.append(float(np.mean(entropy[tok_indices])))
        else:
            step_ti_max.append(0.0)
            step_ti_mean.append(0.0)
            step_ti_top3.append(0.0)
            step_ig.append(0.0)
            step_ent.append(0.0)

    # Check if gold error step is within scored range
    scored_steps = sum(1 for toks in step_to_tokens if toks)

    return {
        'step_ti_max': step_ti_max,
        'step_ti_mean': step_ti_mean,
        'step_ti_top3': step_ti_top3,
        'step_ig': step_ig,
        'step_entropy': step_ent,
        'n_tokens': n_response,
        'n_solution_tokens': len(solution_ids),
        'truncated': truncated,
        'scored_steps': scored_steps,
    }


def textual_alignment_baseline(steps, ref_solution):
    """Simple textual alignment: find where candidate first diverges from reference."""
    ref_steps = ref_solution.split('\n')
    ref_steps = [s.strip() for s in ref_steps if s.strip()]

    scores = []
    for i, step in enumerate(steps):
        best_sim = 0
        for rs in ref_steps:
            sim = SequenceMatcher(None, step.lower(), rs.lower()).ratio()
            best_sim = max(best_sim, sim)
        scores.append(1.0 - best_sim)
    return scores


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


def run(args):
    print(f"Loading model: {args.model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    device = model.device
    print(f"Model on {device}", flush=True)

    # Load references
    gsm8k_refs = {}
    with open("gsm8k_test.jsonl") as f:
        for line in f:
            d = json.loads(line)
            gsm8k_refs[norm_text(d['question'])] = {
                'solution': d['answer'],
                'final_answer': d['final_answer'],
            }
    print(f"GSM8K refs: {len(gsm8k_refs)}", flush=True)

    math_refs = {}
    math_ref_path = "data/math_references.json"
    if os.path.exists(math_ref_path):
        with open(math_ref_path) as f:
            math_refs = json.load(f)
    print(f"MATH refs: {len(math_refs)}", flush=True)

    # Load ProcessBench data
    all_data = []
    for fname in args.data_files:
        with open(fname) as f:
            data = json.load(f)
        source = Path(fname).stem
        for d in data:
            d['source'] = source
        all_data.extend(data)

    if args.n_samples > 0:
        all_data = all_data[:args.n_samples]

    random.seed(42)
    other_answers = []
    for ex in all_data:
        ref = gsm8k_refs.get(norm_text(ex['problem'])) or math_refs.get(norm_text(ex['problem']))
        if ref and 'final_answer' in ref:
            other_answers.append(ref['final_answer'])
        elif ref and 'solution' in ref:
            sol = ref['solution']
            nums = re.findall(r'[\-]?\d+\.?\d*', sol)
            other_answers.append(nums[-1] if nums else "0")

    results = []
    stats = {'gsm8k_matched': 0, 'math_matched': 0, 'no_ref': 0, 'truncated': 0}

    for idx, ex in enumerate(all_data):
        problem = ex['problem']
        steps = ex['steps']
        label = ex['label']
        source = ex['source']
        n_steps = len(steps)

        if not steps or n_steps < 2:
            continue

        prob_norm = norm_text(problem)
        ref = gsm8k_refs.get(prob_norm) or math_refs.get(prob_norm)
        has_ref = ref is not None

        if has_ref:
            if source == 'gsm8k':
                stats['gsm8k_matched'] += 1
            else:
                stats['math_matched'] += 1
            final_ans = ref.get('final_answer', '')
            if not final_ans:
                sol = ref.get('solution', '')
                nums = re.findall(r'[\-]?\d+\.?\d*', sol)
                final_ans = nums[-1] if nums else ""
            gold_solution = ref.get('solution', '') or ref.get('answer', '')
        else:
            stats['no_ref'] += 1
            final_ans = ""
            gold_solution = ""

        result = {
            'idx': idx, 'source': source, 'label': label,
            'n_steps': n_steps, 'has_error': label != -1, 'has_ref': has_ref,
        }

        # === Z1: Final answer only (non-leaky) ===
        if final_ans:
            z1 = f"Reference: The correct final answer is: {final_ans}"
        else:
            continue  # Skip samples without reference
        res_z1 = compute_contrast_safe(model, tokenizer, problem, steps, z1, device)
        if res_z1 is None:
            continue
        result['final_answer_z'] = res_z1
        if res_z1['truncated']:
            stats['truncated'] += 1

        # === Z2: Gold solution (non-leaky) ===
        if gold_solution:
            z2 = f"Reference solution:\n{gold_solution}"
            res_z2 = compute_contrast_safe(model, tokenizer, problem, steps, z2, device)
            if res_z2:
                result['gold_solution_z'] = res_z2

        # === Z3: Wrong final answer (control) ===
        wrong_ans = str(float(final_ans) + 42) if final_ans and final_ans.replace('.','').replace('-','').isdigit() else "999"
        z3 = f"Reference: The correct final answer is: {wrong_ans}"
        res_z3 = compute_contrast_safe(model, tokenizer, problem, steps, z3, device)
        if res_z3:
            result['wrong_answer_z'] = res_z3

        # === Z4: Another example's answer (control) ===
        other_idx = (idx + 137) % len(other_answers) if other_answers else 0
        other_ans = other_answers[other_idx] if other_answers else "0"
        z4 = f"Reference: The correct final answer is: {other_ans}"
        res_z4 = compute_contrast_safe(model, tokenizer, problem, steps, z4, device)
        if res_z4:
            result['other_answer_z'] = res_z4

        # === Z5: Another example's gold solution (control) ===
        if gold_solution:
            other_prob_idx = (idx + 137) % len(all_data)
            other_prob = all_data[other_prob_idx]
            other_ref = gsm8k_refs.get(norm_text(other_prob['problem'])) or math_refs.get(norm_text(other_prob['problem']))
            if other_ref:
                other_sol = other_ref.get('solution', '') or other_ref.get('answer', '')
                if other_sol:
                    z5 = f"Reference solution:\n{other_sol}"
                    res_z5 = compute_contrast_safe(model, tokenizer, problem, steps, z5, device)
                    if res_z5:
                        result['other_solution_z'] = res_z5

        # === Z6: Shuffled (control) ===
        z6 = "Reference: This problem has no simple closed-form solution."
        res_z6 = compute_contrast_safe(model, tokenizer, problem, steps, z6, device)
        if res_z6:
            result['shuffled_z'] = res_z6

        # === Textual alignment baseline (for gold_solution only) ===
        if gold_solution and result['has_error']:
            ta_scores = textual_alignment_baseline(steps, gold_solution)
            result['textual_alignment'] = ta_scores

        results.append(result)
        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  [{idx+1}/{len(all_data)}] src={source} ref={has_ref} label={label}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "results.json"), 'w') as f:
        json.dump(results, f)
    print(f"\nSaved {len(results)} results. Stats: {stats}", flush=True)

    analyze(results, stats, args.output_dir)


def analyze(results, stats, output_dir):
    print(f"\n{'='*80}", flush=True)
    print("CLEAN EXPERIMENT ANALYSIS", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"Total: {len(results)} (all have reference)")
    print(f"Match stats: {stats}")

    z_variants = [
        ('final_answer_z', 'Final answer z'),
        ('gold_solution_z', 'Gold solution z'),
        ('wrong_answer_z', 'Wrong answer z (control)'),
        ('other_answer_z', 'Other example answer z (control)'),
        ('other_solution_z', 'Other example solution z (control)'),
        ('shuffled_z', 'Shuffled z (control)'),
    ]

    agg_methods = ['step_ti_max', 'step_ti_top3', 'step_ti_mean']
    summary = {}

    for source in ['gsm8k', 'math', 'ALL']:
        if source == 'ALL':
            subset = results
        else:
            subset = [r for r in results if r['source'] == source]

        has_err = [r for r in subset if r['has_error']]
        no_err = [r for r in subset if not r['has_error']]

        print(f"\n{'='*80}")
        print(f"SOURCE: {source} (n={len(subset)}, errors={len(has_err)}, correct={len(no_err)})")
        print(f"{'='*80}")

        # Step-level localization
        print(f"\n  Step-level error localization (erroneous samples):")
        print(f"  {'Z variant':40s} {'Agg':>10s} {'N':>5s} {'Exact':>7s} {'±1':>7s} {'AUROC':>7s}")
        print(f"  {'-'*75}")

        for z_key, z_label in z_variants:
            for agg in agg_methods:
                exact = within1 = 0
                aurocs = []
                total = 0
                for r in has_err:
                    if z_key not in r:
                        continue
                    lab = r['label']
                    scores = r[z_key][agg]
                    ns = r['n_steps']
                    if ns < 2 or lab >= ns:
                        continue
                    if lab >= r[z_key]['scored_steps']:
                        continue
                    total += 1
                    pred = int(np.argmax(scores))
                    if pred == lab:
                        exact += 1
                    if abs(pred - lab) <= 1:
                        within1 += 1
                    step_labels = [1 if i == lab else 0 for i in range(ns)]
                    aurocs.append(compute_auroc(scores, step_labels))

                if total > 0 and agg == 'step_ti_max':
                    em = exact / total * 100
                    w1 = within1 / total * 100
                    sa = np.mean(aurocs)
                    print(f"  {z_label:40s} {agg:>10s} {total:5d} {em:7.1f} {w1:7.1f} {sa:7.4f}")
                    summary[f"{source}_{z_key}_{agg}"] = {
                        'exact': em, 'within1': w1, 'step_auroc': sa, 'n': total
                    }

                if total > 0 and agg == 'step_ti_top3' and z_key in ['final_answer_z', 'gold_solution_z']:
                    em = exact / total * 100
                    sa = np.mean(aurocs)
                    print(f"  {z_label:40s} {agg:>10s} {total:5d} {em:7.1f} {'':>7s} {sa:7.4f}")

        # Entropy baseline
        total_ent = exact_ent = 0
        aurocs_ent = []
        for r in has_err:
            if 'final_answer_z' not in r:
                continue
            lab = r['label']
            scores = r['final_answer_z']['step_entropy']
            ns = r['n_steps']
            if ns < 2 or lab >= ns:
                continue
            total_ent += 1
            if int(np.argmax(scores)) == lab:
                exact_ent += 1
            step_labels = [1 if i == lab else 0 for i in range(ns)]
            aurocs_ent.append(compute_auroc(scores, step_labels))
        if total_ent > 0:
            print(f"  {'Entropy (no z)':40s} {'entropy':>10s} {total_ent:5d} {exact_ent/total_ent*100:7.1f} {'':>7s} {np.mean(aurocs_ent):7.4f}")
            summary[f"{source}_entropy"] = {'exact': exact_ent/total_ent*100, 'step_auroc': np.mean(aurocs_ent), 'n': total_ent}

        # Textual alignment baseline
        ta_err = [r for r in has_err if 'textual_alignment' in r]
        if ta_err:
            exact_ta = 0
            aurocs_ta = []
            for r in ta_err:
                lab = r['label']
                scores = r['textual_alignment']
                ns = r['n_steps']
                if ns < 2 or lab >= ns or lab >= len(scores):
                    continue
                if int(np.argmax(scores)) == lab:
                    exact_ta += 1
                step_labels = [1 if i == lab else 0 for i in range(ns)]
                aurocs_ta.append(compute_auroc(scores, step_labels))
            if aurocs_ta:
                print(f"  {'Textual alignment':40s} {'text_sim':>10s} {len(aurocs_ta):5d} {exact_ta/len(aurocs_ta)*100:7.1f} {'':>7s} {np.mean(aurocs_ta):7.4f}")
                summary[f"{source}_textual_alignment"] = {'exact': exact_ta/len(aurocs_ta)*100, 'step_auroc': np.mean(aurocs_ta), 'n': len(aurocs_ta)}

        # Sequence-level AUROC
        print(f"\n  Sequence-level AUROC:")
        for z_key, z_label in z_variants:
            scores_seq = []
            labels_seq = []
            for r in subset:
                if z_key not in r:
                    continue
                scores_seq.append(max(r[z_key]['step_ti_max']))
                labels_seq.append(1 if r['has_error'] else 0)
            if scores_seq and sum(labels_seq) > 0 and sum(labels_seq) < len(labels_seq):
                auroc = compute_auroc(scores_seq, labels_seq)
                print(f"  {z_label:40s} AUROC={auroc:.4f} (n={len(scores_seq)})")
                summary[f"{source}_{z_key}_seq_auroc"] = auroc

        ent_seq = [(max(r['final_answer_z']['step_entropy']), 1 if r['has_error'] else 0) for r in subset if 'final_answer_z' in r]
        if ent_seq:
            auroc_ent_seq = compute_auroc([x[0] for x in ent_seq], [x[1] for x in ent_seq])
            print(f"  {'Entropy (no z)':40s} AUROC={auroc_ent_seq:.4f}")
            summary[f"{source}_entropy_seq_auroc"] = auroc_ent_seq

    # Save
    with open(os.path.join(output_dir, "clean_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone. Summary saved.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="~/data/models_dl/Qwen2.5-3B-Instruct")
    parser.add_argument("--data_files", nargs="+", default=[
        "data/processbench/gsm8k.json",
        "data/processbench/math.json",
    ])
    parser.add_argument("--output_dir", type=str, default="results/clean")
    parser.add_argument("--n_samples", type=int, default=0)
    args = parser.parse_args()
    args.model_path = os.path.expanduser(args.model_path)
    run(args)

"""
J2 Pilot: Privileged Distributional Self-Contrast for Error Localization
=========================================================================
For each GSM8K problem:
  1. Generate student response (on-policy rollout)
  2. Compute token-level distributions with and without privileged info z
  3. Calculate InfoGain (KL) and TokenIncompatibility scores
  4. Evaluate whether high-score tokens correspond to error locations
"""

import argparse
import json
import os
import re
import math
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_numeric_answer(text):
    """Extract the final numeric answer from a GSM8K-style response."""
    patterns = [
        r'####\s*([\-\d,\.]+)',
        r'(?:answer|Answer|ANSWER)\s*(?:is|:)\s*([\-\d,\.]+)',
        r'(?:=)\s*([\-\d,\.]+)\s*$',
        r'\\boxed\{([\-\d,\.]+)\}',
    ]
    for pat in patterns:
        m = re.findall(pat, text)
        if m:
            return m[-1].replace(',', '').strip()
    nums = re.findall(r'[\-]?\d+\.?\d*', text)
    return nums[-1] if nums else ""


def generate_student_response(model, tokenizer, question, max_new_tokens=512):
    """Generate on-policy student rollout."""
    prompt = f"Solve this math problem step by step.\n\nQuestion: {question}\n\nSolution:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_ids = outputs[0][prompt_len:]
    response = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return response, gen_ids


def compute_self_contrast(model, tokenizer, question, student_response, z_text):
    """
    Compute token-level self-contrast scores.

    Returns:
        info_gain: KL(p_T || p_S) at each token  (how much z changes the distribution)
        token_incompat: log p_S(y_t) - log p_T(y_t|z)  (does teacher disagree with actual token)
        entropy_s: student entropy at each token  (baseline: uncertainty without z)
        tokens: decoded token strings
    """
    base_prompt = f"Solve this math problem step by step.\n\nQuestion: {question}\n\nSolution:"
    teacher_prompt = f"Solve this math problem step by step.\n\nQuestion: {question}\n\nThe correct answer is: {z_text}\n\nSolution:"

    full_student = base_prompt + student_response
    full_teacher = teacher_prompt + student_response

    student_inputs = tokenizer(full_student, return_tensors="pt").to(model.device)
    teacher_inputs = tokenizer(full_teacher, return_tensors="pt").to(model.device)

    base_ids = tokenizer(base_prompt, return_tensors="pt").input_ids
    teacher_prefix_ids = tokenizer(teacher_prompt, return_tensors="pt").input_ids
    response_ids = tokenizer(student_response, add_special_tokens=False, return_tensors="pt").input_ids

    base_prefix_len = base_ids.shape[1]
    teacher_prefix_len = teacher_prefix_ids.shape[1]
    response_len = response_ids.shape[1]

    with torch.no_grad():
        student_out = model(**student_inputs)
        teacher_out = model(**teacher_inputs)

    student_logits = student_out.logits[0]
    teacher_logits = teacher_out.logits[0]

    n_response_tokens = min(
        student_logits.shape[0] - base_prefix_len,
        teacher_logits.shape[0] - teacher_prefix_len,
        response_len
    )

    if n_response_tokens <= 1:
        return None, None, None, None

    s_logits = student_logits[base_prefix_len - 1: base_prefix_len - 1 + n_response_tokens]
    t_logits = teacher_logits[teacher_prefix_len - 1: teacher_prefix_len - 1 + n_response_tokens]

    s_probs = F.softmax(s_logits, dim=-1)
    t_probs = F.softmax(t_logits, dim=-1)
    s_log_probs = F.log_softmax(s_logits, dim=-1)
    t_log_probs = F.log_softmax(t_logits, dim=-1)

    # InfoGain: KL(p_T || p_S) at each position
    info_gain = (t_probs * (t_log_probs - s_log_probs)).sum(dim=-1)

    actual_token_ids = student_inputs.input_ids[0][base_prefix_len: base_prefix_len + n_response_tokens]

    # TokenIncompatibility: log p_S(y_t) - log p_T(y_t|z)
    s_actual = s_log_probs[range(n_response_tokens), actual_token_ids]
    t_actual = t_log_probs[range(n_response_tokens), actual_token_ids]
    token_incompat = s_actual - t_actual

    # Student entropy (baseline)
    entropy_s = -(s_probs * s_log_probs).sum(dim=-1)

    tokens = [tokenizer.decode([tid]) for tid in actual_token_ids]

    return (
        info_gain.cpu().float().numpy(),
        token_incompat.cpu().float().numpy(),
        entropy_s.cpu().float().numpy(),
        tokens
    )


def find_error_step(student_response, correct_answer):
    """
    Heuristic: split response into steps (by newline/period),
    mark the step where the final answer diverges.
    Returns (step_boundaries, is_correct).
    """
    steps = re.split(r'\n|(?<=\.)\s', student_response)
    steps = [s.strip() for s in steps if s.strip()]

    student_answer = extract_numeric_answer(student_response)
    is_correct = False
    try:
        if student_answer and correct_answer:
            is_correct = abs(float(student_answer) - float(correct_answer)) < 1e-6
    except (ValueError, TypeError):
        is_correct = (student_answer == correct_answer)

    return steps, is_correct


def run_pilot(args):
    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded on {model.device}")

    with open(args.data_path) as f:
        data = [json.loads(line) for line in f]

    if args.n_samples > 0:
        data = data[:args.n_samples]
    print(f"Processing {len(data)} examples")

    results = []
    n_correct = 0
    n_incorrect = 0

    for idx, ex in enumerate(data):
        question = ex['question']
        correct_answer = ex['final_answer']
        full_answer = ex['answer']

        # 1) Generate student response
        student_resp, gen_ids = generate_student_response(
            model, tokenizer, question, max_new_tokens=args.max_tokens
        )

        steps, is_correct = find_error_step(student_resp, correct_answer)
        if is_correct:
            n_correct += 1
        else:
            n_incorrect += 1

        # 2) Compute scores with TRUE z (correct answer)
        true_z = correct_answer
        ig_true, ti_true, ent, tokens = compute_self_contrast(
            model, tokenizer, question, student_resp, true_z
        )

        if ig_true is None:
            print(f"  [{idx}] Skipped (too short)")
            continue

        # 3) Compute scores with SHUFFLED z (wrong answer)
        wrong_answer = str(float(correct_answer) + 42) if correct_answer else "999"
        ig_shuf, ti_shuf, _, _ = compute_self_contrast(
            model, tokenizer, question, student_resp, wrong_answer
        )

        # 4) Compute scores with FULL SOLUTION as z
        ig_full, ti_full, _, _ = compute_self_contrast(
            model, tokenizer, question, student_resp, full_answer
        )

        result = {
            'idx': idx,
            'question': question[:100],
            'correct_answer': correct_answer,
            'student_answer': extract_numeric_answer(student_resp),
            'is_correct': is_correct,
            'student_response': student_resp,
            'n_tokens': len(tokens),
            'tokens': tokens,
            'info_gain_true_z': ig_true.tolist(),
            'token_incompat_true_z': ti_true.tolist(),
            'info_gain_shuffled_z': ig_shuf.tolist() if ig_shuf is not None else None,
            'token_incompat_shuffled_z': ti_shuf.tolist() if ti_shuf is not None else None,
            'info_gain_full_z': ig_full.tolist() if ig_full is not None else None,
            'token_incompat_full_z': ti_full.tolist() if ti_full is not None else None,
            'entropy_student': ent.tolist(),
        }
        results.append(result)

        if (idx + 1) % 10 == 0 or idx == 0:
            student_ans = result['student_answer']
            mean_ig = float(ig_true.mean())
            mean_ti = float(ti_true.mean())
            print(f"  [{idx+1}/{len(data)}] correct={is_correct} "
                  f"ans={student_ans}/{correct_answer} "
                  f"mean_IG={mean_ig:.4f} mean_TI={mean_ti:.4f} "
                  f"n_tok={len(tokens)}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "pilot_results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {out_path}")
    print(f"Correct: {n_correct}, Incorrect: {n_incorrect}")

    # Quick analysis
    analyze_results(results, args.output_dir)


def analyze_results(results, output_dir):
    """Compute aggregate statistics for the pilot."""
    correct_results = [r for r in results if r['is_correct']]
    incorrect_results = [r for r in results if not r['is_correct']]

    print(f"\n{'='*60}")
    print(f"PILOT ANALYSIS")
    print(f"{'='*60}")
    print(f"Total: {len(results)}, Correct: {len(correct_results)}, Incorrect: {len(incorrect_results)}")

    # Compare mean scores: correct vs incorrect responses
    for label, subset in [("CORRECT", correct_results), ("INCORRECT", incorrect_results)]:
        if not subset:
            continue
        mean_ig = sum(sum(r['info_gain_true_z']) / len(r['info_gain_true_z']) for r in subset) / len(subset)
        mean_ti = sum(sum(r['token_incompat_true_z']) / len(r['token_incompat_true_z']) for r in subset) / len(subset)
        mean_ent = sum(sum(r['entropy_student']) / len(r['entropy_student']) for r in subset) / len(subset)
        print(f"\n{label} responses (n={len(subset)}):")
        print(f"  Mean InfoGain (true z):        {mean_ig:.4f}")
        print(f"  Mean TokenIncompat (true z):   {mean_ti:.4f}")
        print(f"  Mean Student Entropy:          {mean_ent:.4f}")

    # Compare true z vs shuffled z
    diffs_ig = []
    diffs_ti = []
    for r in incorrect_results:
        if r['info_gain_shuffled_z'] is None:
            continue
        ig_true_mean = sum(r['info_gain_true_z']) / len(r['info_gain_true_z'])
        ig_shuf_mean = sum(r['info_gain_shuffled_z']) / len(r['info_gain_shuffled_z'])
        ti_true_mean = sum(r['token_incompat_true_z']) / len(r['token_incompat_true_z'])
        ti_shuf_mean = sum(r['token_incompat_shuffled_z']) / len(r['token_incompat_shuffled_z'])
        diffs_ig.append(ig_true_mean - ig_shuf_mean)
        diffs_ti.append(ti_true_mean - ti_shuf_mean)

    if diffs_ig:
        print(f"\nTRUE z vs SHUFFLED z (incorrect responses):")
        print(f"  InfoGain diff (true - shuffled): {sum(diffs_ig)/len(diffs_ig):.4f}")
        print(f"  TokenIncompat diff:              {sum(diffs_ti)/len(diffs_ti):.4f}")

    # Top-k token analysis: do high-scoring tokens cluster in later (error) positions?
    for r in incorrect_results[:3]:
        tokens = r['tokens']
        ti_scores = r['token_incompat_true_z']
        n = len(ti_scores)
        if n < 10:
            continue
        sorted_indices = sorted(range(n), key=lambda i: ti_scores[i], reverse=True)
        top_k = sorted_indices[:max(5, n // 10)]
        avg_pos = sum(top_k) / len(top_k) / n
        print(f"\n  Example (idx={r['idx']}, ans={r['student_answer']}/{r['correct_answer']}):")
        print(f"    Top-k TI tokens cluster at avg relative pos: {avg_pos:.2f} (1.0=end)")
        top_tokens = [(i, tokens[i], f"{ti_scores[i]:.3f}") for i in top_k[:8]]
        print(f"    Top TI tokens: {top_tokens}")

    # Save analysis
    analysis_path = os.path.join(output_dir, "pilot_analysis.txt")
    with open(analysis_path, 'w') as f:
        f.write(f"Total: {len(results)}, Correct: {len(correct_results)}, Incorrect: {len(incorrect_results)}\n")
        for label, subset in [("CORRECT", correct_results), ("INCORRECT", incorrect_results)]:
            if not subset:
                continue
            mean_ig = sum(sum(r['info_gain_true_z'])/len(r['info_gain_true_z']) for r in subset) / len(subset)
            mean_ti = sum(sum(r['token_incompat_true_z'])/len(r['token_incompat_true_z']) for r in subset) / len(subset)
            f.write(f"{label}: mean_IG={mean_ig:.4f}, mean_TI={mean_ti:.4f}\n")
    print(f"\nAnalysis saved to {analysis_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="~/data/models_dl/Qwen2.5-3B-Instruct")
    parser.add_argument("--data_path", type=str, default="gsm8k_test.jsonl")
    parser.add_argument("--output_dir", type=str, default="results/pilot")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--max_tokens", type=int, default=512)
    args = parser.parse_args()
    args.model_path = os.path.expanduser(args.model_path)
    run_pilot(args)

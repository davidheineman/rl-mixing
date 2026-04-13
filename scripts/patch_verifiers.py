"""Patch ground_truth_utils.py verifiers for thinking-model compatibility.

StringMatcherVerifier: when the model wraps a full sentence like
"Therefore, the answer is (C)" inside <answer> tags, the raw normalized
comparison fails. This patch adds an MCQ-regex fallback on the <answer>
tag content.

CodeVerifier: when the model produces code blocks inside <think> tags,
the "last fenced block" heuristic picks up reasoning-phase code instead
of the final answer. This patch prefers code blocks outside <think>.
"""
import re

PATH = "open_instruct/ground_truth_utils.py"
t = open(PATH).read()

# --- StringMatcherVerifier: add MCQ regex fallback on <answer> content ---
OLD_SM = (
    '        if "<answer>" in prediction and "</answer>" in prediction:\n'
    '            answer_string = prediction.split("<answer>")[-1].split("</answer>")[0]\n'
    '            score = float(normalize_answer(answer_string) == normalize_answer(label))\n'
    '            return VerificationResult(score=score)'
)
NEW_SM = (
    '        if "<answer>" in prediction and "</answer>" in prediction:\n'
    '            answer_string = prediction.split("<answer>")[-1].split("</answer>")[0]\n'
    '            score = float(normalize_answer(answer_string) == normalize_answer(label))\n'
    '            if score > 0:\n'
    '                return VerificationResult(score=score)\n'
    '            # Fallback: try MCQ regex on the <answer> tag content\n'
    '            if re.fullmatch(r"[A-D]", label.strip()):\n'
    '                extracted = self._extract_mcq_answer(answer_string)\n'
    '                if extracted is not None:\n'
    '                    score = float(normalize_answer(extracted) == normalize_answer(label))\n'
    '                    if score > 0:\n'
    '                        return VerificationResult(score=score)'
)
if OLD_SM in t:
    t = t.replace(OLD_SM, NEW_SM)
    print("[patch] StringMatcherVerifier: added MCQ regex fallback on <answer> content")
else:
    print("[patch] StringMatcherVerifier: target not found, skipping")

# --- CodeVerifier: prefer code blocks outside <think> ---
OLD_CODE = "        return matches[-1].strip()"
NEW_CODE = (
    '        # Prefer code blocks outside <think>...</think>\n'
    '        text_no_think = re.sub(r"<think>.*?</think>", "", model_output, flags=re.DOTALL)\n'
    '        outside_matches = re.findall(r"```(?:python)?(.*?)```", text_no_think, re.DOTALL)\n'
    '        if outside_matches:\n'
    '            return outside_matches[-1].strip()\n'
    '        return matches[-1].strip()'
)
if OLD_CODE in t:
    t = t.replace(OLD_CODE, NEW_CODE, 1)
    print("[patch] CodeVerifier: prefer code blocks outside <think>")
else:
    print("[patch] CodeVerifier: target not found, skipping")

open(PATH, "w").write(t)
print("[patch] Done")

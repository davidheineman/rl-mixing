"""Patch ground_truth_utils.py verifiers for thinking-model compatibility.

Uses line-level insertion rather than exact string matching, so it works
regardless of minor formatting differences between image versions.
"""
import re

PATH = "open_instruct/ground_truth_utils.py"
lines = open(PATH).readlines()
new_lines = []

# --- StringMatcherVerifier: add MCQ regex fallback on <answer> content ---
# Look for the pattern: if "<answer>" in prediction ... return VerificationResult(score=score)
# and replace the unconditional return with a conditional one + MCQ regex fallback
sm_patched = False
i = 0
while i < len(lines):
    line = lines[i]

    # Find: return VerificationResult(score=score) inside the <answer> extraction block
    if (not sm_patched
        and 'return VerificationResult(score=score)' in line
        and i >= 2
        and 'normalize_answer(answer_string)' in lines[i-1]
        and '<answer>' in lines[i-2]):

        indent = line[:len(line) - len(line.lstrip())]
        new_lines.append(f'{indent}if score > 0:\n')
        new_lines.append(f'{indent}    return VerificationResult(score=score)\n')
        new_lines.append(f'{indent}# Fallback: MCQ regex on <answer> tag content\n')
        new_lines.append(f'{indent}if re.fullmatch(r"[A-D]", label.strip()):\n')
        new_lines.append(f'{indent}    extracted = self._extract_mcq_answer(answer_string)\n')
        new_lines.append(f'{indent}    if extracted is not None:\n')
        new_lines.append(f'{indent}        score = float(normalize_answer(extracted) == normalize_answer(label))\n')
        new_lines.append(f'{indent}        if score > 0:\n')
        new_lines.append(f'{indent}            return VerificationResult(score=score)\n')
        sm_patched = True
        i += 1
        continue

    new_lines.append(line)
    i += 1

if sm_patched:
    print("[patch] StringMatcherVerifier: added MCQ regex fallback on <answer> content")
else:
    print("[patch] StringMatcherVerifier: target not found, skipping")

# --- CodeVerifier: prefer code blocks outside <think> ---
# Look for: return matches[-1].strip() inside extract_python_code
code_patched = False
final_lines = []
for i, line in enumerate(new_lines):
    if (not code_patched
        and 'return matches[-1].strip()' in line
        and any('extract_python_code' in new_lines[j] for j in range(max(0, i-15), i))):

        indent = line[:len(line) - len(line.lstrip())]
        final_lines.append(f'{indent}# Prefer code blocks outside <think>...</think>\n')
        final_lines.append(f'{indent}text_no_think = re.sub(r"<think>.*?</think>", "", model_output, flags=re.DOTALL)\n')
        final_lines.append(f'{indent}outside_matches = re.findall(r"```(?:python)?(.*?)```", text_no_think, re.DOTALL)\n')
        final_lines.append(f'{indent}if outside_matches:\n')
        final_lines.append(f'{indent}    return outside_matches[-1].strip()\n')
        final_lines.append(line)
        code_patched = True
        continue

    final_lines.append(line)

if code_patched:
    print("[patch] CodeVerifier: prefer code blocks outside <think>")
else:
    print("[patch] CodeVerifier: target not found, skipping")

# --- Reward lookup: add prefix-based fallback for dataset name matching ---
# The dataset field can be "math_aime_2025" but the verifier registers as "math".
# Add a fallback that tries progressively shorter prefixes.
reward_patched = False
result_lines = []
for i, line in enumerate(final_lines):
    if (not reward_patched
        and 'No reward function found for dataset' in line
        and i >= 1
        and 'reward_fn_mapping.get' in final_lines[i-2]):

        # Insert prefix fallback before the warning
        indent = final_lines[i-2][:len(final_lines[i-2]) - len(final_lines[i-2].lstrip())]
        # Replace the if block: instead of just .get(ds.lower()), try prefix matching
        # Go back to the .get line and add fallback logic after it
        result_lines.append(f'{indent}if reward_func is None:\n')
        result_lines.append(f'{indent}    for key in reward_fn_mapping:\n')
        result_lines.append(f'{indent}        if ds.lower().startswith(key):\n')
        result_lines.append(f'{indent}            reward_func = reward_fn_mapping[key]\n')
        result_lines.append(f'{indent}            break\n')
        result_lines.append(line)  # keep the original warning line
        reward_patched = True
        continue

    result_lines.append(line)

if reward_patched:
    print("[patch] Reward lookup: added prefix-based dataset name fallback")
else:
    print("[patch] Reward lookup: target not found, skipping")

open(PATH, "w").writelines(result_lines)
print("[patch] Done")

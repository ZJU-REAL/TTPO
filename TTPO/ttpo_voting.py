import re

from math_verify import parse, verify


def extract_boxed_answer(text):
    """Extract the answer inside the first \\boxed{...} after </think>.

    For thinking models (e.g. Qwen3) only search after </think> to avoid picking up
    intermediate answers from the thinking block. Handles nested braces. Returns the
    stripped content, or None if no well-formed \\boxed{} is found.
    """
    if text is None:
        return None
    think_end = text.rfind("</think>")
    search_text = text[think_end + len("</think>"):] if think_end != -1 else text

    idx = search_text.find(r"\boxed{")
    if idx == -1:
        return None
    start = idx + len(r"\boxed{")
    depth = 1
    i = start
    while i < len(search_text) and depth > 0:
        if search_text[i] == "{":
            depth += 1
        elif search_text[i] == "}":
            depth -= 1
        i += 1
    if depth == 0:
        return search_text[start: i - 1].strip()
    return None


def _preprocess_for_parse(answer):
    """Convert ratio notation a:b -> \\frac{a}{b} so math_verify can parse it."""
    if answer is None:
        return None
    ratio_match = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*:\s*(-?\d+(?:\.\d+)?)\s*", answer)
    if ratio_match:
        return rf"\frac{{{ratio_match.group(1)}}}{{{ratio_match.group(2)}}}"
    return answer


def answers_equivalent(a, b):
    """Whether two answer strings are equivalent.

    Tries math_verify mathematical equivalence first (fractions, algebra, ...), then
    falls back to whitespace-stripped case-insensitive string match (handles MCQ like
    "E" which parse() returns None for). Mirrors grpo_train.reward_correctness.
    """
    if a is None or b is None:
        return False
    try:
        a_parsed = parse(_preprocess_for_parse(a))
        b_parsed = parse(_preprocess_for_parse(b))
        if a_parsed is not None and b_parsed is not None:
            if verify(a_parsed, b_parsed):
                return True
    except Exception:
        pass
    a_norm = re.sub(r"\s+", "", a).lower()
    b_norm = re.sub(r"\s+", "", b).lower()
    return bool(a_norm) and a_norm == b_norm


def majority_vote(answers):
    """Majority-vote pseudo-labelling over K answers.

    Args:
        answers: list of K answer strings (or None for trajectories with no \\boxed{}).

    Returns:
        pseudo_label:    representative answer string of the (selected) largest cluster,
                         or None if there are zero valid answers.
        correct_mask:    list[bool] of length K — True for trajectories whose answer is
                         in the selected cluster (the positive samples).
        consensus_count: size of the selected cluster (0 if no valid answers).

    Tie-breaking: when multiple clusters share the same maximum size, the cluster whose
    representative answer string is shortest (after stripping) is chosen.
    """
    K = len(answers)
    valid_idx = [i for i, a in enumerate(answers) if a is not None and a != ""]

    clusters = []  # each: {"rep": str, "members": [idx, ...]}
    for i in valid_idx:
        placed = False
        for c in clusters:
            if answers_equivalent(answers[i], c["rep"]):
                c["members"].append(i)
                placed = True
                break
        if not placed:
            clusters.append({"rep": answers[i], "members": [i]})

    correct_mask = [False] * K
    if not clusters:
        return None, correct_mask, 0

    best = max(clusters, key=lambda c: (len(c["members"]), -len(c["rep"].strip())))
    consensus_count = len(best["members"])
    for i in best["members"]:
        correct_mask[i] = True
    return best["rep"], correct_mask, consensus_count

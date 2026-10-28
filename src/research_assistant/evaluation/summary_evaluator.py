import re


def tokenize(text: str) -> set[str]:
    """Normalize text into a set of tokens."""
    return set(re.findall(r"\b[a-zA-Z0-9-]+\b", text.lower()))


def evaluate_summary(
    generated_summary: str,
    reference_summary: str,
) -> dict:
    """Evaluate lexical overlap with a reference summary."""
    generated_tokens = tokenize(generated_summary)
    reference_tokens = tokenize(reference_summary)

    overlap = generated_tokens & reference_tokens

    precision = (
        len(overlap) / len(generated_tokens)
        if generated_tokens
        else 0.0
    )
    recall = (
        len(overlap) / len(reference_tokens)
        if reference_tokens
        else 0.0
    )
    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "generated_token_count": len(generated_tokens),
        "reference_token_count": len(reference_tokens),
        "overlapping_tokens": len(overlap),
        "token_precision": round(precision, 4),
        "token_recall": round(recall, 4),
        "token_f1": round(f1_score, 4),
    }

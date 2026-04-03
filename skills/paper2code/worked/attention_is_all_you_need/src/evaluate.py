"""
Attention Is All You Need — Evaluation Metrics

Paper: https://arxiv.org/abs/1706.03762
Implements: BLEU score computation for translation evaluation.

Section references:
  §6 — "Results" — BLEU scores reported in Table 2

NOTE: For reproducible BLEU, use sacrebleu (pip install sacrebleu).
      Different BLEU implementations give different numbers (sometimes 1-2 points)
      due to tokenization and smoothing differences.
"""

from typing import Dict, List


def compute_bleu(
    hypotheses: List[str],
    references: List[List[str]],
) -> float:
    """Compute BLEU score using sacrebleu for reproducibility.

    §6, Table 2 — "BLEU" score on WMT 2014 EN-DE and EN-FR.

    The paper reports BLEU scores of:
      - 27.3 BLEU on EN-DE (base model)
      - 28.4 BLEU on EN-DE (big model)
      - 38.1 BLEU on EN-FR (big model)

    These scores use beam search with beam size 4 and length penalty α=0.6 (§6.1).

    Args:
        hypotheses: list of predicted translations (detokenized strings)
        references: list of lists of reference translations

    Returns:
        BLEU score as a float

    NOTE: [UNSPECIFIED] The paper does not specify which BLEU implementation
    was used. We use sacrebleu (Post, 2018) for reproducibility, which is the
    current standard. Original 2017 results may have used multi-bleu.perl.
    """
    try:
        import sacrebleu
        # sacrebleu expects: hypotheses as list[str], references as list[list[str]]
        bleu = sacrebleu.corpus_bleu(hypotheses, references)
        return bleu.score
    except ImportError:
        raise ImportError(
            "sacrebleu is required for BLEU computation. "
            "Install with: pip install sacrebleu"
        )


def compute_all_metrics(
    hypotheses: List[str],
    references: List[List[str]],
) -> Dict[str, float]:
    """Compute all evaluation metrics reported in the paper.

    §6, Table 2 — "We report BLEU scores"

    Args:
        hypotheses: list of predicted translations
        references: list of lists of reference translations

    Returns:
        Dict mapping metric name to value
    """
    return {
        "BLEU": compute_bleu(hypotheses, references),
    }

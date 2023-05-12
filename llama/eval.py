import json
import evaluate
import math
import re
import numpy as np
from pathlib import Path

"""
one eval pipeline for bleu-4, recall, recall_unseen.
input: json file with one dictionary
    the keys are metadata on the experiment, as a tuple (n icl exemplars, true/false retrieval, true/false rationale)
    the values are a list of 3-tuples, each 3-tuple is a pairing (question id, LLM output code snippet as a string, oracle/ground truth code snippet as a string), 
        and the list spans the whole eval set for that experiment
output: 3-tuple of float values, (bleu_4, recall, recall_unseen)
"""

DEVICE_ID = "cuda" if torch.cuda.is_available() else "cpu"
USE_DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

ROOT_DIR = Path(__file__).resolve().parents[1]

BLEU_EVALUATOR = evaluate.load("neulab/python_bleu")


def _get_ngrams(segment, max_order):
    """Extracts all n-grams upto a given maximum order from an input segment.

  Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this
        methods.

  Returns:
    The Counter containing all n-grams upto max_order in segment
    with a count of how many times each n-gram occurred.
  """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 smooth=False):
    """Computes BLEU score of translated segments against one or more references.

  Args:
    reference_corpus: list of lists of references for each translation. Each
        reference should be tokenized into a list of tokens.
    translation_corpus: list of translations to score. Each translation
        should be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.
    smooth: Whether or not to apply Lin et al. 2004 smoothing.

  Returns:
    3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
    precisions and brevity penalty.
  """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus,
                                         translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                             (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                                 possible_matches_by_order[i])
                # print(i, f"{precisions[i]:.03f}={float(matches_by_order[i]):.03f}/{possible_matches_by_order[i]}")
            else:
                precisions[i] = 0.0
    # print("========")
    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.
    else:
        bp = math.exp(1 - 1. / ratio)

    bleu = geo_mean * bp

    # print(bleu, precisions, bp, ratio, translation_length, reference_length)
    return (bleu, precisions, bp, ratio, translation_length, reference_length)


""" The tokenizer that we use for code submissions, from Wang Ling et al., Latent Predictor Networks for Code Generation (2016)
    @param code: string containing a code snippet
    @return: list of code tokens
"""


def tokenize_for_bleu_eval(code):
    code = re.sub(r'([^A-Za-z0-9_])', r' \1 ', code)
    code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
    code = re.sub(r'\s+', ' ', code)
    code = code.replace('"', '`')
    code = code.replace('\'', '`')
    tokens = [t for t in code.split(' ') if t]
    return tokens


def _bleu(ref_file, trans_file, subword_option=None, smooth=True, code_tokenize=False):
    assert code_tokenize
    assert not smooth
    max_order = 4
    ref_files = [ref_file]
    reference_text = []
    for reference_filename in ref_files:
        with open(reference_filename) as fh:
            reference_text.append(fh.readlines())
    per_segment_references = []
    for references in zip(*reference_text):
        reference_list = []
        for reference in references:
            if code_tokenize:
                reference_list.append(tokenize_for_bleu_eval(reference.strip()))
            else:
                reference_list.append(reference.strip().split())
        per_segment_references.append(reference_list)
    translations = []
    with open(trans_file) as fh:
        for line in fh:
            if code_tokenize:
                translations.append(tokenize_for_bleu_eval(line.strip()))
            else:
                translations.append(line.strip().split())
    print(f'src length: {len(per_segment_references)}, tgt length: {len(translations)}')
    bleu_score, _, _, _, _, _ = compute_bleu(per_segment_references, translations, max_order, smooth)
    return round(100 * bleu_score, 2)


def get_metrics(results_file):
    ### LOAD IN JSON
    with open(results_file, 'r') as fin:
        # Read the contents of the file
        json_string = fin.read()

    # Parse the JSON string into a dictionary
    experiments = json.load(json_string)

    results_dict = {}

    for experiment, results in experiments.items():
        predictions = []
        references = []
        question_ids = []
        ### loop thru dataset, get predictions and references
        for line in results:
            question_ids.append(line[0])
            predictions.append(line[1])
            references.append(line[2])

        
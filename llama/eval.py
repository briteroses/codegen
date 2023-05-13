import json
import collections
import math
import re
import torch
import numpy as np

from pathlib import Path
from pprint import pprint

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


def compute_bleu(reference_corpus, translation_corpus, max_order=4):
    """Computes BLEU score of translated segments against one or more references.

  Args:
    reference_corpus: list of lists of references for each translation. Each
        reference should be tokenized into a list of tokens.
    translation_corpus: list of translations to score. Each translation
        should be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.

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

    return bleu


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

'''
based on docprompting conala BLEU-4, but text files are switched out for lists
'''
def _bleu(references, predictions, max_order=4):
    # compute_bleu expects references a list of lists
    # and also for all strings to be tokenized
    references_for_bleu = [[tokenize_for_bleu_eval(refer)] for refer in references]
    predictions_for_bleu = [tokenize_for_bleu_eval(snippet.strip()) for snippet in predictions]
    bleu_score = compute_bleu(references_for_bleu, predictions_for_bleu, max_order)
    return round(100 * bleu_score, 2)


def exact_match(references, predictions):
    em = sum(int(references[i] == predictions[i]) for i in range(len(references)))/len(references)
    return round(100 * em, 2)


def get_metrics(results_file):

    with open(results_file, 'r') as fin:
        experiments = json.load(fin)
    
    results_dict = {}
    for experiment, results in experiments.items():
        question_ids = [sample[0] for sample in results]
        predictions = [sample[1] for sample in results]
        references = [sample[2] for sample in results]

        bleu_4 = _bleu(references, predictions)
        em = exact_match(references, predictions)

        results_dict[experiment] = (bleu_4, em)
    
    return results_dict


if __name__ == "__main__":
    json_location = str(ROOT_DIR / f'llama/completions/mosaic-chat')
    all_ablations = json_location + "/res.json"
    pprint(get_metrics(all_ablations))
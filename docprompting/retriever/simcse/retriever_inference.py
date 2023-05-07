import os
import time
import pandas as pd
import warnings
import faiss
import numpy as np
import torch
from tqdm import tqdm
import transformers
from transformers import AutoModel, AutoTokenizer, AutoConfig
import sys
import json
import copy
from collections import OrderedDict
from torch import nn
from transformers import PreTrainedModel
from collections import defaultdict
from transformers.modeling_outputs import SequenceClassifierOutput
torch.set_printoptions(profile="full")
import numpy as np
from utils import (
    get_model,
    get_tokenizer,
    get_idf_dict,
    bert_cos_score_idf,
    model2layers,
    get_hash,
    greedy_cos_idf_for_train,
    greedy_cos_idf
)

sys.path.append('..')
TQDM_DISABLED = os.environ['TQDM_DISABLED'] if 'TQDM_DISABLED' in os.environ else False
TOP_K = [1, 3, 5, 8, 10, 12, 15, 20, 30, 50, 100, 200]

class RetrievalModel(PreTrainedModel):
    """
    Adapt the implementation of BertScore to calculate the similarity between a query and a doc
    with either CLS mean pooling distance or BERTScore F1.
    """

    def __init__(
        self,
            config: object,
            model_type: object = None,
            num_layers: object = None,
            batch_size: object = 64,
            nthreads: object = 4,
            all_layers: object = False,
            idf: object = False,
            idf_sents: object = None,
            device: object = None,
            lang: object = None,
            rescale_with_baseline: object = False,
            baseline_path: object = None,
            use_fast_tokenizer: object = False,
            tokenizer: object = None,
            training_args: object = None,
            model_args: object = None
    ) -> object:
        super().__init__(config)
        """
        Args:
            - :param: `model_type` (str): contexual embedding model specification, default using the suggested
                      model for the target langauge; has to specify at least one of
                      `model_type` or `lang`
            - :param: `num_layers` (int): the layer of representation to use.
                      default using the number of layer tuned on WMT16 correlation data
            - :param: `verbose` (bool): turn on intermediate status update
            - :param: `idf` (bool): a booling to specify whether to use idf or not (this should be True even if `idf_sents` is given)
            - :param: `idf_sents` (List of str): list of sentences used to compute the idf weights
            - :param: `device` (str): on which the contextual embedding model will be allocated on.
                      If this argument is None, the model lives on cuda:0 if cuda is available.
            - :param: `batch_size` (int): bert score processing batch size
            - :param: `nthreads` (int): number of threads
            - :param: `lang` (str): language of the sentences; has to specify
                      at least one of `model_type` or `lang`. `lang` needs to be
                      specified when `rescale_with_baseline` is True.
            - :param: `return_hash` (bool): return hash code of the setting
            - :param: `rescale_with_baseline` (bool): rescale bertscore with pre-computed baseline
            - :param: `baseline_path` (str): customized baseline file
            - :param: `use_fast_tokenizer` (bool): `use_fast` parameter passed to HF tokenizer
        """

        assert lang is not None or model_type is not None, "Either lang or model_type should be specified"
        if rescale_with_baseline:
            assert lang is not None, "Need to specify Language when rescaling with baseline"
        self.training_args = training_args
        self.model_args = model_args
        self._lang = lang
        self._rescale_with_baseline = rescale_with_baseline
        self._idf = idf
        self.batch_size = batch_size
        self.nthreads = nthreads
        self.all_layers = all_layers

        assert model_type is not None
        self._model_type = model_type

        if num_layers is None:
            self._num_layers = model2layers[self.model_type]
        else:
            self._num_layers = num_layers


        self._use_fast_tokenizer = use_fast_tokenizer
        self._tokenizer = get_tokenizer(self.model_type, self._use_fast_tokenizer) if tokenizer is None else tokenizer
        self._model = get_model(self.model_type, self.num_layers, self.all_layers)

        self._idf_dict = None
        if idf_sents is not None:
            self.compute_idf(idf_sents)

        self._baseline_vals = None
        self.baseline_path = baseline_path
        self.use_custom_baseline = self.baseline_path is not None
        if self.baseline_path is None:
            self.baseline_path = os.path.join(
                os.path.dirname(__file__), f"rescale_baseline/{self.lang}/{self.model_type}.tsv"
            )

    @property
    def lang(self):
        return self._lang

    @property
    def idf(self):
        return self._idf

    @property
    def model_type(self):
        return self._model_type

    @property
    def num_layers(self):
        return self._num_layers

    @property
    def rescale_with_baseline(self):
        return self._rescale_with_baseline

    @property
    def baseline_vals(self):
        if self._baseline_vals is None:
            if os.path.isfile(self.baseline_path):
                if not self.all_layers:
                    self._baseline_vals = torch.from_numpy(
                        pd.read_csv(self.baseline_path).iloc[self.num_layers].to_numpy()
                    )[1:].float()
                else:
                    self._baseline_vals = (
                        torch.from_numpy(pd.read_csv(self.baseline_path).to_numpy())[:, 1:].unsqueeze(1).float()
                    )
            else:
                raise ValueError(f"Baseline not Found for {self.model_type} on {self.lang} at {self.baseline_path}")

        return self._baseline_vals

    @property
    def use_fast_tokenizer(self):
        return self._use_fast_tokenizer

    @property
    def hash(self):
        return get_hash(
            self.model_type, self.num_layers, self.idf, self.rescale_with_baseline, self.use_custom_baseline, self.use_fast_tokenizer
        )

    def compute_idf(self, sents):
        """
        Args:

        """
        if self._idf_dict is not None:
            warnings.warn("Overwriting the previous importance weights.")

        self._idf_dict = get_idf_dict(sents, self._tokenizer, nthreads=self.nthreads)

    def get_pooling_embedding(self, input_ids, attention_mask, lengths, pooling="mean", normalize=False):
        out = self._model(input_ids, attention_mask=attention_mask, output_hidden_states=self.all_layers)
        if self.all_layers:
            emb = torch.stack(out[-1], dim=2)
        else:
            emb = out[0]
        if pooling == "mean":
            emb.masked_fill_(~attention_mask.bool().unsqueeze(-1), 2)
            max_len = max(lengths)
            base = torch.arange(max_len, dtype=torch.long).expand(len(lengths), max_len).to(lengths.device)
            pad_mask = base < lengths.unsqueeze(1)
            emb = (emb * pad_mask.unsqueeze(-1)).sum(dim=1) / pad_mask.sum(-1).unsqueeze(-1)
            if normalize:
                emb = emb / emb.norm(dim=1, keepdim=True)
        else:
            raise NotImplementedError
        return emb


    def calc_pair_tok_embedding(self, input_ids, attention_mask, lengths=None, input_idf=None, num_sent=None):

        batch_size = int(input_ids.shape[0] / num_sent)
        # calc embeddings
        out = self._model(input_ids, attention_mask=attention_mask, output_hidden_states=self.all_layers)
        if self.all_layers:
            emb = torch.stack(out[-1], dim=2)
        else:
            emb = out[0]

        def split_to_pair(m):
            dim_size = len(m.shape)
            if dim_size == 1:
                m = m.view(batch_size, num_sent)
            elif dim_size == 2:
                m = m.view(batch_size, num_sent, -1)
            elif dim_size == 3:
                max_len = m.size(-2)
                m = m.view(batch_size, num_sent, max_len, -1)
            else:
                raise ValueError('dimension should be only 2 or 3')
            return m[:, 0], m[:, 1]

        def length_to_mask(lens, max_len):
            base = torch.arange(max_len, dtype=torch.long).expand(len(lens), max_len).to(lens.device)
            return base < lens.unsqueeze(1)

        ref_emb, hyp_emb = split_to_pair(emb)
        ref_att_mask, hyp_attn_mask = split_to_pair(attention_mask)
        ref_len, hyp_len = split_to_pair(lengths)
        ref_idf, hyp_idf = split_to_pair(input_idf)

        # pad to calculate greedy cos idf
        # emb_pad: padding with value 2
        ref_emb.masked_fill_(~ref_att_mask.bool().unsqueeze(-1), 2)
        hyp_emb.masked_fill_(~hyp_attn_mask.bool().unsqueeze(-1), 2)
        # idf_pad: padding with value 0 (already satisfied)
        # pad_mask: length mask
        max_len = max(max(ref_len), max(hyp_len))
        ref_pad_mask = length_to_mask(ref_len, max_len)
        hyp_pad_mask = length_to_mask(hyp_len, max_len)

        return ref_emb, ref_pad_mask, ref_idf, hyp_emb, hyp_pad_mask, hyp_idf

    def forward(self, input_ids=None, attention_mask=None, negative_sample_mask=None,
                lengths=None, input_idf=None,
                num_sent=None, pairwise_similarity=False):
        ref_emb, ref_pad_mask, ref_idf, \
        hyp_emb, hyp_pad_mask, hyp_idf = self.calc_pair_tok_embedding(input_ids, attention_mask,
                                                             lengths, input_idf, num_sent)

        if 'cls_distance' in self.model_args.sim_func:
            # ref_emb [B, max_len, embed_size]
            # ref_pad_mask [B, max_len]
            # m_ref_emb [B, embed_size]
            m_ref_emb = (ref_emb * ref_pad_mask.unsqueeze(-1)).sum(dim=1) / ref_pad_mask.sum(-1).unsqueeze(-1)
            m_hyp_emb = (hyp_emb * hyp_pad_mask.unsqueeze(-1)).sum(dim=1) / hyp_pad_mask.sum(-1).unsqueeze(-1)

            if 'cosine' in self.model_args.sim_func:
                cos_sim = nn.CosineSimilarity(dim=-1)
                sim_score = cos_sim(m_ref_emb.unsqueeze(1), m_hyp_emb.unsqueeze(0))
            elif 'l2' in self.model_args.sim_func:
                sim_score = torch.matmul(m_ref_emb, m_hyp_emb.transpose(0, 1))
            else:
                raise NotImplementedError

            if pairwise_similarity: # pairwise score only
                return torch.diagonal(sim_score, 0)
            else:
                loss_fct = nn.CrossEntropyLoss(reduction='mean')
                labels = torch.arange(sim_score.size(0)).long().to(sim_score.device)
                # mask the conflict negative examples
                sim_score.masked_fill_(~negative_sample_mask, -1e10)
                sim_score = sim_score / self.model_args.temp
                loss = loss_fct(sim_score, labels)
                return SequenceClassifierOutput(loss=loss)

        elif self.model_args.sim_func == 'bertscore':
            if pairwise_similarity:
                _, _, sim_score = greedy_cos_idf(ref_emb, ref_pad_mask, ref_idf,
                                      hyp_emb, hyp_pad_mask, hyp_idf,
                                      self.all_layers)
                return sim_score
            else:
                _, _, sim_score = greedy_cos_idf_for_train(ref_emb, ref_pad_mask, ref_idf,
                                          hyp_emb, hyp_pad_mask, hyp_idf,
                                          self.all_layers)
                labels = torch.arange(sim_score.size(0)).long().to(sim_score.device)
                # print(sim_score)
                if self.model_args.bert_score_loss == 'hinge':
                    loss_fct = nn.MultiMarginLoss(margin=self.model_args.hinge_margin, reduction='none')
                    sim_score.masked_fill_(~negative_sample_mask, -1e10)
                    loss = loss_fct(sim_score, labels)
                    loss = loss * sim_score.shape[1] / negative_sample_mask.long().sum(-1) # recover x.size(0)
                    loss = torch.mean(loss)
                else:
                    loss_fct = nn.CrossEntropyLoss(reduction='mean')
                    sim_score.masked_fill_(~negative_sample_mask, -1e10)
                    sim_score = sim_score / self.model_args.temp
                    loss = loss_fct(sim_score, labels)
                # loss_fct = nn.CrossEntropyLoss()
                return SequenceClassifierOutput(loss = loss)

        else:
            raise NotImplementedError



    def score(self, cands, refs, verbose=False, batch_size=64, return_hash=False):
        """
        Args:
            - :param: `cands` (list of str): candidate sentences
            - :param: `refs` (list of str or list of list of str): reference sentences

        Return:
            - :param: `(P, R, F)`: each is of shape (N); N = number of input
                      candidate reference pairs. if returning hashcode, the
                      output will be ((P, R, F), hashcode). If a candidate have
                      multiple references, the returned score of this candidate is
                      the *best* score among all references.
        """

        ref_group_boundaries = None
        if not isinstance(refs[0], str):
            ref_group_boundaries = []
            ori_cands, ori_refs = cands, refs
            cands, refs = [], []
            count = 0
            for cand, ref_group in zip(ori_cands, ori_refs):
                cands += [cand] * len(ref_group)
                refs += ref_group
                ref_group_boundaries.append((count, count + len(ref_group)))
                count += len(ref_group)

        if verbose:
            print("calculating scores...")
            start = time.perf_counter()

        if self.idf:
            assert self._idf_dict, "IDF weights are not computed"
            idf_dict = self._idf_dict
        else:
            idf_dict = defaultdict(lambda: 1.0)
            idf_dict[self._tokenizer.sep_token_id] = 0
            idf_dict[self._tokenizer.cls_token_id] = 0

        all_preds = bert_cos_score_idf(
            self._model,
            refs,
            cands,
            self._tokenizer,
            idf_dict,
            verbose=verbose,
            device=self.device,
            batch_size=batch_size,
            all_layers=self.all_layers,
        ).cpu()

        if ref_group_boundaries is not None:
            max_preds = []
            for start, end in ref_group_boundaries:
                max_preds.append(all_preds[start:end].max(dim=0)[0])
            all_preds = torch.stack(max_preds, dim=0)

        if self.rescale_with_baseline:
            all_preds = (all_preds - self.baseline_vals) / (1 - self.baseline_vals)

        out = all_preds[..., 0], all_preds[..., 1], all_preds[..., 2]  # P, R, F

        if verbose:
            time_diff = time.perf_counter() - start
            print(f"done in {time_diff:.2f} seconds, {len(refs) / time_diff:.2f} sentences/sec")

        if return_hash:
            out = tuple([out, self.hash])

        return out


    def __repr__(self):
        return f"{self.__class__.__name__}(hash={self.hash}, batch_size={self.batch_size}, nthreads={self.nthreads})"

    def __str__(self):
        return self.__repr__()


class Dummy:
    pass

class CodeT5Retriever:
    def __init__(self, args):
        self.args = args

    def prepare_model(self, model=None, tokenizer=None, config=None):
        if self.args.log_level == 'verbose':
            transformers.logging.set_verbosity_info()
        self.model_name = self.args.model_name

        if model is None:
            self.tokenizer = transformers.RobertaTokenizer.from_pretrained(self.model_name)
            model_arg = Dummy()
            setattr(model_arg, 'sim_func', args.sim_func)
            config = AutoConfig.from_pretrained(self.model_name)
            self.model = RetrievalModel(
                config=config,
                model_type=self.model_name,
                num_layers=args.num_layers,
                tokenizer=tokenizer,
                training_args=None,
                model_args=model_arg)
            self.device = torch.device('cuda') if not self.args.cpu else torch.device('cpu')
            self.model.eval()
            self.model = self.model.to(self.device)
        else: # this is only for evaluation durning training time
            self.model = model
            self.tokenizer = tokenizer
            self.device = self.model.device

    def encode_file(self, text_file, save_file, **kwargs):
        normalize_embed = kwargs.get('normalize_embed', False)
        with open(text_file, "r") as f:
            dataset = []
            for line in f:
                dataset.append(line.strip())
                # print(line)
        print(f"number of sentences in {text_file}: {len(dataset)}")

        def pad_batch(examples):
            sentences = examples
            sent_features = self.tokenizer(
                sentences,
                add_special_tokens=True,
                max_length=self.tokenizer.model_max_length,
                truncation=True
            )
            arr = sent_features['input_ids']
            lens = torch.LongTensor([len(a) for a in arr])
            max_len = lens.max().item()
            padded = torch.ones(len(arr), max_len, dtype=torch.long) * self.tokenizer.pad_token_id
            mask = torch.zeros(len(arr), max_len, dtype=torch.long)
            for i, a in enumerate(arr):
                padded[i, : lens[i]] = torch.tensor(a, dtype=torch.long)
                mask[i, : lens[i]] = 1
            return {'input_ids': padded, 'attention_mask': mask, 'lengths': lens}

        bs = 128
        with torch.no_grad():
            all_embeddings = []
            for i in tqdm(range(0, len(dataset), bs), disable=TQDM_DISABLED):
                batch = dataset[i: i + bs]
                padded_batch = pad_batch(batch)
                for k in padded_batch:
                    if isinstance(padded_batch[k], torch.Tensor):
                        padded_batch[k] = padded_batch[k].to(self.device)
                output = self.model.get_pooling_embedding(**padded_batch, normalize=normalize_embed).detach().cpu().numpy()
                all_embeddings.append(output)

            all_embeddings = np.concatenate(all_embeddings, axis=0)
            print(f"done embedding: {all_embeddings.shape}")

        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))

        np.save(save_file, all_embeddings)

    @staticmethod
    def retrieve(source_embed_file, target_embed_file, source_id_file, target_id_file, top_k, save_file):
        print(f'source: {source_embed_file}, target: {target_embed_file}')
        with open(source_id_file, "r") as f:
            source_id_map = {}
            for idx, line in enumerate(f):
                source_id_map[idx] = line.strip()
        with open(target_id_file, "r") as f:
            target_id_map = {}
            for idx, line in enumerate(f):
                target_id_map[idx] = line.strip()

        source_embed = np.load(source_embed_file + ".npy")
        target_embed = np.load(target_embed_file + ".npy")
        assert len(source_id_map) == source_embed.shape[0]
        assert len(target_id_map) == target_embed.shape[0]
        indexer = faiss.IndexFlatIP(target_embed.shape[1])
        indexer.add(target_embed)
        print(source_embed.shape, target_embed.shape)
        D, I = indexer.search(source_embed, top_k)

        results = {}
        for source_idx, (dist, retrieved_index) in enumerate(zip(D, I)):
            source_id = source_id_map[source_idx]
            results[source_id] = {}
            retrieved_target_id = [target_id_map[x] for x in retrieved_index]
            results[source_id]['retrieved'] = retrieved_target_id
            results[source_id]['score'] = dist.tolist()

        # with open(save_file, "w+") as f:
        #     json.dump(results, f, indent=2)

        return results

def align_src_pred(src_file, pred_file):
    with open(src_file, "r") as fsrc, open(pred_file, "r") as fpred:
        src = json.load(fsrc)
        pred = json.load(fpred)['results']
        # assert len(src) == len(pred), (len(src), len(pred))

    # re-order src
    src_nl = [x['nl'] for x in src]
    _src = []
    _pred = []
    for p in pred:
        if p['nl'] in src_nl:
            _src.append(src[src_nl.index(p['nl'])])
            _pred.append(p)

    src = _src
    pred = _pred

    for s, p in zip(src, pred):
        assert s['nl'] == p['nl'], (s['nl'], p['nl'])

    print(f"unique nl: {len(set(src_nl))}")
    print(f"number of samples (src/pred): {len(src)}/{len(pred)}")
    print("pass nl matching check")

    return src, pred

def calc_metrics(src_file, pred_file):
    src, pred = align_src_pred(src_file, pred_file)

    _src = []
    _pred = []
    for s, p in zip(src, pred):
        cmd_name = s['cmd_name']
        oracle_man = get_oracle(s, cmd_name)
        pred_man = p['pred']
        _src.append(oracle_man)
        _pred.append(pred_man)
    calc_recall(_src, _pred)

    # description only
    _src = []
    for s in src:
        _src.append(s['matching_info']['|main|'])
    calc_recall(_src, _pred)

    _src = []
    _pred = []
    for s, p in zip(src, pred):
        cmd_name = s['cmd_name']
        pred_man = p['pred']
        _src.append(cmd_name)
        _pred.append(pred_man)
    calc_hit(_src, _pred)
    # calc_mean_rank(src, pred)


def calc_mean_rank(src, pred):
    rank = []
    for s, p in zip(src, pred):
        cur_rank = []
        cmd_name = s['cmd_name']
        pred_man = p['pred']
        oracle_man = get_oracle(s, cmd_name)
        for o in oracle_man:
            if o in pred_man:
                cur_rank.append(oracle_man.index(o))
            else:
                cur_rank.append(101)
        if cur_rank:
            rank.append(np.mean(cur_rank))

    print(np.mean(rank))


def calc_hit(src, pred, top_k=None):
    top_k = TOP_K if top_k is None else top_k
    hit_n = {x: 0 for x in top_k}
    assert len(src) == len(pred), (len(src), len(pred))

    for s, p in zip(src, pred):
        cmd_name = s
        pred_man = p

        for tk in hit_n.keys():
            cur_result_vids = pred_man[:tk]
            cur_hit = any([cmd_name in x for x in cur_result_vids])
            hit_n[tk] += cur_hit

    hit_n = {k: v / len(pred) for k, v in hit_n.items()}
    for k in sorted(hit_n.keys()):
        print(f"{hit_n[k] :.3f}", end="\t")
    print()
    return hit_n

def get_oracle(item, cmd_name):
    # oracle = [f"{cmd_name}_{x}" for x in itertools.chain(*item['matching_info'].values())]
    oracle = [f"{cmd_name}_{x}" for x in item['oracle_man']]
    return oracle

def calc_recall(src, pred, print_result=True, top_k=None):
    top_k = TOP_K if top_k is None else top_k
    recall_n = {x: 0 for x in top_k}
    precision_n = {x: 0 for x in top_k}

    for s, p in zip(src, pred):
        # cmd_name = s['cmd_name']
        oracle_man = s
        pred_man = p

        for tk in recall_n.keys():
            cur_result_vids = pred_man[:tk]
            cur_hit = sum([x in cur_result_vids for x in oracle_man])
            # recall_n[tk] += cur_hit / (len(oracle_man) + 1e-10)
            recall_n[tk] += cur_hit / (len(oracle_man)) if len(oracle_man) else 1
            precision_n[tk] += cur_hit / tk
    recall_n = {k: v / len(pred) for k, v in recall_n.items()}
    precision_n = {k: v / len(pred) for k, v in precision_n.items()}

    if print_result:
        for k in sorted(recall_n.keys()):
            print(f"{recall_n[k] :.3f}", end="\t")
        print()
        for k in sorted(precision_n.keys()):
            print(f"{precision_n[k] :.3f}", end="\t")
        print()
        for k in sorted(recall_n.keys()):
            print(f"{2 * precision_n[k] * recall_n[k] / (precision_n[k] + recall_n[k] + 1e-10) :.3f}", end="\t")
        print()

    return {'recall': recall_n, 'precision': precision_n}

def clean_dpr_results(result_file):
    results = {'results': [], 'metrics': {}}
    with open(result_file, "r") as f:
        d = json.load(f)
    for _item in d:
        item = {}
        item['nl'] = _item['question']
        item['pred'] = [x['id'] for x in _item['ctxs']]
        results['results'].append(item)

    with open(result_file + ".clean", "w+") as f:
        json.dump(results, f, indent=2)


def eval_retrieval_from_loaded(data_file, r_d):
    # for conala
    with open(data_file, "r") as f:
        d = json.load(f)
    gold = [item['oracle_man'] for item in d]
    pred = [r_d[x['question_id']]['retrieved'] for x in d]
    metrics = calc_recall(gold, pred, print_result=False)
    return metrics
  
## run everything above this only once
if __name__ == "__main__":
    args = {}
    args["model_name"] = "neulab/docprompting-codet5-python-doc-retriever"
    args["source_file"] = "data/conala/conala_nl.txt"
    args["target_file"] = "data/conala/python_manual_firstpara.tok.txt"
    args["source_embed_save_file"] = "data/conala/.tmp/src_embedding"
    args["target_embed_save_file"] = "data/conala/.tmp/tgt_embedding"
    args["sim_func"] = "cls_distance.cosine"
    args["num_layers"] = 12
    args["oracle_eval_file"] = "data/conala/cmd_dev.oracle_man.full.json"
    args["save_file"] = "data/conala/retrieval_results.json"
    args["source_idx_file"] = args.source_file.replace(".txt", ".id")
    args["target_idx_file"] = args.target_file.replace(".txt", ".id")

    searcher = CodeT5Retriever(args)
    searcher.prepare_model()
    searcher.encode_file(args['source_file'], args['source_embed_save_file'], normalize_embed=args.normalize_embed)
    searcher.encode_file(args['target_file'], args['target_embed_save_file'], normalize_embed=args.normalize_embed)
    results = searcher.retrieve(args.source_embed_save_file,
                    args.target_embed_save_file, args.source_idx_file,
                    args.target_idx_file, args.top_k, args.save_file)

    flag = 'recall'
    top_n = 10

    m1 = eval_retrieval_from_loaded(args.oracle_eval_file, results)
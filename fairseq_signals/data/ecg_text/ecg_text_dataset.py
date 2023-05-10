import os
import sys
import logging

import scipy.io
import numpy as np
import torch

from transformers import (
    DataCollatorForWholeWordMask,
    DataCollatorForLanguageModeling,
    BertTokenizerFast
)

from fairseq_signals.data.ecg.raw_ecg_dataset import RawECGDataset

logger = logging.getLogger(__name__)

class RawECGTextDataset(RawECGDataset):
    def __init__(
        self,
        max_text_size=512,
        tokenizer="bert-base-uncased",
        compute_mlm_indices=False,
        mlm_prob=0.15,
        #XXX to be removed
        medvill=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.tokenizer = BertTokenizerFast.from_pretrained(
            tokenizer, do_lower_case="uncased" in tokenizer
        )

        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token = self.tokenizer.pad_token_id
        self.sep_token = self.tokenizer.sep_token_id

        # we minus 2 to the max_text_size since special tokens [SEP] will be pre / appended there
        self.max_text_size = (
            max_text_size - 2 if max_text_size is not None else sys.maxsize
        )
        self.min_text_size = 0
        if medvill:
            self.min_text_size = 10

        self.compute_mlm_indices = compute_mlm_indices
        self.mlm_prob = mlm_prob
        if self.compute_mlm_indices:
            self.mlm_collator = DataCollatorForWholeWordMask(
                tokenizer=self.tokenizer,
                mlm=True,
                mlm_probability=self.mlm_prob
            )
        
        #XXX to be removed
        self.medvill = medvill

    def collator(self, samples):
        texts = [s['text'] for s in samples]
        if len(texts) == 0:
            return {}

        encodings = self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=self.max_text_size,
            return_special_tokens_mask=True,
            return_offsets_mapping=True,
        )

        collated_texts = torch.LongTensor(encodings["input_ids"])
        text_padding_mask = ~torch.BoolTensor(encodings["attention_mask"])

        text_sizes = [(x != 0).sum().item() for x in collated_texts]
        valid_indices = [i for i, x in enumerate(text_sizes) if x > self.min_text_size]

        samples = [x for i, x in enumerate(samples) if i in valid_indices]
        collated_texts = collated_texts[valid_indices]
        text_padding_mask = text_padding_mask[valid_indices]

        _collated_ecgs = super().collator(
            [{'source': s['ecg'], 'id': s['id']} for s in samples]
        )
        if len(_collated_ecgs) == 0:
            return {}

        collated_ecgs = _collated_ecgs['net_input']['source']
        ecg_padding_mask = _collated_ecgs['net_input']['padding_mask']

        input = {
            'ecg': collated_ecgs,
            'ecg_padding_mask': ecg_padding_mask,
        }

        # TODO to be moved to MedViLLECGDataset
        if "diagnoses" in samples[0]:
            diagnoses = [s["diagnoses"] for s in samples]

            bsz = len(diagnoses)
            candidates = set(np.arange(bsz))
            is_aligned = torch.ones((bsz,))
            num_negatives = int((bsz * 0.5 + np.random.rand()) / 2.0)

            neg_idcs = np.random.choice(bsz, size=num_negatives, replace=False)
            candidates = candidates - set(neg_idcs)
            failed = []

            candidates = np.random.permutation(list(candidates))
            for neg_idx in neg_idcs:
                dx = diagnoses[neg_idx]
                neg = self.sample_negative(dx, candidates, diagnoses)
                if neg is not None:
                    candidates = candidates[candidates != neg]
                    # swap texts between negative samples
                    collated_texts[neg_idx], collated_texts[neg] = (
                        collated_texts[neg], collated_texts[neg_idx]
                    )
                    text_padding_mask[neg_idx], text_padding_mask[neg] = (
                        text_padding_mask[neg], text_padding_mask[neg_idx]
                    )
                    is_aligned[neg_idx] = 0
                    is_aligned[neg] = 0
                else:
                    failed.append(neg_idx)

            # if some samples failed to find negative samples, retry with other candidates
            if len(failed) > 0:
                num_negatives = len(failed)
                candidates = np.append(candidates, failed)
                neg_idcs = np.random.choice(candidates, size=num_negatives, replace=False)

                candidates = set(candidates) - set(neg_idcs)
                candidates = np.random.permutation(list(candidates))
                for neg_idx in neg_idcs:
                    dx = diagnoses[neg_idx]
                    neg = self.sample_negative(dx, candidates, diagnoses)
                    if neg is not None:
                        candidates = candidates[candidates != neg]
                        # swap texts between negative samples
                        collated_texts[neg_idx], collated_texts[neg] = (
                            collated_texts[neg], collated_texts[neg_idx]
                        )
                        text_padding_mask[neg_idx], text_padding_mask[neg] = (
                            text_padding_mask[neg], text_padding_mask[neg_idx]
                        )
                        is_aligned[neg_idx] = 0
                        is_aligned[neg] = 0

        if self.compute_mlm_indices:
            flatten_encodings = [
                {key: encodings[key][i] for key in encodings.keys()}
                for i in range(len(encodings["input_ids"]))
            ]
            collated_encodings = self.mlm_collator(flatten_encodings)
            collated_texts = collated_encodings["input_ids"]
            
            input["text"] = collated_texts
        else:
            input["text"] = collated_texts

        text_attention_mask = ~text_padding_mask
        input["text_padding_mask"] = text_padding_mask
        input["text_attention_mask"] = text_attention_mask

        out = {
            'id': torch.LongTensor([s['id'] for s in samples]),
            'net_input': input
        }
        # TODO change condition with arguments
        if "diagnoses" in samples[0]:
            out["is_aligned"] = is_aligned
        if self.compute_mlm_indices:
            out["mlm_labels"] = collated_encodings["labels"]

        return out

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""

        if self.shuffle:
            order = [np.random.permutation(len(self))]
            return order[0]
        else:
            return np.arange(len(self))
    
    def sample_negative(self, dx, candidates, diagnoses):
        for cand in candidates:
            # NOTE define negative samples as "not exactly matched samples"
            #   we can modify this to restrict to be of "not sharing any dx codes samples"
            #   by checking intersection between dx and diagnoses[cand]
            if dx != diagnoses[cand]:
                return cand
        return None

class FileECGTextDataset(RawECGTextDataset):
    def __init__(self, manifest_path, num_buckets=0, **kwargs):
        super().__init__(**kwargs)

        skipped = 0
        self.fnames = []
        sizes = []
        self.skipped_indices = set()

        with open(manifest_path, 'r') as f:
            self.root_dir = f.readline().strip()
            for i, line in enumerate(f):
                items = line.strip().split('\t')
                assert len(items) == 2, line
                ecg_sz = int(items[1])
                if self.min_sample_size is not None and ecg_sz < self.min_sample_size:
                    skipped += 1
                    self.skipped_indices.add(i)
                    continue
                self.fnames.append(items[0])
                sizes.append(ecg_sz)
        logger.info(f"loaded {len(self.fnames)}, skipped {skipped} samples")

        self.sizes = np.array(sizes, dtype=np.int64)

        try:
            import pyarrow

            self.fnames = pyarrow.array(self.fnames)
        except:
            logger.debug(
                "Could not create a pyarraw array. Please install pyarrow for better performance"
            )
            pass

        self.set_bucket_info(num_buckets)

    def __getitem__(self, index):
        path = os.path.join(self.root_dir, str(self.fnames[index]))

        res = {'id': index}

        data = scipy.io.loadmat(path)

        curr_sample_rate = data['curr_sample_rate']
        feats = torch.from_numpy(data['feats'])
        res['ecg'] = self.postprocess(feats, curr_sample_rate)

        res["text"] = data["text"][0]

        #XXX to be moved to MedViLLDataset
        if "diagnoses" in data:
            res["diagnoses"] = [x.strip() for x in data["diagnoses"]]

        return res

    def __len__(self):
        return len(self.fnames)
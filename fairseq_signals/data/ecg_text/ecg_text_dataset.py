import os
import sys
import logging

import scipy.io
import numpy as np
import torch

from fairseq_signals.data.ecg.raw_ecg_dataset import RawECGDataset

logger = logging.getLogger(__name__)

class RawECGTextDataset(RawECGDataset):
    def __init__(
        self,
        pad_token_id,
        sep_token_id,
        max_text_size=512,
        min_text_size=10,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.pad_token = pad_token_id
        self.sep_token = sep_token_id

        # we minus 2 to the max_text_size since special tokens [SEP] will be pre / appended there
        self.max_text_size = (
            max_text_size - 2 if max_text_size is not None else sys.maxsize
        )
        self.min_text_size = min_text_size

    def collator(self, samples):
        _collated_ecgs = super().collator(
            [{'source': s['ecg'], 'id': s['id']} for s in samples]
        )
        if len(_collated_ecgs) == 0:
            return {}
        collated_ecgs = _collated_ecgs['net_input']['source']
        ecg_padding_mask = _collated_ecgs['net_input']['padding_mask']

        texts = [s['text'] for s in samples]
        sizes = [t.size(-1) for t in texts]

        if self.pad:
            target_size = min(max(sizes), self.max_text_size)
        else:
            # target_size = min(min(sizes), self.min_text_size)
            raise AssertionError(
                'ECG-TEXT multimodal task should be run with padding'
            )
        
        collated_texts = texts[0].new_zeros((len(texts), target_size))
        text_padding_mask = (
            torch.BoolTensor(collated_texts.shape).fill_(False) if self.pad else None
        )
        for i, (text, size) in enumerate(zip(texts, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_texts[i] = text
            elif diff < 0:
                assert self.pad
                collated_texts[i] = torch.cat(
                    [text, text.new_full((-diff,), self.pad_token)]
                )
                text_padding_mask[i, diff:] = True
            else:
                collated_texts[i] = self.crop_to_max_size(text, target_size, rand=False)

        # TODO change condition with arguments
        if 'diagnoses' in samples[0]:
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

        input = {
            'ecg': collated_ecgs,
            'ecg_padding_mask': ecg_padding_mask,
            'text': collated_texts,
            'text_padding_mask': text_padding_mask,
        }

        out = {
            'id': torch.LongTensor([s['id'] for s in samples]),
            'net_input': input
        }
        # TODO change condition with arguments
        if 'diagnoses' in samples[0]:
            out["is_aligned"] = is_aligned

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
                assert len(items) == 3, line
                ecg_sz = int(items[1])
                text_sz = int(items[2])
                if self.min_sample_size is not None and ecg_sz < self.min_sample_size:
                    skipped += 1
                    self.skipped_indices.add(i)
                    continue
                if self.min_text_size is not None and text_sz < self.min_text_size:
                    skipped += 1
                    self.skipped_indices.add(i)
                    continue
                self.fnames.append(items[0])
                sizes.append(ecg_sz + text_sz)
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

        text = torch.from_numpy(data['text'][0])
        res['text'] = torch.cat([
            torch.LongTensor([self.sep_token]),
            text,
            torch.LongTensor([self.sep_token])
        ])

        if "diagnoses" in data:
            res["diagnoses"] = [x.strip() for x in data["diagnoses"]]

        return res

    def __len__(self):
        return len(self.fnames)
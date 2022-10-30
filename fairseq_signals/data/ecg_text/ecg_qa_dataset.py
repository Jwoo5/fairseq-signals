import os
import sys
import logging

import wfdb
import torch
import numpy as np
import scipy.io

from fairseq_signals.data.ecg_text.ecg_text_dataset import RawECGTextDataset

logger = logging.getLogger(__name__)

class FileECGQADataset(RawECGTextDataset):
    def __init__(
        self,
        manifest_path,
        num_buckets=0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        skipped = 0
        self.fnames = []
        sizes = []
        self.skipped_indices = set()

        with open(manifest_path, 'r') as f:
            self.root_dir = f.readline().strip()
            self.tokenized = eval(f.readline().strip())
            self.num_labels = int(f.readline().strip())

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

        if not self.tokenized:
            from transformers import BertTokenizer
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        try:
            import pyarrow

            self.fnames = pyarrow.array(self.fnames)
        except:
            logger.debug(
                "Could not create a pyarraw array. Please install pyarrow for better performance"
            )
            pass

        self.set_bucket_info(num_buckets)

    def collator(self, samples):
        collated_samples = super().collator(samples)
        if len(collated_samples) == 0:
            return {}
        input = {
            'ecg': collated_samples['net_input']['ecg'],
            'ecg_padding_mask': collated_samples['net_input']['ecg_padding_mask'],
            'text': collated_samples['net_input']['text'],
            'text_padding_mask': collated_samples['net_input']['text_padding_mask']
        }


        out = {'id': torch.LongTensor([s['id'] for s in samples])}
        samples = [s for s in samples if s['ecg'] is not None]
        answers = [s['answer'] for s in samples]
        out['answer'] = torch.cat(answers)

        out['net_input'] = input

        return out

    def __getitem__(self, index):
        path = os.path.join(self.root_dir, str(self.fnames[index]))
        
        res = {'id': index}

        data = scipy.io.loadmat(path)

        ecg, _ = wfdb.rdsamp(data['ecg_path'][0])
        ecg = torch.from_numpy(ecg.T)
        res['ecg'] = self.postprocess(ecg)

        question = torch.from_numpy(data['question'][0])
        if not self.tokenized:
            question = self.tokenizer.encode(question.lower(), add_special_tokens=False)
        res['text'] = torch.cat([
            torch.LongTensor([self.sep_token]),
            question,
            torch.LongTensor([self.sep_token])
        ])

        answer = data['answer']
        res['answer'] = torch.LongTensor(answer)

        return res

    def __len__(self):
        return len(self.fnames)
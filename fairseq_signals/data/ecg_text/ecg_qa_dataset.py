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
                assert len(items) == 3 or len(items) == 4, line
                if len(items) == 3:
                    ecg_sz = int(items[1])
                    text_sz = int(items[2])
                else:
                    ecg_sz = min(int(items[1]), int(items[2]))
                    text_sz = int(items[3])
                if self.min_sample_size is not None and ecg_sz < self.min_sample_size:
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
            'text_padding_mask': collated_samples['net_input']['text_padding_mask'],
        }

        samples = [s for s in samples if s['ecg'] is not None]

        # we allow to see up to 2 ecgs simultaneously for a question
        ecgs_2 = [s["ecg_2"] for s in samples]
        sizes = [s.size(-1) if s is not None else 0 for s in ecgs_2]
        target_size = min(max(sizes), self.max_sample_size)
        # if there is no second ecg in the batch
        if target_size == 0:
            input["ecg_2"] = None
            input["ecg_2_padding_mask"] = None
        else:
            if target_size > 0:
                for i in range(len(ecgs_2)):
                    if ecgs_2[i] is not None:
                        not_null_index = i
                        break
                collated_ecgs_2 = ecgs_2[not_null_index].new_zeros(
                    (len(ecgs_2), len(ecgs_2[not_null_index]), target_size)
                )
            else:
                collated_ecgs_2 = torch.zeros(
                    (len(ecgs_2), len(input["ecg"][0]), 1),
                )
            padding_mask = (
                torch.BoolTensor(collated_ecgs_2.shape).fill_(False)
            )
            for i, (ecg, size) in enumerate(zip(ecgs_2, sizes)):
                diff = size - target_size
                if diff == 0:
                    collated_ecgs_2[i] = ecg
                elif diff < 0:
                    if ecg is not None:
                        collated_ecgs_2[i] = torch.cat(
                            [ecg, ecg.new_full((ecg.shape[0], -diff,), 0.0)], dim=-1
                        )
                    else:
                        collated_ecgs_2[i] = torch.full(
                            (ecgs_2[not_null_index].shape[0], -diff,), 0.0
                        )
                    padding_mask[i, :, diff:] = True
                else:
                    collated_ecgs_2[i], start, end = self.crop_to_max_size(ecg, target_size, rand=True)

            input["ecg_2"] = collated_ecgs_2
            input["ecg_2_padding_mask"] = padding_mask

        out = {'id': torch.LongTensor([s['id'] for s in samples])}
        if "template_id" in samples[0]:
            out["template_id"] = torch.LongTensor([s["template_id"] for s in samples])
        if "attribute_id" in samples[0]:
            out["attribute_id"] = torch.LongTensor([s["attribute_id"] for s in samples])
        #XXX zero-shot
        # if "attribute_idx" in samples[0]:
        #     out["attribute_idx"] = torch.LongTensor([s["attribute_idx"] for s in samples])
        out["question_id"] = torch.LongTensor([s["question_id"] for s in samples])
        #XXX
        if "question_type1" in samples[0]:
            out["question_type1"] = torch.LongTensor([s["question_type1"] for s in samples])
            out["question_type2"] = torch.LongTensor([s["question_type2"] for s in samples])
            out["question_type3"] = torch.LongTensor([s["question_type3"] for s in samples])

        out["is_multi_class"] = torch.LongTensor([s["is_multi_class"] for s in samples])
        out['answer'] = torch.cat([s['answer'] for s in samples])
        out["classes"] = [s["classes"] for s in samples]

        out['net_input'] = input

        #XXX
        # out["ecg_id"] = torch.LongTensor([s["ecg_id"] for s in samples])

        return out

    def __getitem__(self, index):
        path = os.path.join(self.root_dir, str(self.fnames[index]))
        
        res = {'id': index}

        data = scipy.io.loadmat(path)

        ecg, _ = wfdb.rdsamp(data["ecg_path"][0])
        ecg = torch.from_numpy(ecg.T)
        res["ecg"] = self.postprocess(ecg)
        # we allow to see up to 2 ecgs simultaneously for a question
        if len(data["ecg_path"]) == 2:
            ecg_2, _ = wfdb.rdsamp(data["ecg_path"][1])
            ecg_2 = torch.from_numpy(ecg_2.T)
            res["ecg_2"] = self.postprocess(ecg_2)
        else:
            res["ecg_2"] = None

        # question = torch.from_numpy(data['question'][0])
        # if not self.tokenized:
        #     question = self.tokenizer.encode(question.lower(), add_special_tokens=False)
        # res['text'] = torch.cat([
        #     torch.LongTensor([self.sep_token]),
        #     question,
        #     torch.LongTensor([self.sep_token])
        # ])

        #XXX
        res["text"] = data["question_str"][0]

        answer = data['answer']
        res['answer'] = torch.LongTensor(answer)

        if "template_id" in data:
            res["template_id"] = data["template_id"][0][0]
        if "attribute_id" in data:
            res["attribute_id"] = data["attribute_id"][0][0]
        #XXX zero-shot
        # if "attribute_idx" in data:
        #     res["attribute_idx"] = data["attribute_idx"][0][0]
        res["question_id"] = data["question_id"][0][0]
        res["question_type1"] = data["qtype1"][0][0]
        res["question_type2"] = data["qtype2"][0][0]
        res["question_type3"] = data["qtype3"][0][0]
        res["is_multi_class"] = data["atype"][0][0]
        res["classes"] = torch.LongTensor(data["classes"][0])

        #XXX
        # ecg_id = data["ecg_id"][0]
        # res["ecg_id"] = ecg_id

        return res

    def __len__(self):
        return len(self.fnames)
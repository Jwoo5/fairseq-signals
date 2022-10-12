import sys
import logging

import json
import wfdb
import torch
import numpy as np

from fairseq_signals.data.ecg.raw_ecg_dataset import RawECGDataset

logger = logging.getLogger(__name__)

class JsonECGQADataset(RawECGDataset):
    def __init__(
        self,
        json_path,
        pad_token_id,
        sep_token_id,
        max_text_size=512,
        min_text_size=0,
        pad=True,
        **kwargs,
    ):
        super().__init__(pad=pad, **kwargs)

        self.pad_token = pad_token_id
        self.sep_token = sep_token_id

        self.max_text_size = (
            max_text_size if max_text_size is not None else sys.maxsize
        )
        self.min_text_size = min_text_size

        with open(json_path, 'r') as f:
            json_data = json.load(f)

        self.fnames = [sample['ecg_path'] for sample in json_data['samples']]
        self.answer_type = json_data['answer_type']

        if not json_data['tokenized']:
            logger.info(
                "Detected data hasn't been tokenized yet. "
                "So we automatically tokenize questions from pretrained "
                "`transformers.models.bert.tokenization_bert.BertTokenizer` "
                "which is loaded with 'bert-base-uncased'"
            )

            from transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            questions = [sample['question'].lower() for sample in json_data['samples']]
            answers = [sample['answer'].lower() for sample in json_data['samples']]

            self.questions = tokenizer(questions, add_special_tokens=False)['input_ids']
            if self.answer_type == 'text':
                self.answers = tokenizer(answers, add_special_tokens=False)['input_ids']
        else:
            self.questions = [sample['question'] for sample in json_data['samples']]
            self.answers = [sample['answer'] for sample in json_data['samples']]

        ecg_sizes = [sample['size'] for sample in json_data['samples']]
        text_sizes = [len(q) for q in self.questions]
        sizes = [size1 + size2 for size1, size2 in zip(ecg_sizes, text_sizes)]
        self.skipped_indices = set()
        for i, (ecg_size, text_size) in enumerate(zip(ecg_sizes, text_sizes)):
            if self.min_sample_size is not None and ecg_size < self.min_sample_size:
                self.skipped_indices.add(i)
            if self.min_text_size is not None and text_size < self.min_text_size:
                self.skipped_indices.add(i)

        skipped = len(self.skipped_indices)
        logger.info(f"loaded {len(self.fnames)}, skipped {skipped} samples")

        self.sizes = np.array(sizes, dtype=np.int64)

        self.use_pa = False
        try:
            import pyarrow
            self.use_pa = True

            self.fnames = pyarrow.array(self.fnames)
            self.questions = pyarrow.array(self.questions)
            self.answers = pyarrow.array(self.answers)
        except:
            logger.debug(
                "Could not create a pyarraw array. Please install pyarrow for better performance"
            )
            pass

    def postprocess(self, feats):
        feats = super().postprocess(feats, curr_sample_rate=self.sample_rate)

        return feats

    def collator(self, samples):
        _collated_ecgs = super().collator(
            [{'source': s['ecg'], 'id': s['id']} for s in samples]
        )
        if len(_collated_ecgs) == 0:
            return {}
        collated_ecgs = _collated_ecgs['net_input']['source']
        ecg_padding_mask = _collated_ecgs['net_input']['padding_mask']

        questions = [s['question'] for s in samples]
        sizes = [q.size(-1) for q in questions]

        if self.pad:
            target_size = min(max(sizes), self.max_text_size)
        else:
            # target_size = min(min(sizes), self.min_text_size)
            raise AssertionError(
                'QA task should be run with --pad=True. '
                'Check your configurations.'
            )

        collated_questions = questions[0].new_zeros((len(questions), target_size))
        question_padding_mask = (
            torch.BoolTensor(collated_questions.shape).fill_(False) if self.pad else None
        )
        for i, (question, size) in enumerate(zip(questions, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_questions[i] = question
            elif diff < 0:
                assert self.pad
                collated_questions[i] = torch.cat(
                    [question, question.new_full((-diff,), self.pad_token)]
                )
                question_padding_mask[i, diff:] = True
            else:
                collated_questions[i] = self.crop_to_max_size(question, target_size, rand=False)

        input = {
            'ecg': collated_ecgs,
            'ecg_padding_mask': ecg_padding_mask,
            'question': collated_questions,
            'question_padding_mask': question_padding_mask,
        }

        out = {
            'id': torch.LongTensor([s['id'] for s in samples]),
            'answer': torch.stack([s['answer'] for s in samples]),
            'net_input': input
        }
        return out

    def __getitem__(self, index):
        res = {'id': index}

        ecg, _ = wfdb.rdsamp(str(self.fnames[index]))
        ecg = torch.from_numpy(ecg.T)
        res['ecg'] = self.postprocess(ecg)

        question = self.questions[index]
        answer = self.answers[index]
        if self.use_pa:
            question = question.as_py()
            answer = answer.as_py()

        res['question'] = torch.cat([
            torch.LongTensor([self.sep_token]),
            torch.LongTensor(question),
            torch.LongTensor([self.sep_token])
        ])
        res['answer'] = torch.LongTensor(answer)

        return res

    def __len__(self):
        return len(self.fnames)
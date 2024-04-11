import os
import glob
import argparse
import json

import pandas as pd
import numpy as np
import wfdb
import scipy.io

from tqdm import tqdm

from transformers import BertTokenizerFast

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", metavar="DIR", default=".",
        help='root directory containing ptbxl files to pre-process'
    )
    parser.add_argument(
        "--dest", type=str, metavar="DIR", default=".",
        help='output directory'
    )
    parser.add_argument(
        "--rebase", action="store_true",
        help="if set, remove and create directory for --dest"
    )
    parser.add_argument(
        "--apply_paraphrase", action="store_true",
        help="if set, preprocess paraphrased version of ecgqa dataset"
    )

    return parser

def main(args):
    split = ["train", "valid", "test"]

    dir_path = os.path.realpath(args.root)
    dest_path = os.path.realpath(args.dest)

    subdir = "paraphrased" if args.apply_paraphrase else "template"

    classes = dict(pd.read_csv(os.path.join(dir_path, "answers.csv"))["class"])
    classes = {v: k for k, v in classes.items()}
    del classes["none"]
    classes = {k: i for i, k in enumerate(classes.keys())}

    classes_for_each_template = pd.read_csv(os.path.join(dir_path, "answers_for_each_template.csv"))
    classes_for_each_template = {
        i: eval(c) for i, c in zip(classes_for_each_template["template_id"], classes_for_each_template["classes"])
    }
    for k, v in classes_for_each_template.items():
        if "none" in v:
            v.remove("none")

    tokenizer = BertTokenizerFast.from_pretrained(
        "bert-base-uncased", do_lower_case=True
    )

    if args.rebase:
        import shutil
        shutil.rmtree(dest_path)

    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    for subset in split:
        with open(os.path.join(dest_path, subset + ".tsv"), "w") as f:
            print(os.path.join(dest_path, subset), file=f)

            if not os.path.exists(os.path.join(dest_path, subset)):
                os.makedirs(os.path.join(dest_path, subset))

            data = []
            for fname in sorted(glob.glob(os.path.join(dir_path, subdir, subset, "*.json"))):
                with open(fname, "r") as data_f:
                    data.extend(json.load(data_f))

            # look over single-verify samples to collect attribute ids aligned with upperbound experiments
            attribute_ids = []
            for sample_data in tqdm(data, total=len(data)):
                if sample_data["question_type"] == "single-verify":
                    # for ptb-xl
                    if "myocardial infarction in " in sample_data["attribute"][0]:
                        sample_data["attribute"][0] = "myocardial infarction"
                    elif "subendocardial injury in " in sample_data["attribute"][0]:
                        sample_data["attribute"][0] = "subendocardial injury"
                    elif "ischemic in " in sample_data["attribute"][0]:
                        sample_data["attribute"][0] = "ischemic"
                    # for mimic-iv-ecg
                    elif "Myocardial infarction in " in sample_data["attribute"][0]:
                        sample_data["attribute"][0] = "Myocardial infarction"
                    elif "ischemic ST-T changes in " in sample_data["attribute"][0]:
                        sample_data["attribute"][0] = "ischemic ST-T changes"

                    if sample_data["attribute"][0] not in attribute_ids:
                        attribute_ids.append(sample_data["attribute"][0])
            attribute_ids.sort()
            attribute_id_map = {k: i for i, k in enumerate(attribute_ids)}

            for i, sample_data in enumerate(tqdm(data, total=len(data), desc=subset)):
                sample = {}
                sample["ecg_path"] = sample_data["ecg_path"]
                sample["question"] = sample_data["question"]
                sample["answer"] = encode_answer(sample_data["answer"], classes)

                qtype1, qtype2 = tuple(sample_data["question_type"].split("-"))
                atype = sample_data["attribute_type"]

                if qtype2 == "choose":
                    if "the normal range" in sample_data["attribute"][0]:
                        sample["valid_classes"] = [classes[a[:a.index(" of ")]] for a in sample_data["attribute"]]
                    else:
                        sample["valid_classes"] = [classes[c] for c in sample_data["attribute"]]
                    sample["valid_classes"].sort()
                else:
                    sample["valid_classes"] = [classes[c] for c in classes_for_each_template[sample_data["template_id"]]]

                qtype1, qtype2, qtype3 = determine_question_type(qtype1, qtype2, atype)
                sample["question_type1"] = qtype1
                sample["question_type2"] = qtype2
                sample["question_type3"] = qtype3

                if sample_data["question_type"] == "single-verify" and sample_data["answer"][0] != "not sure":
                    sample["attribute_id"] = attribute_id_map[sample_data["attribute"][0]]
                else:
                    sample["attribute_id"] = -1

                fname = str(i) + ".mat"
                scipy.io.savemat(os.path.join(dest_path, subset, fname), sample)
                print(fname, file=f, end="\t")
                for ecg_path in sample["ecg_path"]:
                    ecg, _ = wfdb.rdsamp(ecg_path)
                    if np.isnan(ecg).any():
                        print()
                        print("nan detected")
                        print()
                        breakpoint()
                    print(len(ecg), file=f, end="\t")
                text_sz = len(tokenizer.encode(sample_data["question"]))
                print(text_sz, file=f)

def encode_answer(answer, classes):
    label = np.zeros(len(classes), dtype=int)
    # "none" is an empty label
    idx = [classes[a] for a in answer if a != "none"]
    label[idx] = 1
    return label

def determine_question_type(qtype1, qtype2, attr_type):
    num_qtype2 = 3
    num_attr_type = 6

    if qtype1 == "single":
        question_type1 = 0
    elif qtype1 == "comparison_consecutive":
        question_type1 = 1
    elif qtype1 == "comparison_irrelevant":
        question_type1 = 2
    else:
        raise ValueError(qtype1)

    base = question_type1 * num_qtype2
    if qtype2 == "verify":
        question_type2 = base
    elif qtype2 == "choose":
        question_type2 = base + 1
    elif qtype2 == "query":
        question_type2 = base + 2
    else:
        raise ValueError(qtype1)

    base = question_type2 * num_attr_type
    if attr_type == "scp_code":
        question_type3 = base
    elif attr_type == "noise":
        question_type3 = base + 1
    elif attr_type == "stage_of_infarction":
        question_type3 = base + 2
    elif attr_type == "extra_systole":
        question_type3 = base + 3
    elif attr_type == "heart_axis":
        question_type3 = base + 4
    elif attr_type == "numeric_feature":
        question_type3 = base + 5
    else:
        raise ValueError(attr_type)
    
    return question_type1, question_type2, question_type3

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
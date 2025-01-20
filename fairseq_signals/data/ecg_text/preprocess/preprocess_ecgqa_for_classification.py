import os
import re
import glob
import argparse
import json
import collections
from typing import List, Dict

import pandas as pd
import wfdb
import scipy.io
import numpy as np

from tqdm import tqdm

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", metavar="DIR", default=".",
        help='root directory containing ecg-qa files to pre-process'
    )
    parser.add_argument(
        "--dest", type=str, metavar="DIR", default=".",
        help='output directory'
    )

    return parser

def main(args):
    split = ["train", "valid", "test"]

    dir_path = os.path.realpath(args.root)
    dest_path = os.path.realpath(args.dest)

    lead_positions = {
        "entire": (0,1,2,3,4,5,6,7,8,9,10,11),
        "lead I": (0,),
        "lead II": (1,),
        "lead III": (2,),
        "lead aVR": (3,),
        "lead aVL": (4,),
        "lead aVF": (5,),
        "lead V1": (6,),
        "lead V2": (7,),
        "lead V3": (8,),
        "lead V4": (9,),
        "lead V5": (10,),
        "lead V6": (11,),
        "limb leads": (0, 1, 2, 3, 4, 5),
        "chest leads": (6, 7, 8, 9, 10, 11),
        "inferior leads": (1, 2, 5,),
        "inferoseptal leads": (1, 2, 5, 6, 7),
        "inferolateral leads": (0, 1, 2, 4, 5, 10, 11,),
        "inferoposterior leads": (1, 2, 5, 6, 7, 8,),
        "inferoposterolateral leads": (0, 1, 2, 4, 5, 6, 7, 8, 10, 11,),
        "anterior leads": (8, 9,),
        "anteroseptal leads": (6, 7, 8, 9,),
        "anterolateral leads": (0, 4, 8, 9, 10, 11,),
        "lateral leads": (0, 4, 10, 11,),
        "posterior leads": (6, 7, 8,),
        "septal leads": (6, 7,),
        "posterior leads": (9, 10, 11),
        "frontal leads": (0, 1, 2, 3, 4, 5),
        "horizontal leads": (6, 7, 8, 9, 10, 11),
        "diffuse leads": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
        "frontal and horizontal leads": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
    }
    lead_names = [
        'lead I', 'lead II', 'lead III', 'lead aVR', 'lead aVL', 'lead aVF',
        'lead V1', 'lead V2', 'lead V3', 'lead V4', 'lead V5', 'lead V6'
    ]
    lead_groups = {
        "limb leads": ["lead I", "lead II", "lead III", "lead aVR", "lead aVL", "lead aVF"],
        "chest leads": ["lead V1", "lead V2", "lead V3", "lead V4", "lead V5", "lead V6"]
    }

    classes = dict(pd.read_csv(os.path.join(dir_path, "answers.csv"))["class"])
    classes = {v: k for k, v in classes.items()}
    del classes["none"]
    classes = {k: i for i, k in enumerate(classes.keys())}

    grounding_data = {"train": [], "valid": [], "test": []}
    for subset in split:
        if not os.path.exists(os.path.join(dest_path, subset)):
            os.makedirs(os.path.join(dest_path, subset))

        data = []
        for fname in sorted(glob.glob(os.path.join(dir_path, "template", subset, "*.json"))):
            with open(fname, "r") as data_f:
                data.extend(json.load(data_f))
        
        grounding_samples = []
        for i, sample_data in enumerate(tqdm(data, total=len(data), postfix=subset)):
            # we only convert single questions to classification format
            if not sample_data["question_type"].startswith("single"):
                continue
            
            ecg_id = sample_data["ecg_id"][0]
            ecg_path = sample_data["ecg_path"][0]
            # lead
            attr = sample_data["attribute"]
            assert attr is not None
            question = sample_data["question"]
            answer = sample_data["answer"]
            question_type2 = sample_data["question_type"].split("-")[-1]
            
            grounding_ans = None
            grounding_attr = None
            grounding_obj = None
            if question_type2 == "verify":
                if answer[0] == "not sure":
                    continue
                else:
                    if "excluding uncertain symptoms" in question:
                        if answer[0] == "no":
                            continue
                        else:
                            grounding_ans = ["yes"]
                    elif "including uncertain symptoms" in question:
                        if answer[0] == "yes":
                            continue
                        else:
                            grounding_ans = ["no"]
                    else:
                        grounding_ans = answer.copy()
                    
                    grounding_attr = attr.copy()
                    
                    assert len(grounding_attr) == 1
                    # in case of verifying existence of an attribute in a "specific lead"
                    # (e.g.,) "Does this ECG show symptoms of ... in lead I?"
                    if (
                        lead := re.search(
                            r"((?<= in )lead ((III)|(II)|(I)|(aVR)|(aVL)|(aVF)|(V1)|(V2)|(V3)|(V4)|(V5)|(V6)))"
                            + r"|((?<= in )((limb leads)|(chest leads)))",
                            question
                        )
                    ) is not None:
                        lead = lead.group()
                        grounding_obj = [lead]
                    # in case of verifying existence of an attribute associated with "a group of leads"
                    # (e.g.,) "Does this ECG show symptoms of ST-T changes in anterior leads?"
                    elif (
                        lead := re.search(
                            r"(?<=" + grounding_attr[0].replace("(", "\(").replace(")", "\)").replace("-", "\-") + r" in )"
                            r"((anterior leads)|(inferior leads)|(frontal and horizontal leads)"
                            r"|(septal leads)|(lateral leads)|(posterior leads)|(frontal leads)"
                            r"|(horizontal leads)|(diffuse leads))",
                            question
                        )
                    ) is not None:
                        lead = lead.group()
                        if lead in ["frontal and horizontal leads", "diffuse leads"]:
                            lead = "entire"
                        grounding_obj = [lead]
                    # in case of verifying existence of an attribute from the "entire" ecg,
                    # but possibility of having lead positions for attributes that inherently
                    # have lead positions in themselves (e.g.,) MI in anterior leads
                    else:
                        if " the normal range of " not in question:
                            grounding_obj = [parse_lead_position(attr[0])]
                        else:
                            grounding_obj = ["entire"]
            elif question_type2 == "choose":
                # we do not convert choose / query questions in test split
                if subset == "test":
                    continue

                if "excluding uncertain symptoms" in question:
                    grounding_attr = list(set(answer) - {"none"})
                    grounding_ans = ["yes"] * len(grounding_attr)
                elif "including uncertain symptoms" in question:
                    grounding_attr = list(set(attr) - set(answer))
                    grounding_ans = ["no"] * len(grounding_attr)
                else:
                    grounding_attr = attr.copy()
                    if " the normal range" in question:
                        assert len(answer) == 1
                        grounding_ans = ["yes" if x[:5] == answer[0][:5] else "no" for x in grounding_attr]
                    else:
                        grounding_ans = ["yes" if x in answer else "no" for x in grounding_attr]

                # in case of choosing an attribute in a "specific lead"
                # (e.g.,) "Which noises does this ECG show in lead ..., A or B?"
                # note that we are asssuming there is no question of choosing a specific "lead"
                if (
                    lead := re.search(
                        r"((?<= in )lead ((III)|(II)|(I)|(aVR)|(aVL)|(aVF)|(V1)|(V2)|(V3)|(V4)|(V5)|(V6)))"
                        + r"|((?<= in )((limb leads)|(chest leads)))",
                        question
                    )
                ) is not None:
                    lead = lead.group()
                    grounding_obj = [lead] * len(grounding_attr)
                else:
                    grounding_obj = []
                    for j, attribute in enumerate(grounding_attr):
                        # in case of choosing an attribute from the "entire" ecg,
                        # but possibility of having lead positions for attributes that inherently
                        # have lead positions in themselves (e.g.,) MI in anterior leads
                        grounding_obj.append(parse_lead_position(attribute))
            elif question_type2 == "query":
                # we do not convert choose / query questions in test split
                if subset == "test":
                    continue
                
                # in case of querying attributes in a "specific lead"
                # (e.g.,) "What form-related symptoms does this ECG show in ${lead}?"
                if (
                    lead := re.search(
                        r"((?<= in )lead ((III)|(II)|(I)|(aVR)|(aVL)|(aVF)|(V1)|(V2)|(V3)|(V4)|(V5)|(V6)))"
                        + r"|((?<= in )((limb leads)|(chest leads)))",
                        question
                    )
                ) is not None:
                    lead = lead.group()
                    grounding_ans = ["yes" if x in answer else "no" for x in attr]
                    grounding_attr = attr.copy()
                    grounding_obj = [lead] * len(attr)
                
                # in case of querying leads of a specific attribute
                # (e.g.,) "What leads are showing symptoms of ${scp_codes_form_with_leads} in this ECG?"
                elif "What leads" in question:
                    if attr[0] in [
                        "voltage criteria (qrs) for left ventricular hypertrophy", # VCLVH
                        "low qrs voltages in the frontal and horizontal leads", # LVOLT
                        "high qrs voltage", # HVOLT
                    ]:
                        grounding_ans = []
                        grounding_obj = []
                        for lead_group, leads in lead_groups.items():
                            grounding_obj.append(lead_group)
                            if set(leads).issubset(set(answer)):
                                grounding_ans.append("yes")
                            else:
                                grounding_ans.append("no")
                    else:
                        grounding_ans = ["yes" if x in answer else "no" for x in lead_names]
                        grounding_obj = lead_names.copy()
                    grounding_attr = attr.copy() * len(grounding_ans)
                else:
                    if "excluding uncertain symptoms" in question:
                        grounding_attr = list(set(answer) - {"none"})
                        grounding_ans = ["yes"] * len(grounding_attr)
                    elif "including uncertain symptoms" in question:
                        grounding_attr = list(set(attr) - set(answer))
                        grounding_ans = ["no"] * len(grounding_attr)
                    else:
                        grounding_attr = attr.copy()
                        
                        # we do not use this case since unknown / none of MI is not inquired from "verify"
                        if answer[0] in [
                            "unknown stage of myocardial infarction",
                            "none of myocardial infarction"
                        ]:
                            continue
                        # same with the above
                        if "unknown stage of myocardial infarction" in grounding_attr:
                            grounding_attr.remove("unknown stage of myocardial infarction")
                        if "none of myocardial infarction" in grounding_attr:
                            grounding_attr.remove("none of myocardial infarction")


                        if "What range " in question:
                            grounding_ans = ["yes" if x[:5] == answer[0][:5] else "no" for x in grounding_attr]
                        elif "What numeric features " in question:
                            grounding_ans = ["yes" if x[x.index(" of ") + 4:] in answer else "no" for x in grounding_attr]
                        else:
                            grounding_ans = ["yes" if x in answer else "no" for x in grounding_attr]
                
                    grounding_obj = []
                    for attribute in grounding_attr:
                        # in case of querying attributes from the "entire" ecg
                        # but possibility of having lead positions for attributes that inherently
                        # have lead positions in themselves (e.g.,) MI in anterior leads
                        grounding_obj.append(parse_lead_position(attribute))
            else:
                raise ValueError(question_type2)

            for i in range(len(grounding_attr)):
                # for ptb-xl version
                if "myocardial infarction in" in grounding_attr[i]:
                    grounding_attr[i] = re.sub(
                        r"myocardial infarction in.* leads", "myocardial infarction", grounding_attr[i]
                    )
                elif "subendocardial injury in" in grounding_attr[i]:
                    grounding_attr[i] = re.sub(
                        r"subendocardial injury in.* leads", "subendocardial injury", grounding_attr[i]
                    )
                elif "ischemic in" in grounding_attr[i]:
                    grounding_attr[i] = re.sub(
                        r"ischemic in.* leads", "ischemic", grounding_attr[i]
                    )
                # for mimic-iv-ecg version
                elif "Myocardial infarction in " in grounding_attr[i]:
                    grounding_attr[i] = re.sub(
                        r"Myocardial infarction in.* leads", "myocardial infarction", grounding_attr[i]
                    )
                elif "ischemic ST-T changes in " in grounding_attr[i]:
                    grounding_attr[i] = re.sub(
                        r"ischemic ST-T changes in.* leads", "ischemic ST-T changes", grounding_attr[i]
                    )

            label = [1 if x == "yes" else 0 for x in grounding_ans]
            grounding_samples.append({
                "ecg_id": [ecg_id] * len(grounding_ans),
                "ecg_path": [ecg_path] * len(grounding_ans),
                "attr": grounding_attr.copy(),
                "obj": grounding_obj.copy(),
                "labels": label.copy()
            })
        
        # linearize grounding samples
        grounding_tuples = linearize_grounding_dict(grounding_samples)
        for ecg_id, ecg_path, attr, obj, label in tqdm(grounding_tuples, total=len(grounding_tuples)):
            grounding_sample = get_grounding_sample(
                ecg_id=ecg_id,
                ecg_path=ecg_path,
                attr=attr,
                obj=obj,
                label=label,
            )
            grounding_data[subset].append(grounding_sample)
        duplicates = [
            item for item, count in collections.Counter(
                [(x["ecg_id"], x["obj"], x["attribute"]) for x in grounding_data[subset]]
            ).items() if count > 1
        ]
        # only myocardial infarction is allowed to be duplicated since there are two concepts
        # for myocardial infarction, followed as:
        # 1: "Does this ECG show myocardial infarction?". This kind of questions does not consider
        # likelihood.
        # 2: "Does this ECG show myocardial infarction in inferior leads?". This kind of questions
        # considers likelihood, but shares the attribute 'myocardial infarction'.
        # Hence, if the same ECG that has MI with likelihood 50 is sampled from the both questions,
        # then the answer would be "yes" for the above one, but "no" for the below one, which
        # makes duplicates.
        # we choose "no" for this case when it occurred.
        # (added for MIMIC-IV-ECG version) applied to "ischemic ST-T changes" due to the same reason.
        allowed_duplicates = [item for item in duplicates if item[2] in ["myocardial infarction", "Myocardial infarction", "ischemic ST-T changes"]]
        for item in grounding_data[subset].copy():
            if (item["ecg_id"], item["obj"], item["attribute"]) in allowed_duplicates:
                if item["label"] == 1:
                    grounding_data[subset].remove(item)

        #XXX check the uniqueness of 3-tuple (ecg_id, object, attribute)
        tuples_ = [(x["ecg_id"], x["obj"],x["attribute"]) for x in grounding_data[subset]]

        if len(tuples_) == len(set(tuples_)):
            print("Pass")
        else:
            dicts = {}
            for t in tuples_:
                if t not in dicts:
                    dicts[t] = 0
                dicts[t] += 1
            dups = [k for k, v in dicts.items() if v > 1]
            print(len(dups))

            k = [x for x in grounding_data[subset] if x["ecg_id"] == dups[0][0] and x["obj"] == dups[0][1] and x["attribute"] == dups[0][2]]
            breakpoint()

    grounding_classes = list(set(x["attribute"] for x in grounding_data["test"]))
    grounding_classes.sort()

    pd.DataFrame({"class":grounding_classes}).reset_index().to_csv(
        os.path.join(dest_path, "grounding_class.csv"), index=False
    )

    for subset in grounding_data:
        grounding_data[subset] = [
            s for s in grounding_data[subset] if s["attribute"] in grounding_classes
        ]
        for s in grounding_data[subset]:
            s["target_idx"] = grounding_classes.index(s["attribute"])
            s["attribute_id"] = s["target_idx"]
    
        classify_grounding_data = {}
        
        for i, sample in tqdm(enumerate(grounding_data[subset]), total=len(grounding_data[subset])):
            ecg_id = sample["ecg_id"]
            ecg_path = sample["ecg_path"]
            obj = sample["obj"]
            target_idx = sample["target_idx"]
            label = sample["label"]
            attribute_id = sample["attribute_id"]
            
            grounding_obj = lead_positions[obj]
            if (ecg_id, grounding_obj) not in classify_grounding_data:
                classify_grounding_data[(ecg_id, grounding_obj)] = []
            classify_grounding_data[(ecg_id, grounding_obj)].append(
                (target_idx, label, ecg_path, attribute_id)
            )
        
        with open(os.path.join(dest_path, subset + ".tsv"), "w") as f:
            print(os.path.join(dest_path, subset), file=f)
            for (ecg_id, lead), items in tqdm(
                classify_grounding_data.items(), total=len(classify_grounding_data)
            ):
                lead = list(lead)
                lead.sort()
                target_idx = [x[0] for x in items]
                label = [x[1] for x in items]
                ecg_path = items[0][2]
                attribute_id = [x[3] for x in items]
                
                ecg, _ = wfdb.rdsamp(ecg_path)
                size = len(ecg)

                #XXX
                if np.isnan(ecg).any():
                    print()
                    print("nan detected")
                    print()
                    breakpoint()

                output = {
                    "ecg_path": ecg_path,
                    "lead": lead,
                    "target_idx": target_idx,
                    "label": label,
                    "attribute_id": attribute_id
                }
                
                postfix = "_entire" if len(lead) == 12 else "_" + "_".join(map(str, lead))
                scipy.io.savemat(
                    os.path.join(
                        dest_path, subset, str(ecg_id) + postfix + ".mat"
                    ), output
                )
                print(str(ecg_id) + postfix + ".mat", file=f, end="\t")
                print(size, file=f)

def parse_lead_position(str):
    matched_lead_position = re.search(r"(?<=in ).* leads", str)
    if matched_lead_position is not None:
        matched_lead_position = matched_lead_position.group()
        if "frontal and horizontal" in matched_lead_position:
            return "entire"
        else:
            return matched_lead_position
    return "entire"

def linearize_grounding_dict(ls: List[Dict[str, list]]):
    key = list(ls[0].keys())
    assert key == ["ecg_id", "ecg_path", "attr", "obj", "labels"]

    for x in ls:
        if list(x.keys()) != key:
            raise ValueError("dictionarys in the list should have the same key one another")

    unique_tuples = [
        (ecg_id, ecg_path, attr, obj, label)
        for x in ls for ecg_id, ecg_path, attr, obj, label in zip(
            x["ecg_id"], x["ecg_path"], x["attr"], x["obj"], x["labels"]
        )
    ]
    unique_tuples = list(set(unique_tuples))

    return unique_tuples

def get_grounding_sample(
    ecg_id,
    ecg_path,
    attr,
    obj,
    label,
):
    return {
        "ecg_id": ecg_id,
        "ecg_path": ecg_path,
        "obj": obj,
        "attribute": attr,
        "label": label
    }

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
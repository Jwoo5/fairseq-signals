import os
import shutil

import wfdb
import scipy.io
import pandas as pd

def manifest(data, grounding_data, dest):
    dest_dir = os.path.realpath(dest)
    lead_positions = {
        # "entire": [0,1,2,3,4,5,6,7,8,9,10,11],
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
        "inferior leads": (1, 2, 5,),
        "inferolateral leads": (0, 1, 2, 4, 5, 10, 11,),
        "inferoposterior leads": (1, 2, 5, 6, 7, 8,),
        "inferoposterolateral leads": (0, 1, 2, 4, 5, 6, 7, 8, 10, 11,),
        "anterior leads": (8, 9,),
        "anteroseptal leads": (6, 7, 8, 9,),
        "anterolateral leads": (0, 4, 8, 9, 10, 11,),
        "lateral leads": (0, 4, 10, 11,),
        "posterior leads": (6, 7, 8,),
    }

    for split in data:
        # remove and re-create destination data directories
        shutil.rmtree(os.path.join(dest, "qa", split), ignore_errors=True)
        os.makedirs(os.path.join(dest, "qa", split))
        
        with open(os.path.join(dest, "qa", split +'.tsv'), 'w') as f:
            print(os.path.join(dest_dir, "qa", split), file=f)
            print(data[split]['tokenized'], file=f)
            print(data[split]['num_labels'], file=f)

            for i, sample in enumerate(data[split]['samples']):
                ecg, _ = wfdb.rdsamp(sample['ecg_path'])
                ecg_sz = len(ecg)
                text_sz = len(sample['question'])
                scipy.io.savemat(os.path.join(dest_dir, "qa", split, str(i) + '.mat'), sample)
                print(str(i) + '.mat', file=f, end='\t')
                print(ecg_sz, file=f, end='\t')
                print(text_sz, file=f)

    for split in grounding_data:
        classify_per_lead_grounding_data = {}
        classify_entire_grounding_data = {}

        # we do not need grounding qa dataset for training / validation
        if split == "test":
            shutil.rmtree(os.path.join(dest, "qa", split + "_per_lead_grounding"), ignore_errors=True)
            os.makedirs(os.path.join(dest, "qa", split + "_per_lead_grounding"))
            shutil.rmtree(os.path.join(dest, "qa", split + "_entire_grounding"), ignore_errors=True)
            os.makedirs(os.path.join(dest, "qa", split + "_entire_grounding"))

        shutil.rmtree(os.path.join(dest, "classify", split + "_per_lead_grounding"), ignore_errors=True)
        os.makedirs(os.path.join(dest, "classify", split + "_per_lead_grounding"))
        shutil.rmtree(os.path.join(dest, "classify", split + "_entire_grounding"), ignore_errors=True)
        os.makedirs(os.path.join(dest, "classify", split + "_entire_grounding"))

        if split == "test":
            grounding_classes = list(set(x["attribute"] for x in grounding_data[split]))
            grounding_classes.sort()
            pd.DataFrame(
                {'index': i, 'class': c} for i, c in enumerate(grounding_classes)
            ).to_csv(os.path.join('results', 'grounding_class.csv'), index=False)

            qa_entire_dest = open(os.path.join(dest, "qa", split + "_entire_grounding.tsv"), "w")
            qa_per_lead_dest = open(os.path.join(dest, "qa", split + "_per_lead_grounding.tsv"), "w")
            print(os.path.join(dest_dir, "qa", split + "_entire_grounding"), file=qa_entire_dest)
            print(os.path.join(dest_dir, "qa", split + "_per_lead_grounding"), file=qa_per_lead_dest)
            print(data[split]["tokenized"], file=qa_entire_dest)
            print(data[split]["tokenized"], file=qa_per_lead_dest)
            print(data[split]["num_labels"], file=qa_entire_dest)
            print(data[split]["num_labels"], file=qa_per_lead_dest)

        for i, sample in enumerate(grounding_data[split]):
            ecg_id = sample["ecg_id"]
            qid = sample["question_id"]
            ecg_path = sample["ecg_path"]
            target_idx = grounding_classes.index(sample["attribute"])
            label = sample["answer_bin"]
            obj = sample["obj"]

            if split == "test":
                ecg, _ = wfdb.rdsamp(ecg_path)
                ecg_sz = len(ecg)
                text_sz = len(sample["question"])

                if obj.startswith("lead"):
                    scipy.io.savemat(
                        os.path.join(
                            dest_dir, "qa", split + "_per_lead_grounding", str(i) + ".mat"
                        ), sample
                    )
                    print(str(i) + ".mat", file=qa_per_lead_dest, end="\t")
                    print(ecg_sz, file=qa_per_lead_dest, end="\t")
                    print(text_sz, file=qa_per_lead_dest)
                elif obj == "entire":
                    scipy.io.savemat(
                        os.path.join(
                            dest_dir, "qa", split + "_entire_grounding", str(i) + ".mat"
                        ), sample
                    )
                    print(str(i) + ".mat", file=qa_entire_dest, end="\t")
                    print(ecg_sz, file=qa_entire_dest, end="\t")
                    print(text_sz, file=qa_entire_dest)

            if obj == "entire":
                if ecg_id not in classify_entire_grounding_data:
                    classify_entire_grounding_data[ecg_id] = []
                classify_entire_grounding_data[ecg_id].append((target_idx, label, qid, ecg_path))
            else:
                grounding_obj = lead_positions[obj]
                if ecg_id not in classify_per_lead_grounding_data:
                    classify_per_lead_grounding_data[(ecg_id, grounding_obj)] = []
                classify_per_lead_grounding_data[(ecg_id, grounding_obj)].append((target_idx, label, qid, ecg_path))

        if split == "test":
            qa_entire_dest.close()
            qa_per_lead_dest.close()

        total_f = open(os.path.join(dest, "classify", split + "_total.tsv"), "w")
        print(os.path.join(dest_dir, "classify"), file=total_f)
        with open(os.path.join(dest, "classify", split + "_entire_grounding.tsv"), "w") as f:
            print(os.path.join(dest_dir, "classify", split + "_entire_grounding"), file=f)
            for ecg_id, items in classify_entire_grounding_data.items():
                if len(items) == 0:
                    continue
                target_idx = [x[0] for x in items]
                label = [x[1] for x in items]
                qid = [x[2] for x in items]
                ecg_path = items[0][3]

                ecg, _ = wfdb.rdsamp(ecg_path)
                size = len(ecg)
                
                output = {
                    "ecg_path": ecg_path,
                    "target_idx": target_idx,
                    "label": label,
                    "question_id": qid,
                }
                scipy.io.savemat(
                    os.path.join(
                        dest_dir, "classify", split + "_entire_grounding", str(ecg_id) + ".mat"
                    ), output
                )
                print(str(ecg_id) + ".mat", file=f, end="\t")
                print(os.path.join(split + "_entire_grounding", str(ecg_id) + ".mat"), file=total_f, end="\t")
                print(size, file=f)
                print(size, file=total_f)

        with open(os.path.join(dest, "classify", split + "_per_lead_grounding.tsv"), "w") as f:
            print(os.path.join(dest_dir, "classify", split + "_per_lead_grounding"), file=f)
            for (ecg_id, lead), items in classify_per_lead_grounding_data.items():
                lead = list(lead)
                lead.sort()
                target_idx = [x[0] for x in items]
                label = [x[1] for x in items]
                qid = [x[2] for x in items]
                ecg_path = items[0][3]

                ecg, _ = wfdb.rdsamp(ecg_path)
                size = len(ecg)

                output = {
                    "ecg_path": ecg_path,
                    "lead": lead,
                    "target_idx": target_idx,
                    "label": label,
                    "question_id": qid,
                }
                postfix = "_".join(map(str, lead))
                scipy.io.savemat(
                    os.path.join(
                        dest_dir, "classify", split + "_per_lead_grounding", str(ecg_id) + postfix + ".mat"
                    ), output
                )
                print(str(ecg_id) + postfix + ".mat", file=f, end="\t")
                print(os.path.join(split + "_per_lead_grounding", str(ecg_id) + postfix + ".mat"), file=total_f, end="\t")
                print(size, file=f)
                print(size, file=total_f)
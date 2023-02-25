import os
import shutil

import wfdb
import scipy.io

from tqdm import tqdm

def manifest(data, derived_grounding_data, independent_grounding_data, dest):
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
        "limb leads": (0, 1, 2, 3, 4, 5),
        "chest leads": (6, 7, 8, 9, 10, 11),
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
        print(f"[{split}] Prepare manifest for original QA data")
        # remove and re-create destination data directories
        shutil.rmtree(os.path.join(dest, "qa", split), ignore_errors=True)
        os.makedirs(os.path.join(dest, "qa", split))

        with open(os.path.join(dest, "qa", split +'.tsv'), 'w') as f:
            print(os.path.join(dest_dir, "qa", split), file=f)
            print(data[split]['tokenized'], file=f)
            print(data[split]['num_labels'], file=f)

            for i, sample in tqdm(enumerate(data[split]['samples']), total=len(data[split]["samples"])):
                ecg, _ = wfdb.rdsamp(sample['ecg_path'])
                ecg_sz = len(ecg)
                text_sz = len(sample['question'])
                scipy.io.savemat(os.path.join(dest_dir, "qa", split, str(i) + '.mat'), sample)
                print(str(i) + '.mat', file=f, end='\t')
                print(ecg_sz, file=f, end='\t')
                print(text_sz, file=f)

    grounding_modes = ["derived", "independent"]
    grounding_data = [derived_grounding_data, independent_grounding_data]
    # NOTE grounding_data has 'test' as the **FIRST** key
    for mode, g_data in zip(grounding_modes, grounding_data):
        for split in g_data:
            if split == "test":
                print(f"[{split}] Prepare manifest for {mode} grounding QA data")

            classify_per_lead_grounding_data = {}
            classify_entire_grounding_data = {}

            # we do not need grounding qa dataset for training / validation
            if split == "test":
                shutil.rmtree(os.path.join(dest, "qa", split + "_" + mode + "_per_lead_grounding"), ignore_errors=True)
                os.makedirs(os.path.join(dest, "qa", split + "_" + mode + "_per_lead_grounding"))
                shutil.rmtree(os.path.join(dest, "qa", split + "_" + mode + "_entire_grounding"), ignore_errors=True)
                os.makedirs(os.path.join(dest, "qa", split + "_" + mode + "_entire_grounding"))

            shutil.rmtree(os.path.join(dest, "classify", split + "_" + mode + "_per_lead_grounding"), ignore_errors=True)
            os.makedirs(os.path.join(dest, "classify", split + "_" + mode + "_per_lead_grounding"))
            shutil.rmtree(os.path.join(dest, "classify", split + "_" + mode + "_entire_grounding"), ignore_errors=True)
            os.makedirs(os.path.join(dest, "classify", split + "_" + mode + "_entire_grounding"))

            if split == "test":
                qa_entire_dest = open(os.path.join(dest, "qa", split + "_" + mode + "_entire_grounding.tsv"), "w")
                qa_per_lead_dest = open(os.path.join(dest, "qa", split + "_" + mode + "_per_lead_grounding.tsv"), "w")
                print(os.path.join(dest_dir, "qa", split + "_" + mode + "_entire_grounding"), file=qa_entire_dest)
                print(os.path.join(dest_dir, "qa", split + "_" + mode + "_per_lead_grounding"), file=qa_per_lead_dest)
                print(data[split]["tokenized"], file=qa_entire_dest)
                print(data[split]["tokenized"], file=qa_per_lead_dest)
                print(data[split]["num_labels"], file=qa_entire_dest)
                print(data[split]["num_labels"], file=qa_per_lead_dest)

            for i, sample in tqdm(enumerate(g_data[split]), total=len(g_data[split])):
                ecg_id = sample["ecg_id"]
                attr_id = sample["attribute_id"]
                qid = sample["question_id"]
                ecg_path = sample["ecg_path"]
                target_idx = sample["target_idx"]
                label = sample["answer_bin"]
                obj = sample["obj"]

                if split == "test":
                    ecg, _ = wfdb.rdsamp(ecg_path)
                    ecg_sz = len(ecg)
                    text_sz = len(sample["question"])

                    if obj == "entire":
                        scipy.io.savemat(
                            os.path.join(
                                dest_dir, "qa", split + "_" + mode + "_entire_grounding", str(i) + ".mat"
                            ), sample
                        )
                        print(str(i) + ".mat", file=qa_entire_dest, end="\t")
                        print(ecg_sz, file=qa_entire_dest, end="\t")
                        print(text_sz, file=qa_entire_dest)
                    else:
                        scipy.io.savemat(
                            os.path.join(
                                dest_dir, "qa", split + "_" + mode + "_per_lead_grounding", str(i) + ".mat"
                            ), sample
                        )
                        print(str(i) + ".mat", file=qa_per_lead_dest, end="\t")
                        print(ecg_sz, file=qa_per_lead_dest, end="\t")
                        print(text_sz, file=qa_per_lead_dest)

                if obj == "entire":
                    if ecg_id not in classify_entire_grounding_data:
                        classify_entire_grounding_data[ecg_id] = []
                    classify_entire_grounding_data[ecg_id].append(
                        (target_idx, label, attr_id, qid, ecg_path)
                    )
                else:
                    grounding_obj = lead_positions[obj]
                    if (ecg_id, grounding_obj) not in classify_per_lead_grounding_data:
                        classify_per_lead_grounding_data[(ecg_id, grounding_obj)] = []

                    classify_per_lead_grounding_data[(ecg_id, grounding_obj)].append(
                        (target_idx, label, attr_id, qid, ecg_path)
                    )

            if split == "test":
                qa_entire_dest.close()
                qa_per_lead_dest.close()

            with open(os.path.join(dest, "classify", split + "_" + mode + "_entire_grounding.tsv"), "w") as f:
                print(f"[{split}] Prepare manifest for {mode} entire grounding classifier data")
                print(os.path.join(dest_dir, "classify", split + "_" + mode + "_entire_grounding"), file=f)
                for ecg_id, items in tqdm(
                    classify_entire_grounding_data.items(), total=len(classify_entire_grounding_data)
                ):
                    if len(items) == 0:
                        continue
                    target_idx = [x[0] for x in items]
                    label = [x[1] for x in items]
                    attr_id = [x[2] for x in items]
                    qid = [x[3] for x in items]
                    ecg_path = items[0][4]

                    ecg, _ = wfdb.rdsamp(ecg_path)
                    size = len(ecg)
                    
                    output = {
                        "ecg_path": ecg_path,
                        "target_idx": target_idx,
                        "label": label,
                        "attribute_id": attr_id,
                        "question_id": qid,
                    }
                    scipy.io.savemat(
                        os.path.join(
                            dest_dir, "classify", split + "_" + mode + "_entire_grounding", str(ecg_id) + ".mat"
                        ), output
                    )
                    print(str(ecg_id) + ".mat", file=f, end="\t")
                    print(size, file=f)

            with open(os.path.join(dest, "classify", split + "_" + mode + "_per_lead_grounding.tsv"), "w") as f:
                print(f"[{split}] Prepare manifest for {mode} per-lead grounding classifier data")
                print(os.path.join(dest_dir, "classify", split + "_" + mode + "_per_lead_grounding"), file=f)
                for (ecg_id, lead), items in tqdm(
                    classify_per_lead_grounding_data.items(), total=len(classify_per_lead_grounding_data)
                ):
                    lead = list(lead)
                    lead.sort()
                    target_idx = [x[0] for x in items]
                    label = [x[1] for x in items]
                    attr_id = [x[2] for x in items]
                    qid = [x[3] for x in items]
                    ecg_path = items[0][4]

                    ecg, _ = wfdb.rdsamp(ecg_path)
                    size = len(ecg)

                    output = {
                        "ecg_path": ecg_path,
                        "lead": lead,
                        "target_idx": target_idx,
                        "label": label,
                        "attribute_id": attr_id,
                        "question_id": qid,
                    }
                    postfix = "_" + "_".join(map(str, lead))
                    scipy.io.savemat(
                        os.path.join(
                            dest_dir, "classify", split + "_" + mode + "_per_lead_grounding", str(ecg_id) + postfix + ".mat"
                        ), output
                    )
                    print(str(ecg_id) + postfix + ".mat", file=f, end="\t")
                    print(size, file=f)
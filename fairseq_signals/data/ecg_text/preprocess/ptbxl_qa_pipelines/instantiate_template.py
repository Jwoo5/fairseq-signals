import pickle
import os
import sys
import re
import random
import json
from itertools import combinations
import numpy as np
import pandas as pd

from typing import List, Dict
from tqdm import tqdm

def instantiate_template(
    ptbxl_dir,
    ptbxl_data_dir,
    template_dir,
    encoded_ptbxl,
    valid_percent=0.1,
    test_percent=0.1,
    seed=42,
    cnt_threshold=1,
    tokenize=True,
    answer_encode='multi-label',
):
    pd.set_option('mode.chained_assignment', None)

    assert valid_percent + test_percent <= 1.0, (
        '`--valid_percent` + `--test_percent` shuold not exceed 1.0'
    )
    assert answer_encode in ['multi-label', 'text'], (
        f'invalid --answer_encode: {answer_encode}. '
        'answer_encode should be one of {}'.format(['multi-label', 'text'])
    )
    train_percent = 1 - (valid_percent + test_percent)

    scp_statements = pd.read_csv(os.path.join(ptbxl_dir, 'scp_statements.csv'))

    scp_codes_subclass = scp_statements['Unnamed: 0'].to_list()
    scp_codes_subclass.remove('NORM')

    scp_codes_diagnostic = scp_statements[scp_statements['diagnostic'] == 1]['Unnamed: 0'].to_list()
    scp_codes_diagnostic.remove('NORM')

    scp_codes_superclass = scp_statements[scp_statements['diagnostic'] == 1]['diagnostic_class'].unique().tolist()
    scp_codes_superclass.remove('NORM')

    scp_codes_form = scp_statements[scp_statements['form'] == 1]['Unnamed: 0'].to_list()

    scp_codes_form_with_leads = ['NDT', 'NST_', 'DIG', 'STD_', 'VCLVH', 'QWAVE', 'LOWT', 'NT_', 'INVT', 'LVOLT', 'HVOLT', 'TAB_', 'STE_']

    scp_codes_rhythm = scp_statements[scp_statements['rhythm'] == 1]['Unnamed: 0'].to_list()

    subcategory = dict()

    subcategory['scp_codes_norm'] = ['NORM']
    subcategory['scp_codes_subclass'] = scp_codes_subclass
    subcategory['any_scp_codes_subclass'] = ['ANY_SCP_CODES']
    subcategory['scp_codes_superclass'] = scp_codes_superclass
    subcategory['scp_codes_diagnostic_inc'] = scp_codes_diagnostic
    subcategory['any_scp_codes_diagnostic_inc'] = ['ANY_DX_INC']
    subcategory['scp_codes_diagnostic_exc'] = scp_codes_diagnostic
    subcategory['any_scp_codes_diagnostic_exc'] = ['ANY_DX_EXC']
    subcategory['scp_codes_form'] = scp_codes_form
    subcategory['scp_codes_form_with_leads'] = scp_codes_form_with_leads
    subcategory['any_scp_codes_form'] = ['ANY_FORM']
    subcategory['scp_codes_rhythm'] = scp_codes_rhythm
    subcategory['any_scp_codes_rhythm'] = ['ANY_RHYTHM']

    subcategory['heart_axis_direction'] = [
        'HEART_AXIS_LEFT',
        'HEART_AXIS_RIGHT',
        'HEART_AXIS_EXTR',
        'HEART_AXIS_NORM'
    ]

    subcategory['infarction_stadium'] = ['MI_STAGE1', 'MI_STAGE2', 'MI_STAGE3']
    subcategory['infarction_stadium_ret'] = ['MI_NAN', 'MI_UNK', 'MI_STAGE1', 'MI_STAGE2', 'MI_STAGE3']

    subcategory['noise'] = ['DRIFT', 'STATIC', 'BURST', 'ELECT']
    subcategory['noise_with_leads'] = ['DRIFT', 'STATIC', 'BURST', 'ELECT']
    subcategory['any_noise'] = ['ANY_NOISE']

    subcategory['extra_systole'] = ['ES_COUNT', 'VES_COUNT', 'SVES_COUNT']
    subcategory['any_extra_systole'] = ['ANY_EXTRA']
    # subcategory['extra_systole_with_leads'] = ['ES', 'VES', 'SVES']
    # subcategory['extra_systole_count'] = ['ES_COUNT', 'VES_COUNT', 'SVES_COUNT']

    numeric_features = [
        "rr_interval",
        "p_duration",
        "pr_interval",
        "qrs_duration",
        "qt_interval",
        "qt_corrected"
    ]
    for nf in numeric_features:
        subcategory["max_" + nf] = [
            "MAX_" + nf.upper() + "_LOW",
            "MAX_" + nf.upper() + "_NORM",
            "MAX_" + nf.upper() + "_HIGH"
        ]
        subcategory["min_" + nf] = [
            "MIN_" + nf.upper() + "_LOW",
            "MIN_" + nf.upper() + "_NORM",
            "MIN_" + nf.upper() + "_HIGH"
        ]

    category_to_name = {
        "NORM": "normal ecg",
        "CD": "conduction disturbance",
        "HYP": "hypertrophy",
        "MI": "myocardial infarction",
        "STTC": "st/t change",
        "LAFB": "left anterior fascicular block",
        "LPFB": "left posterior fascicular block",
        "1AVB": "first degree av block",
        "2AVB": "second degree av block",
        "3AVB": "third degree av block",
        "IVCD": "non-specific intraventricular conduction disturbance (block)",
        "WPW": "wolff-parkinson-white syndrome",
        "IRBBB": "incomplete right bundle branch block",
        "ILBBB": "incomplete left bundle branch block",
        "CRBBB": "complete right bundle branch block",
        "CLBBB": "complete left bundle branch block",
        "LAO/LAE": "left atrial overload/enlargement",
        "RAO/RAE": "right atrial overload/enlargement",
        "LVH": "left ventricular hypertrophy",
        "RVH": "right ventricular hypertrophy",
        "SEHYP": "septal hypertrophy",
        "IMI": "myocardial infarction in inferior leads",
        "ILMI": "myocardial infarction in inferolateral leads",
        "IPMI": "myocardial infarction in inferoposterior leads",
        "IPLMI": "myocardial infarction in inferoposterolateral leads",
        "INJIN": "subendocardial injury in inferior leads",
        "INJIL": "subendocardial injury in inferolateral leads",
        "AMI": "myocardial infarction in anterior leads",
        "ASMI": "myocardial infarction in anteroseptal leads",
        "ALMI": "myocardial infarction in anterolateral leads",
        "INJAS": "subendocardial injury in anteroseptal leads",
        "INJAL": "subendocardial injury in anterolateral leads",
        "INJLA": "subendocardial injury in lateral leads",
        "LMI": "myocardial infarction in lateral leads",
        "PMI": "myocardial infarction in posterior leads",
        "NDT": "non-diagnostic t abnormalities",
        "DIG": "digitalis effect",
        "LNGQT": "long qt-interval",
        "ANEUR": "st-t changes compatible with ventricular aneurysm",
        "EL": "electrolytic disturbance or drug (former EDIS)",
        "NST_": "non-specific st changes",
        "ISC_": "non-specific ischemic",
        "ISCAL": "ischemic in anterolateral leads",
        "ISCAS": "ischemic in anteroseptal leads",
        "ISCLA": "ischemic in lateral leads",
        "ISCAN": "ischemic in anterior leads",
        "ISCIN": "ischemic in inferior leads",
        "ISCIL": "ischemic in inferolateral leads",
        "ABQRS": "abnormal qrs",
        "PVC": "ventricular premature complex",
        "STD_": "non-specific st depression",
        "VCLVH": "voltage criteria (qrs) for left ventricular hypertrophy",
        "QWAVE": "q waves present",
        "LOWT": "low amplitude t-wave",
        "NT_": "non-specific t-wave changes",
        "PAC": "atrial premature complex",
        "LPR": "prolonged pr interval",
        "INVT": "inverted t-waves",
        "LVOLT": "low qrs voltages in the frontal and horizontal leads",
        "HVOLT": "high qrs voltage",
        "TAB_": "t-wave abnormality",
        "STE_": "non-specific st elevation",
        "PRC(S)": "premature complex(es)",
        "SR": "sinus rhythm",
        "AFIB": "atrial fibrillation",
        "AFLT": "atrial flutter",
        "STACH": "sinus tachycardia",
        "SVTAC": "supraventricular tachycardia",
        "PSVT": "paroxysmal supraventricular tachycardia",
        "SBRAD": "sinus bradycardia",
        "SARRH": "sinus arrhythmia",
        "SVARR": "supraventricular arrhythmia",
        "PACE": "normal functioning artificial pacemaker",
        "BIGU": "bigeminal pattern (unknown origin, supraventricular, or ventricular)",
        "TRIGU": "trigeminal pattern (unknown origin, supraventricular, or ventricular)",
        "ANY_SCP_CODES": "any kind of abnormal symptoms",
        "ANY_DX_INC": "any diagnostic symptoms",
        "ANY_DX_EXC": "any diagnostic symptoms",
        "ANY_FORM": "any form symptoms",
        "ANY_RHYTHM": "any rhythm symptoms",
        'HEART_AXIS_LEFT': 'left axis deviation',
        'HEART_AXIS_RIGHT': 'right axis deviation',
        'HEART_AXIS_EXTR': 'extreme axis deviation',
        'HEART_AXIS_NORM': 'normal axis',
        'MI_STAGE1': 'early stage of myocardial infarction',
        'MI_STAGE2': 'middle stage of myocardial infarction',
        'MI_STAGE3': 'late stage of myocardial infarction',
        'DRIFT': 'baseline drift',
        'STATIC': 'static noise',
        'BURST': 'burst noise',
        'ELECT': 'electrodes problems',
        "ANY_NOISE": "any kind of noises",
        'ES_COUNT': 'extrasystoles',
        'VES_COUNT': 'ventricular extrasystoles',
        'SVES_COUNT': 'supraventricular extrasystoles',
        # 'extra_systole_with_leads': {
        #     'ES': 'extrasystoles',
        #     'VES': 'ventricular extrasystoles',
        #     'SVES': 'supraventricular extrasystoles',
        # },
        "ANY_EXTRA": "any kind of extra systoles",
        "MAX_RR_INTERVAL_LOW": "below the normal range",
        "MAX_RR_INTERVAL_NORM": "within the normal range",
        "MAX_RR_INTERVAL_HIGH": "above the normal range",
        "MIN_RR_INTERVAL_LOW": "below the normal range",
        "MIN_RR_INTERVAL_NORM": "within the normal range",
        "MIN_RR_INTERVAL_HIGH": "above the normal range",
        "MAX_P_DURATION_LOW": "below the normal range",
        "MAX_P_DURATION_NORM": "within the normal range",
        "MAX_P_DURATION_HIGH": "above the normal range",
        "MIN_P_DURATION_LOW": "below the normal range",
        "MIN_P_DURATION_NORM": "within the normal range",
        "MIN_P_DURATION_HIGH": "above the normal range",
        "MAX_PR_INTERVAL_LOW": "below the normal range",
        "MAX_PR_INTERVAL_NORM": "within the normal range",
        "MAX_PR_INTERVAL_HIGH": "above the normal range",
        "MIN_PR_INTERVAL_LOW": "below the normal range",
        "MIN_PR_INTERVAL_NORM": "within the normal range",
        "MIN_PR_INTERVAL_HIGH": "above the normal range",
        "MAX_QRS_DURATION_LOW": "below the normal range",
        "MAX_QRS_DURATION_NORM": "within the normal range",
        "MAX_QRS_DURATION_HIGH": "above the normal range",
        "MIN_QRS_DURATION_LOW": "below the normal range",
        "MIN_QRS_DURATION_NORM": "within the normal range",
        "MIN_QRS_DURATION_HIGH": "above the normal range",
        "MAX_QT_INTERVAL_LOW": "below the normal range",
        "MAX_QT_INTERVAL_NORM": "within the normal range",
        "MAX_QT_INTERVAL_HIGH": "above the normal range",
        "MIN_QT_INTERVAL_LOW": "below the normal range",
        "MIN_QT_INTERVAL_NORM": "within the normal range",
        "MIN_QT_INTERVAL_HIGH": "above the normal range",
        "MAX_QT_CORRECTED_LOW": "below the normal range",
        "MAX_QT_CORRECTED_NORM": "within the normal range",
        "MAX_QT_CORRECTED_HIGH": "above the normal range",
        "MIN_QT_CORRECTED_LOW": "below the normal range",
        "MIN_QT_CORRECTED_NORM": "within the normal range",
        "MIN_QT_CORRECTED_HIGH": "above the normal range",
    }
    # ans_to_name = {
    #     'ES_COUNT': 'extrasystoles',
    #     'VES_COUNT': 'ventricular extrasystoles',
    #     'SVES_COUNT': 'supraventricular extrasystoles'
    # }
    per_lead_attr_scp_code = {
        "non-diagnostic t abnormalities": "NDT",
        "non-specific st changes": "NST_",
        "digitalis effect": "DIG",
        "non-specific st depression": "STD_",
        "voltage criteria (qrs) for left ventricular hypertrophy": "VCLVH",
        "q waves present": "QWAVE",
        "low amplitude t-wave": "LOWT",
        "non-specific t-wave changes": "NT_",
        "inverted t-waves": "INVT",
        "low qrs voltages in the frontal and horizontal leads": "LVOLT",
        "high qrs voltage": "HVOLT",
        "t-wave abnormality": "TAB_",
        "non-specific st elevation": "STE_",
        "myocardial infarction": [
            "IMI", "ILMI", "IPMI", "IPLMI", "AMI", "ASMI", "ALMI", "LMI",
            "PMI", "INJAS", "INJIN", "INJIL", "INJAS", "INJAL", "INJLA"
        ],
        "ischemic": ["ISC_", "ISCAL", "ISCAS", "ISCLA", "ISCAN", "ISCIN", "ISCIL"],
        "subendocardial injury": ["INJIN", "INJIL", "INJAS", "INJAL", "INJLA"]
    }
    per_lead_attr_noise_code = {
        "any kind of noises": "ANY_NOISE",
        "static noise": "STATIC",
        "burst noise": "BURST",
        "baseline drift": "DRIFT",
        "electrodes problems": "ELECT"
    }

    templates = pd.read_csv(os.path.join(template_dir, 'type1QA_templates.csv'))

    random.seed(seed)

    unique_pids = encoded_ptbxl['patient_id'].unique()
    random.shuffle(unique_pids)

    train_pids = unique_pids[:int(len(unique_pids) * train_percent)]
    valid_pids = unique_pids[len(train_pids): len(train_pids) + int(len(unique_pids) * valid_percent)]
    test_pids = unique_pids[len(train_pids) + len(valid_pids):]

    splits = ['train', 'valid', 'test']
    data = {}
    data['train'] = encoded_ptbxl[encoded_ptbxl['patient_id'].isin(train_pids)]
    data['valid'] = encoded_ptbxl[encoded_ptbxl['patient_id'].isin(valid_pids)]
    data['test'] = encoded_ptbxl[encoded_ptbxl['patient_id'].isin(test_pids)]

    if not os.path.exists('results'):
        os.mkdir('results')
    for split in splits:
        ecg_ids = data[split]['ecg_id'].reset_index(drop=True).to_dict()
        ecg_ids_stdout = '\n'.join([str(k) + '\t' + str(v) for k, v in ecg_ids.items()])
        with open(os.path.join('results', split + '_ecgs.tsv'), 'w') as f:
            print(ecg_ids_stdout, file=f)

    lead_names = [
        'lead I', 'lead II', 'lead III', 'lead aVR', 'lead aVL', 'lead aVF',
        'lead V1', 'lead V2', 'lead V3', 'lead V4', 'lead V5', 'lead V6'
    ]
    lead_groups = {
        "limb leads": ["lead I", "lead II", "lead III", "lead aVR", "lead aVL", "lead aVF"],
        "chest leads": ["lead V1", "lead V2", "lead V3", "lead V4", "lead V5", "lead V6"]
    }

    if tokenize:
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    unique_answers = {s: dict() for s in splits}
    sampled_list = {s: [] for s in splits}
    qid_type = dict()
    for split in splits:
        qid = 0
        print(f"Sample {split} examples for each template")
        for i, template in tqdm(templates.iterrows(), total=len(templates)):
            assigned_ecgs = []

            iterate_over_leads = False
            vars = [x.group() for x in re.finditer(r'(?<=\$\{)[a-zA-Z_]+(?=\d?\})', template["template"])]
            if "lead" in vars:
                vars.remove('lead')
                iterate_over_leads = True

            if len(vars) == 0:
                if template['question_type1'] == 'retrieve':
                    candidates = [template['subcategory']]
                else:
                    candidates = subcategory[template['subcategory']]
            elif len(vars) == 1:
                candidates = subcategory[template['subcategory']]
            elif len(vars) == 2:
                candidates = combinations(subcategory[template['subcategory']], 2)

            for candidate in candidates:
                sampled = []
                if iterate_over_leads:
                    if template["question_type1"] == "verify" and candidate in ["LVOLT", "HVOLT", "VCLVH"]:
                        leads = ["limb leads", "chest leads"]
                    else:
                        leads = lead_names
                        # if template["question_type1"] == "retrieve":
                        #     if "LVOLT" in candidate:
                            #     breakpoint()
                        #     if "HVOLT" in candidate:
                            #     breakpoint()
                        #     if "VCLVH" in candidate:
                            #     breakpoint()

                    for l in leads:
                        sampled_, seed = sample(
                            data[split],
                            candidate,
                            template,
                            subcategory=subcategory,
                            category_to_name=category_to_name,
                            # ans_to_name=ans_to_name,
                            ans_to_name=category_to_name,
                            assigned_ecgs=assigned_ecgs,
                            lead=l,
                            split=split,
                            lead_names=lead_names,
                            lead_groups=lead_groups,
                            cnt_threshold=cnt_threshold,
                            n=5 if split == 'train' else 3,
                            question_id=qid,
                            seed=seed,
                        )
                        sampled.append(sampled_)
                else:
                    sampled_, seed = sample(
                        data[split],
                        candidate,
                        template,
                        subcategory=subcategory,
                        category_to_name=category_to_name,
                        # ans_to_name=ans_to_name,
                        ans_to_name=category_to_name,
                        assigned_ecgs=assigned_ecgs,
                        lead=None,
                        split=split,
                        cnt_threshold=cnt_threshold,
                        n=5 if split == 'train' else 3,
                        question_id=qid,
                        seed=seed,
                    )
                    sampled.append(sampled_)

                for s in sampled:
                    if s is not None:
                        sampled_list[split].append(s)
                        unique_answers[split][qid] = set(
                            [frozenset(x) for x in s['sampled_ids'].values()]
                        )
                        assigned_ecgs.extend(set(s['sampled_ids'].keys()))

                    qid_type[qid] = template['question_type1']
                    if qid_type[qid] == 'retrieve':
                        qid_type[qid] = template['question_type2']

                    qid += 1

    # don't use questions that present in vaild / test but not in train
    cnt = 0
    for qid in unique_answers['train']:
        if qid_type[qid] == 'multi-label':
            answer_set_train = set(sum([list(x) for x in unique_answers['train'][qid]],[]))
            answer_set_test = (
                set(sum([list(x) for x in unique_answers['test'][qid]],[]))
            ) if qid in unique_answers['test'] else set()

            answer_set_test_but_not_train = answer_set_test - answer_set_train
            if len(answer_set_test_but_not_train) > 0:
                for split in ['valid', 'test']:
                    for qa_samples in sampled_list[split]:
                        if qa_samples['question_id'] == qid:
                            qa_samples['sampled_ids'] = {
                                ecg_id: ans for ecg_id, ans in qa_samples['sampled_ids'].items()
                                if len(set(ans).intersection(answer_set_test_but_not_train)) == 0
                            }
                            qa_samples['sizes'] = {
                                k: v for k, v in qa_samples['sizes'].items()
                                if k in qa_samples['sampled_ids']
                            }
                            if split == 'test':
                                cnt += 1
                                print("answer exists in test but not train-" + str(cnt) + ': ' + qa_samples['question'])
                                print(answer_set_test_but_not_train)
                            break
        elif qid_type[qid] == 'multi-class':
            answer_set_train = unique_answers['train'][qid]
            answer_set_test = unique_answers['test'][qid] if qid in unique_answers['test'] else set()

            answer_set_test_but_not_train = answer_set_test - answer_set_train
            if len(answer_set_test_but_not_train) > 0:
                answer_set_test_but_not_train = [set(x) for x in list(answer_set_test_but_not_train)]
                for split in ['valid', 'test']:
                    for qa_samples in sampled_list[split]:
                        if qa_samples['question_id'] == qid:
                            qa_samples['sampled_ids'] = {
                                ecg_id: ans for ecg_id, ans in qa_samples['sampled_ids'].items()
                                if set(ans) not in answer_set_test_but_not_train
                            }
                            qa_samples['sizes'] = {
                                k: v for k, v in qa_samples['sizes']
                                if k in qa_samples['sampled_ids']
                            }
                            if split == 'test':
                                cnt += 1
                                print(str(cnt) + ': ' +qa_samples['question'])
                                print(answer_set_test_but_not_train)
                            break

    classes = []
    classes_for_each_template = {}
    if answer_encode == 'multi-label':
        ###############################################################################
        # the following is just for preserving the order of labels in a very naive way
        # otherwise, we can simply use python set() property
        for samples in sampled_list['train']:
            if samples['template_id'] not in classes_for_each_template:
                classes_for_each_template[samples['template_id']] = []
            
            ans =  list(set(sum([x for x in samples['sampled_ids'].values()], [])))
            for a in ans:
                if a not in classes_for_each_template[samples['template_id']]:
                    classes_for_each_template[samples['template_id']].append(a)
            classes_for_each_template[samples['template_id']].sort()

        for template_id, ans in classes_for_each_template.items():
            for a in ans:
                if a not in classes:
                    classes.append(a)
        ###############################################################################
        # define 'none' class as an empty label
        if "none" in classes:
            classes.remove('none')

    pd.DataFrame(
        {'index': i, 'class': c} for i, c in enumerate(classes)
    ).to_csv(os.path.join('results', 'class.csv'), index=False)

    scp_codes_full_names = [category_to_name[x] for x in scp_codes_subclass]
    scp_codes_full_names.extend([category_to_name[x] for x in scp_codes_superclass])
    scp_codes_full_names.extend(["myocardial infarction", "ischemic", "subendocardial injury"])
    sampled_data = {}
    derived_grounding_data = {}
    derived_grounding_tuples = {}
    independent_grounding_data = {}
    independent_grounding_tuples = {}
    grounding_qid = {}
    entire_grounding_qid = {}
    grounding_qid_i = 0

    # we should iterate for test first to compute num_yes_attributes for test split
    for split in ["test", "train", "valid"]:
        pd.DataFrame(sampled_list[split]).to_pickle(os.path.join('results', split +'.pkl'))
        pd.DataFrame(sampled_list[split]).to_csv(os.path.join('results', split + '.csv'), index=False)
        samples = []
        derived_grounding_samples = []
        independent_grounding_samples = []
        sample_id = 0
        print(f"[{split}] Convert QA samples into appropriate data formats (QA data, grounding data)")
        for s in tqdm(sampled_list[split], total=len(sampled_list[split])):
            qtype = templates[templates['template_id'] == s['template_id']].iloc[0]['question_type1']
            atype = templates[templates['template_id'] == s['template_id']].iloc[0]['question_type2']
            category = templates[templates['template_id'] == s['template_id']].iloc[0]["subcategory"]
            template = templates[templates['template_id'] == s['template_id']].iloc[0]["template"]
            candidate = s["candidate"]
            is_numeric = False
            if category in [
                "max_rr_interval",
                "min_rr_interval",
                "max_p_duration",
                "min_p_duration",
                "max_pr_interval",
                "min_pr_interval",
                "max_qrs_duration",
                "min_qrs_duration",
                "max_qt_interval",
                "min_qt_interval",
                "max_qt_corrected",
                "min_qt_corrected"
            ]:
                is_numeric = True
            lead = s["lead"]

            if qtype == "verify":
                attr = candidate
            elif qtype == "choose":
                attr = candidate
            elif qtype == "retrieve":
                if candidate[0] in subcategory:
                    attr = [category_to_name[x] for x in subcategory[candidate[0]] if x in category_to_name]
                else:
                    attr = candidate

            for (ecg_id, ans), size in zip(s['sampled_ids'].items(), s['sizes'].values()):
                question = s['question']
                if qtype == 'choose':
                    attr = candidate.copy()
                    if random.random() > 0.5:
                        question = swap_words(question, candidate[0], candidate[1])
                        attr[0], attr[1] = attr[1], attr[0]
                        ans = ans[::-1]

                samples.append(
                    get_qa_sample(
                        ecg_id=ecg_id,
                        attr=attr,
                        question=question,
                        ans=ans,
                        size=size,
                        qtype=qtype,
                        atype=atype,
                        ptbxl_data_dir=ptbxl_data_dir,
                        sample_id=sample_id,
                        template_id=s["template_id"],
                        question_id=s["question_id"],
                        classes=classes,
                        classes_for_each_template=classes_for_each_template,
                        tokenizer=tokenizer
                    )
                )
                sample_id += 1

                # derive grounding dataset
                metadata = encoded_ptbxl[encoded_ptbxl["ecg_id"] == ecg_id].iloc[0]
                grounding_sample = convert_to_grounding_format(
                    ecg_id=ecg_id,
                    lead=lead,
                    attr=attr,
                    ans=ans,
                    size=size,
                    template=template,
                    category=category,
                    qtype=qtype,
                    is_numeric=is_numeric,
                    lead_names=lead_names,
                    classes=classes,
                    metadata=metadata,
                    per_lead_attr_scp_code=per_lead_attr_scp_code,
                    per_lead_attr_noise_code=per_lead_attr_noise_code,
                    per_lead_to_entire=True,
                )
                if grounding_sample is not None:
                    derived_grounding_samples.append(grounding_sample)

            # construct independent grounding dataset (from verify questions)
            for (ecg_id, ans), size in zip(
                s["total_sampled_ids"].items(), s["total_sampled_sizes"].values()
            ):
                assert qtype == "verify", qtype
                question = s["question"]

                metadata = encoded_ptbxl[encoded_ptbxl["ecg_id"] == ecg_id].iloc[0]
                grounding_sample = convert_to_grounding_format(
                    ecg_id=ecg_id,
                    lead=lead,
                    attr=attr,
                    ans=ans,
                    size=size,
                    template=template,
                    category=category,
                    qtype=qtype,
                    is_numeric=is_numeric,
                    lead_names=lead_names,
                    classes=classes,
                    metadata=metadata,
                    per_lead_attr_scp_code=per_lead_attr_scp_code,
                    per_lead_attr_noise_code=per_lead_attr_noise_code,
                    per_lead_to_entire=False
                )
                if grounding_sample is not None:
                    independent_grounding_samples.append(grounding_sample)

        with open(os.path.join('results', split + '.json'), 'w') as f:
            json.dump(
                {
                    'tokenized': tokenize,
                    'answer_type': answer_encode,
                    'num_labels': len(classes),
                    'samples': samples
                }, f, indent=4
            )

        derived_grounding_data[split] = []
        # linearize grounding samples
        if len(derived_grounding_samples) > 0:
            print(f"[{split}] Linearize and extract derived grounding samples")

            derived_grounding_tuples[split] = linearize_grounding_dict(derived_grounding_samples)
            for ecg_id, attr, obj, positive_idx, size in tqdm(
                derived_grounding_tuples[split], total=len(derived_grounding_tuples[split])
            ):
                grounding_sample = get_grounding_sample(
                    ecg_id=ecg_id,
                    attr=attr,
                    obj=obj,
                    positive_idx=positive_idx,
                    size=size,
                    numeric_features=numeric_features,
                    scp_codes_full_names=scp_codes_full_names,
                    ptbxl_data_dir=ptbxl_data_dir,
                    classes=classes,
                    tokenizer=tokenizer
                )
                if grounding_sample["question_str"] in grounding_qid:
                    grounding_sample["question_id"] = grounding_qid[grounding_sample["question_str"]]
                else:
                    grounding_sample["question_id"] = grounding_qid_i
                    grounding_qid[grounding_sample["question_str"]] = grounding_qid_i
                    if obj == "entire":
                        entire_grounding_qid[grounding_sample["question_str"]] = grounding_qid_i
                    grounding_qid_i += 1

                derived_grounding_data[split].append(grounding_sample)

        #XXX check the uniqueness of 3-tuple (ecg_id, object, attribute)
        tuples_ = [(x['ecg_id'], x['obj'],x['attribute']) for x in derived_grounding_data[split]]
        if len(tuples_) == len(set(tuples_)):
            print("Pass")
        else:
            breakpoint()

        independent_grounding_data[split] = []
        if len(independent_grounding_samples) > 0:
            print(f"[{split}] Linearize and extract independent grounding samples")

            independent_grounding_tuples[split] = linearize_grounding_dict(
                independent_grounding_samples
            )
            num_positives_per_attr_obj = {}
            for ecg_id, attr, obj, positive_idx, size in independent_grounding_tuples[split]:
                if (attr, obj) not in num_positives_per_attr_obj:
                    num_positives_per_attr_obj[(attr, obj)] = 0
                if classes[positive_idx] == "yes":
                    num_positives_per_attr_obj[(attr, obj)] += 1

            sampled_num_negatives_per_attr_obj = {k: 0 for k in num_positives_per_attr_obj.keys()}
            for ecg_id, attr, obj, positive_idx, size in tqdm(
                independent_grounding_tuples[split], total=len(independent_grounding_tuples[split])
            ):
                if (
                    classes[positive_idx] == "yes"
                    or sampled_num_negatives_per_attr_obj[(attr, obj)] < (
                        num_positives_per_attr_obj[(attr, obj)]
                    )
                ):
                    if classes[positive_idx] == "no":
                        sampled_num_negatives_per_attr_obj[(attr, obj)] += 1
                    grounding_sample = get_grounding_sample(
                        ecg_id=ecg_id,
                        attr=attr,
                        obj=obj,
                        positive_idx=positive_idx,
                        size=size,
                        numeric_features=numeric_features,
                        scp_codes_full_names=scp_codes_full_names,
                        ptbxl_data_dir=ptbxl_data_dir,
                        classes=classes,
                        tokenizer=tokenizer
                    )
                    if grounding_sample["question_str"] in grounding_qid:
                        grounding_sample["question_id"] = grounding_qid[grounding_sample["question_str"]]
                    else:
                        raise ValueError(grounding_sample["question_str"])

                    independent_grounding_data[split].append(grounding_sample)

        #XXX check the uniqueness of 3-tuple (ecg_id, object, attribute)
        tuples_ = [(x['ecg_id'], x['obj'],x['attribute']) for x in independent_grounding_data[split]]
        if len(tuples_) == len(set(tuples_)):
            print("Pass")
        else:
            breakpoint()

        sampled_data[split] = {
            'tokenized': tokenize,
            'answer_type': answer_encode,
            'num_labels': len(classes),
            'samples': samples
        }

    index = list(grounding_qid.values())
    question = list(grounding_qid.keys())
    grounding_questions = pd.DataFrame({"index": index, "question":question})
    grounding_questions.set_index("index", inplace=True)
    # for split in derived_grounding_data:
    #     print(f"[{split}] copy and transfer per-lead grounding samples to entire grounding")
    #     reduced_grounding_tuples = list(map(lambda x: (x[0], x[1], x[2]), derived_grounding_tuples[split]))
    #     grounding_qid_i = len(grounding_questions)
    #     per_lead_to_entire = dict()
    #     for s in derived_grounding_data[split]:
    #         if s["obj"] != "entire":
    #             if (s["ecg_id"], s["attribute"]) not in per_lead_to_entire:
    #                 per_lead_to_entire[(s["ecg_id"], s["attribute"])] = []
    #             per_lead_to_entire[(s["ecg_id"], s["attribute"])].append(s)

    #     final_per_lead_to_entire = []
    #     for (ecg_id, attr), samples in tqdm(per_lead_to_entire.items(), total=len(per_lead_to_entire)):
    #         s = samples[0].copy()
    #         s["question_str"] = s["question_str"].split(" in " + s["obj"])[0] + "?"
    #         if s["question_str"] in entire_grounding_qid:
    #             s["question_id"] = entire_grounding_qid[s["question_str"]]
    #         else:
    #             entire_grounding_qid[s["question_str"]] = grounding_qid_i
    #             s["question_id"] = grounding_qid_i
    #             grounding_questions.loc[grounding_qid_i] = s["question_str"]
    #             grounding_qid_i += 1
    #         s["question"] = tokenizer.encode(s["question_str"], add_special_tokens=False)
    #         s["original_obj"] = s["obj"]
    #         s["obj"] = "entire"

    #         metadata = encoded_ptbxl[encoded_ptbxl["ecg_id"] == ecg_id]
    #         flag = False
    #         if attr in per_lead_attr_scp_code:
    #             attr_code = per_lead_attr_scp_code[attr]
    #             scp_codes = eval(metadata["scp_codes"].iloc[0])

    #             if isinstance(attr_code, list):
    #                 for c in attr_code:
    #                     if c in scp_codes:
    #                         flag = True
    #                         break
    #             else:
    #                 if attr_code in scp_codes:
    #                     flag = True
    #         elif attr in per_lead_attr_noise_code:
    #             attr_code = per_lead_attr_noise_code[attr]
    #             if metadata[attr_code].iloc[0] is not None:
    #                 flag = True

    #         answer = np.zeros(len(s["answer"]), dtype=int)
    #         if flag:
    #             answer[1] = 1
    #             s["answer_bin"] = 1
    #         else:
    #             answer[0] = 1
    #             s["answer_bin"] = 0
    #         s["answer"] = list(answer)
    #         if (ecg_id, attr, "entire") not in reduced_grounding_tuples:
    #             final_per_lead_to_entire.append(s)

    #     derived_grounding_data[split].extend(final_per_lead_to_entire)

    test_num_grounding_per_attribute_obj = dict()
    for s in derived_grounding_data["test"]:
        if (s["attribute"], s["obj"]) not in test_num_grounding_per_attribute_obj:
            test_num_grounding_per_attribute_obj[(s["attribute"], s["obj"])] = 0

        if s["answer_bin"] == 1:
            test_num_grounding_per_attribute_obj[(s["attribute"], s["obj"])] += 1
    exclude = [key for key, value in test_num_grounding_per_attribute_obj.items() if value < 10]

    for split in derived_grounding_data:
        derived_grounding_data[split] = [
            s for s in derived_grounding_data[split]
            if (
                (s["attribute"], s["obj"]) not in exclude
                and (s["attribute"], s["obj"]) in test_num_grounding_per_attribute_obj
            )
        ]

    for split in independent_grounding_data:
        independent_grounding_data[split] = [
            s for s in independent_grounding_data[split]
            if (
                (s["attribute"], s["obj"]) not in exclude
                and (s["attribute"], s["obj"]) in test_num_grounding_per_attribute_obj
            )
        ]

    grounding_questions.to_csv("results/grounding_questions.csv")

    grounding_classes = list(set(x["attribute"] for x in derived_grounding_data["test"]))
    grounding_classes.sort()

    pd.DataFrame(
        {'index': i, 'class': c} for i, c in enumerate(grounding_classes)
    ).to_csv(os.path.join('results', 'grounding_class.csv'), index=False)

    for split in derived_grounding_data:
        for i, s in enumerate(derived_grounding_data[split]):
            # in case that the class (attribute) is present in train/valid set but not in test set
            if s["attribute"] not in grounding_classes:
                continue

            s["target_idx"] = grounding_classes.index(s["attribute"])
            s["attribute_id"] = s["target_idx"]
    for split in independent_grounding_data:
        for i, s in enumerate(independent_grounding_data[split]):
            # in case that the class (attribute) is present in train/valid set but not in test set
            if s["attribute"] not in grounding_classes:
                continue

            s["target_idx"] = grounding_classes.index(s["attribute"])
            s["attribute_id"] = s["target_idx"]

    with open(os.path.join('results', 'sampled_data.pkl'), 'wb') as f:
        pickle.dump(sampled_data, f)
    with open(os.path.join('results', 'derived_grounding_data.pkl'), 'wb') as f:
        pickle.dump(derived_grounding_data, f)
    with open(os.path.join('results', 'independent_grounding_data.pkl'), 'wb') as f:
        pickle.dump(independent_grounding_data, f)

    return sampled_data, derived_grounding_data, independent_grounding_data

def get_qa_sample(
    ecg_id,
    attr,
    question,
    ans,
    size,
    qtype,
    atype,
    ptbxl_data_dir,
    sample_id,
    template_id,
    question_id,
    classes,
    classes_for_each_template,
    tokenizer
):
    question_str = question
    question = tokenizer.encode(question.lower(), add_special_tokens=False)

    answer_str = ans.copy()
    answer = [1 if x in ans else 0 for x in classes]
    
    if qtype == "verify":
        question_type = 0
    elif qtype == "choose":
        question_type = 1
    elif qtype == "retrieve":
        question_type = 2
    else:
        raise ValueError(qtype)
    
    if atype == "multi-label":
        answer_type = 0
    elif atype == "multi-class":
        answer_type = 1
    else:
        raise ValueError(atype)
    
    if qtype == "choose":
        class_idcs = [classes.index(a) for a in attr]
        class_idcs.sort()
    else:
        class_idcs = [
            i for i, c in enumerate(classes)
            if c in classes_for_each_template[template_id]
        ]

    ecg_path = get_ptbxl_data_path(ecg_id, ptbxl_data_dir)

    return {
        "sample_id": sample_id,
        "template_id": template_id,
        "question_id": question_id,
        "question_str": question_str,
        "answer_str": answer_str,
        "qtype": question_type,
        "atype": answer_type,
        "question": question,
        "answer": answer.copy(),
        "classes": class_idcs,
        "ecg_path": ecg_path,
        "ecg_id": ecg_id,
        "size": size
    }

def convert_to_grounding_format(
    ecg_id,
    lead,
    attr,
    ans,
    size,
    template,
    category,
    qtype,
    is_numeric,
    lead_names,
    classes,
    metadata,
    per_lead_attr_scp_code,
    per_lead_attr_noise_code,
    per_lead_to_entire=False,
):
    assert ecg_id == metadata.ecg_id

    grounding_ans = None
    grounding_attr = None
    grounding_obj = None

    if qtype == "verify":
        if ans[0] == "not sure":
            return None
        else:
            if "excluding uncertain symptoms" in template:
                if ans[0] == "no":
                    return None
                else:
                    grounding_ans = ["yes"]
            elif "including uncertain symptoms" in template:
                if ans[0] == "yes":
                    return None
                else:
                    grounding_ans = ["no"]
            else:
                grounding_ans = ans.copy()

            grounding_attr = attr.copy()
            if is_numeric:
                grounding_attr = [x + " of " + category for x in grounding_attr]

            assert len(grounding_attr) == 1
            # in case of verifying existence of an attribute in a "specific lead"
            if lead is not None:
                grounding_obj = [lead]
            # in case of verifying existence of an attribute from the "entire" ecg
            else:
                grounding_obj = [parse_lead_position(attr[0])]
    elif qtype == "choose":
        if "excluding uncertain symptoms" in template:
            grounding_attr = list(set(ans) - {"none"})
            grounding_ans = ["yes"] * len(grounding_attr)
        elif "including uncertain symptoms" in template:
            grounding_attr = list(set(attr) - set(ans))
            grounding_ans = ["no"] * len(grounding_attr)
        else:
            grounding_attr = attr.copy()
            grounding_ans = ["yes" if x in ans else "no" for x in grounding_attr]
            if is_numeric:
                grounding_attr = [x + " of " + category for x in grounding_attr]

        # in case of choosing an attribute in a "specific lead"
        # note that we are assuming there is no question of choosing a specific "lead",
        # which means that `lead` should be not ``None`` if `lead` in question
        if "${lead}" in template:
            assert lead is not None
            grounding_obj = [lead] * len(grounding_attr)
        # in case of choosing an attribute from the "entire" ecg
        else:
            grounding_obj = []
            for attribute in grounding_attr:
                grounding_obj.append(parse_lead_position(attribute))
    elif qtype == "retrieve":
        # in case of retrieving attributes in a specific lead
        if "${lead}" in template:
            assert lead is not None
            grounding_ans = ["yes" if x in ans else "no" for x in attr]
            grounding_attr = attr.copy()
            grounding_obj = [lead] * len(attr)
        # in case of retrieving leads of a specific attribute
        elif "lead" in category:
            grounding_ans = ["yes" if x in ans else "no" for x in lead_names]
            grounding_attr = attr.copy() * len(grounding_ans)
            grounding_obj = lead_names.copy()
        # in case of retrieving attributes from the entire ecg
        else:
            if "excluding uncertain symptoms" in template:
                grounding_attr = list(set(ans) - {"none"})
                grounding_ans = ["yes"] * len(grounding_attr)
            elif "including uncertain symptoms" in template:
                grounding_attr = list(set(attr) - set(ans))
                grounding_ans = ["no"] * len(grounding_attr)
            else:
                grounding_attr = attr.copy()
                grounding_ans = ["yes" if x in ans else "no" for x in grounding_attr]
                if is_numeric:
                    grounding_attr = [x + " of " + category for x in grounding_attr]
            
            grounding_obj = []
            for attribute in grounding_attr:
                grounding_obj.append(parse_lead_position(attribute))
    else:
        raise ValueError(qtype)


    for i in range(len(grounding_attr)):
        if "myocardial infarction" in grounding_attr[i]:
            grounding_attr[i] = re.sub(
                r"myocardial infarction in.* leads", "myocardial infarction", grounding_attr[i]
            )
        elif "subendocardial injury" in grounding_attr[i]:
            grounding_attr[i] = re.sub(
                r"subendocardial injury in.* leads", "subendocardial injury", grounding_attr[i]
            )
        elif "ischemic" in grounding_attr[i]:
            grounding_attr[i] = re.sub(
                r"ischemic in.* leads", "ischemic", grounding_attr[i]
            )

    per_lead_to_entire_attr = []
    per_lead_to_entire_obj = []
    per_lead_to_entire_ans = []
    for i in range(len(grounding_obj)):
        if grounding_obj[i] != "entire":
            if (
                grounding_attr[i] in [
                    "subendocardial injury",
                    "ischemic"
                ]
                or per_lead_to_entire
            ):
                per_lead_to_entire_attr.append(grounding_attr[i])
                per_lead_to_entire_obj.append("entire")

                flag = False
                if grounding_attr[i] in per_lead_attr_scp_code:
                    attr_code = per_lead_attr_scp_code[grounding_attr[i]]
                    scp_codes = eval(metadata["scp_codes"])

                    if isinstance(attr_code, list):
                        for c in attr_code:
                            if c in scp_codes:
                                flag = True
                                break
                    else:
                        if attr_code in scp_codes:
                            flag = True
                elif grounding_attr[i] in per_lead_attr_noise_code:
                    attr_code = per_lead_attr_noise_code[grounding_attr[i]]
                    if metadata[attr_code] is not None:
                        flag = True
                else:
                    raise AssertionError()

                if flag:
                    per_lead_to_entire_ans.append("yes")
                else:
                    per_lead_to_entire_ans.append("no")

    checked = []
    for attr_, obj_, ans_ in zip(
        per_lead_to_entire_attr, per_lead_to_entire_obj, per_lead_to_entire_ans
    ):
        if (attr_, obj_, ans_) not in checked:
            grounding_attr.append(attr_)
            grounding_obj.append(obj_)
            grounding_ans.append(ans_)
            checked.append((attr_, obj_, ans_))

    positive_idcs = [classes.index(x) for x in grounding_ans]

    return {
        "ecg_id": [ecg_id] * len(grounding_ans),
        "attr": grounding_attr.copy(),
        "obj": grounding_obj.copy(),
        "positive_idcs": positive_idcs.copy(),
        "size": [size] * len(grounding_ans)
    }

def linearize_grounding_dict(ls: List[Dict[str, list]]):
    key = list(ls[0].keys())
    assert key == ["ecg_id", "attr", "obj", "positive_idcs", "size"]

    for x in ls:
        if list(x.keys()) != key:
            raise ValueError("dictionarys in the list should have the same key one another")

    unique_tuples = [
        (ecg_id, attr, obj, positive_idx, size)
        for x in ls for ecg_id, attr, obj, positive_idx, size in zip(
            x["ecg_id"], x["attr"], x["obj"], x["positive_idcs"], x["size"]
        )
    ]
    unique_tuples = list(set(unique_tuples))

    return unique_tuples

def get_grounding_sample(
    ecg_id,
    attr,
    obj,
    positive_idx,
    size,
    numeric_features,
    scp_codes_full_names,
    ptbxl_data_dir,
    classes,
    tokenizer
):
    if (numeric_feature := re.search("|".join(numeric_features), attr)) is not None:
        numeric_feature = " ".join(numeric_feature.group().split("_"))
        adj = "lowest" if "min" in attr else "highest"
        level = "below" if "below" in attr else "within" if "within" in attr else "above"
        question = (
            "does the " + adj + " " + numeric_feature + " of this ecg fall "
            + level + " the normal range"
        )
    elif attr == "normal ecg":
        question = "is this a normal ecg"
    elif attr in scp_codes_full_names:
        question = "does this ecg show symptoms of " + attr
    else:
        question = "does this ecg show " + attr
    
    assert obj is not None
    if obj == "entire":
        question += "?"
    else:
        question += " in " + obj + "?"

    question_str = question
    question = tokenizer.encode(question.lower(), add_special_tokens=False)

    answer_str = classes[positive_idx]
    answer = [0] * len(classes)
    answer[positive_idx] = 1
    answer_bin = 0 if classes[positive_idx] == "no" else 1

    ecg_path = get_ptbxl_data_path(ecg_id, ptbxl_data_dir)

    return {
        "question_str": question_str,
        "answer_str": answer_str,
        "qtype": 0, # verify
        "atype": 1, # multi-class
        "question": question,
        "attribute": attr,
        "answer": answer,
        "answer_bin": answer_bin,
        "classes": [0, 1],
        "ecg_path": ecg_path,
        "ecg_id": ecg_id,
        "size": size,
        "obj": obj
    }

def parse_lead_position(str):
    matched_lead_position = re.search(r"(?<=in ).* leads", str)
    if matched_lead_position is not None:
        matched_lead_position = matched_lead_position.group()
        if "frontal" in matched_lead_position:
            return "entire"
        else:
            return matched_lead_position
    return "entire"

def _count_min_from_n(ls, n, cnt_threshold=1):
    conds = list(map(lambda x: len(x) >= cnt_threshold, ls))
    if sum(conds) >= n:
        cnt = min([len(x) if conds[i] else sys.maxsize for i, x in enumerate(ls)])
    else:
        cnt = 0
    return cnt

def _sample(samples_with_candidate: dict, qtype, assigned_ecgs=[], n=1, seed=None, update_seed=True):
    res = dict()
    # random sample
    num = n
    for k, v in samples_with_candidate.items():
        if qtype == "choose":
            v = v[~v['ecg_id'].isin(assigned_ecgs)]
            if k == "none":
                num = int(n/2 + random.random())
        elif qtype == "verify" and k in ["no_type1", "no_type2"]:
            if k == "no_type1":
                num = n - min(int(n / 2), len(samples_with_candidate["no_type2"]))
            elif k == "no_type2":
                num = int(n / 2)

        res[k] = v.sample(n=min(num, len(v)), random_state=seed)
        if seed is not None and update_seed:
            seed = seed + random.randint(1, 10)

    return res, seed

def sample(
    data,
    candidate,
    template,
    subcategory,
    category_to_name,
    ans_to_name,
    assigned_ecgs=[],
    lead=None,
    split='train',
    lead_names=[
        'lead I', 'lead II', 'lead III', 'lead aVR', 'lead aVL', 'lead aVF',
        'lead V1', 'lead V2', 'lead V3', 'lead V4', 'lead V5', 'lead V6'
    ],
    lead_groups={
        "limb leads": ["lead I", "lead II", "lead III", "lead aVR", "lead aVL", "lead aVF"],
        "chest leads": ["lead V1", "lead V2", "lead V3", "lead V4", "lead V5", "lead V6"]
    },
    cnt_threshold=1,
    n=5,
    question_id=-1,
    seed=None,
):
    assert template['question_type1'] in ['verify', 'choose', 'retrieve']

    if lead is not None:
        lead_name = lead
        if lead in lead_names:
            lead = lead_names.index(lead)
        elif lead in lead_groups:
            lead = [lead_names.index(l) for l in lead_groups[lead]]
        else:
            raise ValueError("{} is invalid lead indicator".format(lead))

    if template['subcategory'] in [
            'scp_codes_superclass',
            'any_scp_codes_subclass',
            'any_scp_codes_diagnostic_inc',
            'any_scp_codes_diagnostic_exc',
            'any_scp_codes_form',
            'any_scp_codes_rhythm',
            'heart_axis_direction',
            'infarction_stadium',
            'infarction_stadium_ret',
            'any_extra_systole',
            "max_rr_interval",
            "min_rr_interval",
            "max_p_duration",
            "min_p_duration",
            "max_pr_interval",
            "min_pr_interval",
            "max_qrs_duration",
            "min_qrs_duration",
            "max_qt_interval",
            "min_qt_interval",
            "max_qt_corrected",
            "min_qt_corrected"
    ]:
        is_exists = lambda x, c: x[c] == True
    elif template['subcategory'] in [
            'scp_codes_form_with_leads',
            'noise',
            'noise_with_leads',
            'any_noise',
    ]:
        is_exists = lambda x, c: x[c].notnull()
    elif template['subcategory'] in [
            'scp_codes_subclass',
            'scp_codes_norm',
            'scp_codes_diagnostic_inc',
            'scp_codes_diagnostic_exc',
            'scp_codes_form',
            'scp_codes_rhythm',
    ]:
        is_exists = lambda x, c: x['scp_codes'].str.contains("'" + c + "'", regex=False)
    elif template['subcategory'] in [
            'extra_systole',
    ]:
        is_exists = lambda x, c: x[c] >= 0
    else:
        raise ValueError(
            template['subcategory']
        )

    min_answer_cnt = 2 if split == 'train' else 1
    if template['question_type1'] == 'choose':
        if lead is not None:
            # the following condition is just for readability and sanity check
            if not isinstance(lead, list):
                if template['subcategory'] in [
                    "noise",
                ]:
                    subsample = data[
                        is_exists(data, candidate[0]) | is_exists(data, candidate[1])
                    ].sort_index()
                    a = data[is_exists(data, candidate[0])][candidate[0]].apply(
                        lambda x: x[lead]
                    ).astype(bool)
                    b = data[is_exists(data, candidate[1])][candidate[1]].apply(
                        lambda x: x[lead]
                    ).astype(bool)
                    both = subsample[a & b]
                    a_only = subsample[a & ~b]
                    b_only = subsample[~a & b]
                    cnt = _count_min_from_n([both, a_only, b_only], n=min_answer_cnt)

                    none = subsample[~a & ~b]
                    subsample = subsample[~subsample['ecg_id'].isin(none['ecg_id'])]
                    none = data[~data['ecg_id'].isin(subsample['ecg_id'])]

                    ans1 = ans_to_name[candidate[0]] if candidate[0] in ans_to_name else candidate[0]
                    ans2 = ans_to_name[candidate[1]] if candidate[1] in ans_to_name else candidate[1]
                    samples_with_candidate = {
                        ans1: a_only,
                        ans2: b_only,
                        frozenset([ans1, ans2]): both,
                        'none': none,
                    }
            else:
                raise ValueError(
                    template['subcategory']
                )
        elif template['subcategory'] in [
                'scp_codes_diagnostic_exc'
        ]:
            def _search(pattern, string, default_value=None):
                matched = re.search(pattern, string)
                if matched is not None:
                    return matched.group()
                return default_value

            samples_with_candidate = data[
                is_exists(data, candidate[0]) | is_exists(data, candidate[1])
            ]
            likelihoods = [
                samples_with_candidate['scp_codes'].apply(
                    lambda x: _search(r'(?<=\'' + c + r'\': )\d+\.\d', x, default_value=-1)
                ).astype(float) for c in candidate
            ]
            a_only = samples_with_candidate[
                ((likelihoods[0] == 0) | (likelihoods[0] == 100))
                & ((likelihoods[1] != 0) & (likelihoods[1] != 100))
            ]
            b_only = samples_with_candidate[
                ((likelihoods[0] != 0) & (likelihoods[0] != 100))
                & ((likelihoods[1] == 0) | (likelihoods[1] == 100))
            ]
            both = samples_with_candidate[
                ((likelihoods[0] == 0) | (likelihoods[0] == 100))
                & ((likelihoods[1] == 0) | (likelihoods[1] == 100))
            ]
            ecg_ids = sum([x['ecg_id'].to_list() for x in [a_only, b_only, both]],[])
            cnt = _count_min_from_n([both, a_only, b_only], n=min_answer_cnt)

            none = data[~data['ecg_id'].isin(ecg_ids)]

            ans1 = ans_to_name[candidate[0]] if candidate[0] in ans_to_name else candidate[0]
            ans2 = ans_to_name[candidate[1]] if candidate[1] in ans_to_name else candidate[1]
            samples_with_candidate = {
                ans1: a_only,
                ans2: b_only,
                frozenset([ans1, ans2]): both,
                'none': none,
            }
        else:
            a_only = data[
                is_exists(data, candidate[0]) & ~is_exists(data, candidate[1])
            ]
            b_only = data[
                ~is_exists(data, candidate[0]) & is_exists(data, candidate[1])
            ]
            both = data[
                is_exists(data, candidate[0]) & is_exists(data, candidate[1])
            ]
            cnt = _count_min_from_n([both, a_only, b_only], n=min_answer_cnt)

            ans1 = ans_to_name[candidate[0]] if candidate[0] in ans_to_name else candidate[0]
            ans2 = ans_to_name[candidate[1]] if candidate[1] in ans_to_name else candidate[1]
            samples_with_candidate = {
                ans1: a_only,
                ans2: b_only,
                frozenset([ans1, ans2]): both,
            }

            none = data[
                ~is_exists(data, candidate[0]) & ~is_exists(data, candidate[1])
            ]
            samples_with_candidate['none'] = none
    elif template['question_type1'] == 'verify':
        if lead is not None:
            # the following condition is just for readability and sanity check
            if not isinstance(lead, list):
                if template['subcategory'] in [
                    'scp_codes_form_with_leads',
                    'noise',
                    'any_noise',
                ]:
                    subsample = data[is_exists(data, candidate)]
                    yes = subsample[
                        subsample[candidate].apply(lambda x: x[lead]).astype(bool)
                    ]
                    # type1: samples that have no corresponding attribute at all
                    no_type1 = data[~is_exists(data, candidate)]
                    # type2: samples that have the corresponding attribute but not in that lead
                    no_type2 = subsample[
                        subsample[candidate].apply(lambda x: x.sum() > 0 and x[lead] == 0).astype(bool)
                    ]

                    cnt = len(yes)
                    samples_with_candidate = {
                        'yes': yes,
                        "no_type1": no_type1,
                        "no_type2": no_type2
                    }
                else:
                    raise ValueError(
                        template['subcategory']
                    )
            else:
                if template["subcategory"] in [
                    "scp_codes_form_with_leads"
                ]:
                    subsample = data[is_exists(data, candidate)]
                    yes = subsample[
                        subsample[candidate].apply(lambda x: x[lead].all())
                    ]
                    # type1: samples that have no corresponding attribute at all
                    no_type1 = data[~is_exists(data, candidate)]
                    # type2: samples that have the corresponding attribute but not in that lead
                    no_type2 = subsample[
                        subsample[candidate].apply(lambda x: x.sum() > 0 and not x[lead].all()).astype(bool)
                    ]
                    cnt = len(yes)
                    samples_with_candidate = {
                        "yes": yes,
                        "no_type1": no_type1,
                        "no_type2": no_type2,
                    }
                else:
                    raise ValueError(
                        template['subcategory']
                    )
        else:
            if (
                template['subcategory'] == 'scp_codes_subclass'
                and candidate in subcategory['scp_codes_diagnostic_inc']
            ):
                samples_with_candidate = data[is_exists(data, candidate)]
                likelihood = samples_with_candidate['scp_codes'].apply(
                    lambda x: re.search(r'(?<=\'' + candidate + r'\': )\d+\.\d', x).group()
                ).astype(float)
                p_0_100 = likelihood[(likelihood == 0) | (likelihood == 100)]
                yes = samples_with_candidate.loc[p_0_100.index]

                p_15_35_50_80 = likelihood[(likelihood == 15) | (likelihood == 35) | (likelihood == 50) | (likelihood == 80)]
                not_sure = samples_with_candidate.loc[p_15_35_50_80.index]

                no = data[~(data['ecg_id'].isin(yes['ecg_id']) | data['ecg_id'].isin(not_sure['ecg_id']))]

                cnt = _count_min_from_n([yes, not_sure, no], n=min_answer_cnt)
                samples_with_candidate = {
                    'yes': yes,
                    'not sure': not_sure,
                    'no': no,
                }
            # the following is the case that we should exclude nan rows in heart_axis_direction column
            elif template['subcategory'] in [
                'heart_axis_direction',
                "max_rr_interval",
                "min_rr_interval",
                "max_p_duration",
                "min_p_duration",
                "max_pr_interval",
                "min_pr_interval",
                "max_qrs_duration",
                "min_qrs_duration",
                "max_qt_interval",
                "min_qt_interval",
                "max_qt_corrected",
                "min_qt_corrected"
            ]:
                subsample = data[~data[candidate].isna()]
                yes = subsample[is_exists(subsample, candidate)]
                no = subsample[~is_exists(subsample, candidate)]
                cnt = len(yes)
                samples_with_candidate = {
                    'yes': yes,
                    'no': no
                }
            else:
                yes = data[is_exists(data, candidate)]
                no = data[~is_exists(data, candidate)]
                cnt = len(yes)
                samples_with_candidate = {
                    'yes': yes,
                    'no': no
                }
    elif template['question_type1'] == 'retrieve':
        # grounded to a specific lead
        if lead is not None:
            if not isinstance(lead, list):
                candidates = subcategory[template['subcategory']]
                samples_with_candidate = dict()
                # the following condition is just for readability and sanity check
                # multi-label retrieve (with a specific lead)
                if template['subcategory'] in [
                    'noise'
                ]:
                    data['_interests'] = data.apply(
                        lambda x: frozenset(
                            ans_to_name[noise] if noise in ans_to_name else noise
                            for noise in candidates
                            if x[noise] is not None and x[noise][lead] == 1
                        ),
                        axis=1
                    )
                else:
                    raise ValueError(
                        template['subcategory']
                    )
            else:
                raise ValueError(
                    template['subcategory']
                )

            unique_answer_combinations = data['_interests'].unique()
            for answer_combination in unique_answer_combinations:
                if len(answer_combination) == 0:
                    key = 'none'
                else:
                    key = answer_combination
                samples_with_candidate[key] = (
                    data[data['_interests'] == answer_combination]
                )
            cnt = _count_min_from_n(samples_with_candidate.values(), n=min_answer_cnt+1) # +1 since it contains 'none'
        else:
            # retrieve, which needs to convert subcategory to its elements
            if template['subcategory'] in [
                'scp_codes_diagnostic_inc',
                'scp_codes_diagnostic_exc',
                'scp_codes_rhythm',
                'scp_codes_form',
                'heart_axis_direction',
                'infarction_stadium_ret',
                'noise',
                'extra_systole',
                "max_rr_interval",
                "min_rr_interval",
                "max_p_duration",
                "min_p_duration",
                "max_pr_interval",
                "min_pr_interval",
                "max_qrs_duration",
                "min_qrs_duration",
                "max_qt_interval",
                "min_qt_interval",
                "max_qt_corrected",
                "min_qt_corrected"
            ]:
                candidates = subcategory[template['subcategory']]
                samples_with_candidate = dict()

                # multi-class retrieve
                if template['subcategory'] in [
                    'heart_axis_direction',
                    'infarction_stadium_ret',
                    "max_rr_interval",
                    "min_rr_interval",
                    "max_p_duration",
                    "min_p_duration",
                    "max_pr_interval",
                    "min_pr_interval",
                    "max_qrs_duration",
                    "min_qrs_duration",
                    "max_qt_interval",
                    "min_qt_interval",
                    "max_qt_corrected",
                    "min_qt_corrected"
                ]:
                    for c in candidates:
                        samples_with_candidate[
                            ans_to_name[c] if c in ans_to_name else c
                        ] = data[is_exists(data, c)]
                # multi-label retrieve
                else:
                    # special case that needs to check something (likelihood in this case)
                    if template['subcategory'] == 'scp_codes_diagnostic_exc':
                        data['_interests'] = data['scp_codes'].apply(
                            lambda x: frozenset(
                                ans_to_name[code] if code in ans_to_name else code
                                for code, likelihood in eval(x).items()
                                if ((likelihood == 0) | (likelihood == 100)) and code in candidates
                            )
                        )
                    # for scp_codes that need to be parsed from `scp_codes` column
                    elif template['subcategory'] in [
                        'scp_codes_diagnostic_inc',
                        'scp_codes_rhythm',
                        'scp_codes_form'
                    ]:
                        data['_interests'] = data['scp_codes'].apply(
                            lambda x: frozenset(
                                ans_to_name[code] if code in ans_to_name else code
                                for code in eval(x)
                                if code in candidates
                            )
                        )
                    # for noise and extra_systole that are already encoded
                    elif template['subcategory'] in [
                        'noise',
                    ]:
                        data['_interests'] = data.apply(
                            lambda x: frozenset(
                                ans_to_name[noise] if noise in ans_to_name else noise
                                for noise in candidates
                                if x[noise] is not None
                            ),
                            axis=1
                        )
                    elif template['subcategory'] in [
                        'extra_systole' # ES_COUNT / VES_COUNT / SVES_COUNT
                    ]:
                        data['_interests'] = data.apply(
                            lambda x: frozenset(
                                ans_to_name[noise] if noise in ans_to_name else noise
                                for noise in candidates
                                if x[noise] >= 0
                            ),
                            axis=1
                        )

                    unique_answer_combinations = data['_interests'].unique()
                    for answer_combination in unique_answer_combinations:
                        if len(answer_combination) == 0:
                            key = 'none'
                        else:
                            key = answer_combination
                        samples_with_candidate[key] = (
                            data[data['_interests'] == answer_combination]
                        )

                cnt = _count_min_from_n(samples_with_candidate.values(), n=min_answer_cnt)
            # multi-label retrieve for leads
            elif template['subcategory'] in [
                'scp_codes_form_with_leads',
                'noise_with_leads',
                # 'extra_systole_with_leads',
            ]:
                subsample = data[is_exists(data, candidate)]
                samples_with_candidate = dict()
                unique_answer_combinations = [
                    tuple(x) for x in set(tuple(x) for x in subsample[candidate].to_list())
                ]
                for answer_combination in unique_answer_combinations:
                    if sum(answer_combination) == 0:
                        continue
                    key = tuple([lead_names[i] for i, v in enumerate(answer_combination) if v == 1])
                    samples_with_candidate[key] = (
                        subsample[subsample[candidate].apply(lambda x: tuple(x) == answer_combination)]
                    )
                samples_with_candidate['none'] = data[~is_exists(data, candidate)]
                cnt = _count_min_from_n(samples_with_candidate.values(), n=min_answer_cnt+1) # +1 since it contains 'none'
            else:
                raise ValueError(
                    template['subcategory']
                )

    res = None
    if cnt >= cnt_threshold:
        if template['question_type1'] == 'choose':
            _candidate = list(candidate)
        else:
            _candidate = [candidate]

        for i, c in enumerate(_candidate):
            if c not in category_to_name:
                if template["question_type1"] == "retrieve":
                    continue
                else:
                    raise AssertionError()
            _candidate[i] = category_to_name[c]

        if template['question_type1'] == 'choose':
            question = template['template'].replace('${' + template['subcategory'] + '1}', _candidate[0])
            question = question.replace('${' + template['subcategory'] + '2}', _candidate[1])
        else:
            question = template['template'].replace('${' + template['subcategory'] + '}', _candidate[0])

        if lead is not None:
            question = question.replace("${lead}", lead_name)

        if template["question_type1"] == "verify":
            n = 4 * n
        elif template["question_type1"] == "retrieve":
            n = 2 * n

        if split != "train" and template["question_type1"] != "retrieve":
            n = min(cnt, n)

        _sampled, seed = _sample(
            samples_with_candidate,
            qtype=template['question_type1'],
            assigned_ecgs=assigned_ecgs,
            n=n,
            seed=seed
        )
        if "no_type1" in _sampled:
            _sampled["no"] = pd.concat([_sampled["no_type1"], _sampled["no_type2"]])
            del _sampled["no_type1"]
            del _sampled["no_type2"]

        _total_sampled = None
        if template["question_type1"] == "verify":
            if "no_type1" in samples_with_candidate:
                n = min(
                    len(samples_with_candidate["yes"]),
                    len(samples_with_candidate["no_type1"]) + len(samples_with_candidate["no_type2"])
                )
            else:
                n = min(len(samples_with_candidate["yes"]), len(samples_with_candidate["no"]))
            _total_sampled, _ = _sample(
                samples_with_candidate,
                qtype=template["question_type1"],
                n=n,
                seed=seed,
                update_seed=False
            )
            if "no_type1" in _total_sampled:
                _total_sampled["no"] = pd.concat([_total_sampled["no_type1"], _total_sampled["no_type2"]])
                del _total_sampled["no_type1"]
                del _total_sampled["no_type2"]

        sampled = dict()
        sizes = dict()
        for k, v in _sampled.items():
            _ids = v['ecg_id'].to_list()
            _sizes = v['size'].to_list()
            for id, size in zip(_ids, _sizes):
                sampled[id] = [k] if isinstance(k, str) else list(k)
                sizes[id] = size

        total_sampled = dict()
        total_sizes = dict()
        if _total_sampled is not None:
            for k, v in _total_sampled.items():
                _ids = v['ecg_id'].to_list()
                _sizes = v['size'].to_list()
                for id, size in zip(_ids, _sizes):
                    total_sampled[id] = [k] if isinstance(k, str) else list(k)
                    total_sizes[id] = size

        res = {
            'template_id': template['template_id'],
            'question_id': question_id,
            'question': question,
            "lead": lead_name if lead is not None else None,
            'num_samples': cnt,
            'sampled_ids': sampled,
            'sizes': sizes,
            'candidate': _candidate,
            "total_sampled_ids": total_sampled,
            "total_sampled_sizes": total_sizes,
        }

    return res, seed

def swap_words(s, x, y):
    """
    swap words x with y given a string s.
    copied from https://stackoverflow.com/questions/70209111/how-to-swap-words-multiple-characters-in-a-string
    """
    return y.join(part.replace(y, x) for part in s.split(x))

def get_ptbxl_data_path(ecg_id, ptbxl_data_dir):
    return os.path.join(
        ptbxl_data_dir,
        "records500",
        f"{int(ecg_id / 1000) * 1000 :05d}",
        f"{ecg_id:05d}_hr"
    )
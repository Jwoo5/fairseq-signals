import os
import glob
import re
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List

from scipy.spatial import distance
import networkx as nx
from networkx.algorithms.bipartite import matching as mx

# from nltk.translate.bleu_score import sentence_bleu
# from transformers import AutoTokenizer

def map_to_mimiciii(ptbxl_dir, mimic_dir):
    codebook = {
        'NORM': [
            'normal ecg',
            'normal tracing',
        ],
        'LAFB': [
            'lafb',
            'left anterior hemiblock',
            'left anterior fascicular block',
        ],
        'LPFB': [
            'lpfb',
            'left posterior hemiblock',
            'left posterior fascicular block',
        ],
        '1AVB': [
            '1st degree a-v block',
            'first degree av block',
            'first degree a-v block',
        ],
        '2AVB': [
            'second degree av block',
            'second degree a-v block',
            '2nd degree av block',
            '2nd degree a-v block',
            '2:1 av block',
            '2:1 a-v block',
            '3:1 av block',
            '3:1 a-v block',
            '4:1 av block',
            '4:1 a-v block',
        ],
        '3AVB': [
            'third degree av block',
            'third degree a-v block',
            '3rd degree av block',
            '3rd degree a-v block'
        ],
        'IVCD': [
            'intraventricular conduction delay',
            'intraventricular conduction defect',
        ],
        'WPW': [
            'wpw',
            'parkinson-white',
        ],
        'IRBBB': [
            'incomplete right bundle branch block',
            'incomplete right bundle-branch block',
            'incomplete rbbb',
        ],
        'ILBBB': [
            'incomplete left bundle branch block',
            'incomplete left bundle-branch block',
            'incomplete lbbb',
        ],
        'CRBBB': [
            'right bundle branch block',
            'right bundle-branch block',
            'rbbb',
            '!incomplete',
        ],
        'CLBBB': [
            'left bundle branch block',
            'left bundle-branch block',
            'lbbb',
            '!incomplete',
        ],
        'LAO/LAE': [
            'left atrial enlargement',
            'left atrial overload',
        ],
        'RAO/RAE': [
            'right atrial enlargement',
            'right atrial overload',
        ],
        'LVH': [
            'left ventricular hypertrophy',
            'left ventricular overload',
            'left ventricular enlargement',
            'left ventricular predominance',
            'left ventricular preponderance',
        ],
        'RVH': [
            'right ventricular hypertrophy',
            'right ventricular overload',
            'right ventricular enlargement',
            'right ventricular predominance',
            'right ventricular preponderance',
        ],
        'SEHYP': [
            'septal hypertrophy',
        ],
        'IMI': [
            'inferior myocardial infarct',
            'inferior wall infarct',
            'inferior infarct',
            '(?=inferior and|inferior\-|inferior \().*(?=lateral|anterior|inferior|posterior|antero|infero).*infarct',
            'inferior.*(?=lead|st|t).*(?=suggest|consistent with|consider|cannot rule out|'
            + 'cannot exclude|possible|represent|may be due to).*(?=inferior.*infarct|infarct)'
        ],
        'ILMI': [
            'inferolateral myocardial infarct',
            'inferolateral wall infarct',
            'inferolateral infarct',
            '(?=inferolateral and|inferolateral\-|inferolateral \().*(?=lateral|anterior|inferior|posterior|antero|infero).*infarct',
            'inferolateral.*(?=lead|st|t).*(?=suggest|consistent with|consider|cannot rule out|'
            + 'cannot exclude|possible|represent|may be due to).*(?=inferolateral.*infarct|infarct)'
        ],
        'IPMI': [
            'inferoposterior myocardial infarct',
            'inferoposterior wall infarct',
            'inferoposterior infarct',
            '(?=inferoposterior and|inferoposterior\-|inferoposterior \().*(?=lateral|anterior|inferior|posterior|antero|infero).*infarct',
            'inferoposterior.*(?=lead|st|t).*(?=suggest|consistent with|consider|cannot rule out|'
            + 'cannot exclude|possible|represent|may be due to).*(?=inferoposterior.*infarct|infarct)'
        ],
        'IPLMI': [
            'inferoposterolateral myocardial infarct',
            'inferoposterolateral wall infarct',
            'inferoposterolateral infarct',
            '(?=inferoposterolateral and|inferoposterolateral\-|inferoposterolateral \().*(?=lateral|anterior|inferior|posterior|antero|infero).*infarct',
            'inferoposterolateral.*(?=lead|st|t).*(?=suggest|consistent with|consider|cannot rule out|'
            + 'cannot exclude|possible|represent|may be due to).*(?=inferoposterolateral.*infarct|infarct)'
        ],
        'INJIN': [
            'inferior myocardial injury',
            'inferior injury',
            'inferior wall injury',
            '(?=inferior|infero|).*(?=st|t).*(?=suggest|consistent with|consider|cannot rule out|'
            + 'cannot exclude|repeat|possible|represent|may be due to).*(?=inferior.*injury|injury)',
            'injury.*(?=inferior.*t|inferior.*st)',
            '(?=st|t).*(?=suggest|consistent with|consider|cannot rule out|'
            + 'cannot exclude|possible|represent|may be due to).*inferior.*injury',
        ],
        'INJIL': [
            'inferolateral myocardial injury',
            'inferolateral injury',
            'inferolateral wall injury',
            'inferolateral.*(?=st|t).*(?=suggest|consistent with|consider|cannot rule out|'
            + 'cannot exclude|repeat|possible|represent|may be due to).*(?=inferolateral.*injury|injury)',
            'injury.*(?=inferolateral.*t|inferolateral.*st)',
            '(?=st|t).*(?=suggest|consistent with|consider|cannot rule out|'
            + 'cannot exclude|possible|represent|may be due to).*inferolateral.*injury',
        ],
        'AMI': [
            'anterior myocardial infarct',
            'anterior wall infarct',
            'anterior infarct',
            '(?=anterior and|anterior\-|anterior \().*(?=lateral|anterior|inferior|posterior|antero|infero).*infarct',
            'anterior.*(?=lead|st|t).*(?=suggest|consistent with|consider|cannot rule out|'
            + 'cannot exclude|possible|represent|may be due to).*(?=anterior.*infarct|infarct)',
        ],
        'ASMI': [
            'anteroseptal myocardial infarct',
            'anteroseptal wall infarct',
            'anteroseptal infarct',
            '(?=anteroseptal and|anteroseptal\-|anteroseptal \().*(?=lateral|anterior|inferior|posterior|antero|infero).*infarct',
            'anteroseptal.*(?=lead|st|t).*(?=suggest|consistent with|consider|cannot rule out|'
            + 'cannot exclude|possible|represent|may be due to).*(?=anteroseptal.*infarct|infarct)'
        ],
        'ALMI': [
            'anterolateral myocardial infarct',
            'anterolateral wall infarct',
            'anterolateral infarct',
            '(?=anterolateral and|anterolateral\-|anterolateral \().*(?=lateral|anterior|inferior|posterior|antero|infero).*infarct',
            'anterolateral.*(?=lead|st|t).*(?=suggest|consistent with|consider|cannot rule out|'
            + 'cannot exclude|possible|represent|may be due to).*(?=anterolateral.*infarct|infarct)',
        ],
        'INJAS': [
            'anteroseptal myocardial injury',
            'anteroseptal injury',
            'anteroseptal wall injury',
            'anteroseptal.*(?=st|t).*(?=suggest|consistent with|consider|cannot rule out|'
            + 'cannot exclude|repeat|possible|represent|may be due to).*(?=anteroseptal.*injury|injury)',
            'injury.*(?=anteroseptal.*t|anteroseptal.*st)',
            '(?=st|t).*(?=suggest|consistent with|consider|cannot rule out|'
            + 'cannot exclude|possible|represent|may be due to).*anteroseptal.*injury',
        ],
        'INJAL': [
            'anterolateral myocardial injury',
            'anterolateral injury',
            'anterolateral wall injury',
            'anterolateral.*(?=st|t).*(?=suggest|consistent with|consider|cannot rule out|'
            + 'cannot exclude|repeat|possible|represent|may be due to).*(?=anterolateral.*injury|injury)',
            'injury.*(?=anterolateral.*t|anterolateral.*st)',
            '(?=st|t).*(?=suggest|consistent with|consider|cannot rule out|'
            + 'cannot exclude|possible|represent|may be due to).*anterolateral.*injury',
        ],
        'INJLA': [
            'lateral myocardial injury',
            'lateral injury',
            'lateral wall injury',
            'lateral.*(?=st|t).*(?=suggest|consistent with|consider|cannot rule out|'
            + 'cannot exclude|repeat|possible|represent|may be due to).*(?=lateral.*injury|injury)',
            'injury.*(?=lateral.*t|lateral.*st)',
            '(?=st|t).*(?=suggest|consistent with|consider|cannot rule out|'
            + 'cannot exclude|possible|represent|may be due to).*lateral.*injury',
            '!anterolateral & inferolateral & posterolateral',
        ],
        'LMI': [
            'lateral myocardial infarct',
            'lateral wall infarct',
            'lateral infarct',
            '(?=lateral and|lateral\-|lateral \().*(?=lateral|anterior|inferior|posterior|antero|infero).*infarct',
            'lateral.*(?=lead|st|t).*(?=suggest|consistent with|consider|cannot rule out|'
            + 'cannot exclude|possible|represent|may be due to).*(?=lateral.*infarct|infarct)',
            '!anterolateral & inferolateral & posterolateral',
        ],
        'PMI': [
            'posterior myocardial infarct',
            'posterior wall infarct',
            'posterior infarct',
            '(?=posterior and|posterior\-|posterior \().*(?=lateral|anterior|inferior|posterior|antero|infero).*infarct',
            'posterior.*(?=lead|st|t).*(?=suggest|consistent with|consider|cannot rule out|'
            + 'cannot exclude|possible|represent|may be due to).*(?=posterior.*infarct|infarct)',
            '!inferoposterior'
        ],
        'NDT': [
            't wave abnormal & non(-?)specific',
            't abnormal & non(-?)specific',
            't flatten & non(-?)specific',
            't wave flatten & non(-?)specific',
            'flat t & non(-?)specific',
            '!st-t',
        ],
        'DIG': [
            'digitalis',
        ],
        'LNGQT': [
            'long qt interval',
            'long q-t interval',
            'long qtc interval',
            'long qt-c interval',
            'prolonged qt interval',
            'prolonged q-t interval',
            'prolonged qtc interval',
            'prolonged qt-c interval',
            'qt interval prolong',
            'q-t interval prolong',
            'qtc interval prolong',
            'qt-c interval prolong',
            'prolonged qt',
            'prolonged q-t',
            'long qt',
            'long q-t',
            'qt prolong',
            'q-t prolong',
            'qtc prolong',
            'qt-c prolong',
        ],
        'ANEUR': [
            'ventricular aneurysm',
        ],
        'EL': [
            'electrolyte',
            'drug effect',
        ],
        'NST_': [
            'st-t change',
            'st-t wave change',
            'st-t abnormal',
            'st-t wave abnormal',
            'st-t flatten',
            'st-t wave flatten',
            'flat st-t',
        ],
        'ISC_': [
            'nonspecific ische',
            'ische',
            '!lateral & anterior & inferior & posterior & antero & infero',
        ],
        'ISCAL': [
            'anterolateral ische',
            'anterolateral myocardial ische',
            'anterolateral.*(?=st|t).*(?=suggest|consistent with|consider|cannot rule out|'
            + 'cannot exclude|repeat|possible|represent|may be due to|'
            + 'could be secondary to).*(?=anterolateral.*ische|ische)',
            '(?=st|t).*(?=suggest|consistent with|consider|cannot rule out|'
            + 'cannot exclude|possible|represent|may be due to|'
            + 'could be secondary to).*anterolateral.*ische',
            'ische.*(?=anterolateral.*t|anterolateral.*st)',
        ],
        'ISCAS': [
            'anteroseptal ische',
            'anteroseptal myocardial ische',
            'anteroseptal.*(?=st|t).*(?=suggest|consistent with|consider|cannot rule out|'
            + 'cannot exclude|repeat|possible|represent|may be due to|'
            + 'could be secondary to).*(?=anteroseptal.*ische|ische)',
            '(?=st|t).*(?=suggest|consistent with|consider|cannot rule out|'
            + 'cannot exclude|possible|represent|may be due to|'
            + 'could be secondary to).*anteroseptal.*ische',
            'ische.*(?=anteroseptal.*t|anteroseptal.*st)',
        ],
        'ISCLA': [
            'lateral ische',
            'lateral myocardial ische',
            'lateral.*(?=st|t).*(?=suggest|consistent with|consider|cannot rule out|'
            + 'cannot exclude|repeat|possible|represent|may be due to|'
            + 'could be secondary to).*(?=lateral.*ische|ische)',
            '(?=st|t).*(?=suggest|consistent with|consider|cannot rule out|'
            + 'cannot exclude|possible|represent|may be due to|'
            + 'could be secondary to).*lateral.*ische',
            'ische.*(?=lateral.*t|lateral.*st)',
            '!anterolateral & inferolateral & posterolateral',
        ],
        'ISCAN': [
            'anterior ische',
            'anterior myocardial ische',
            'anterior.*(?=st|t).*(?=suggest|consistent with|consider|cannot rule out|'
            + 'cannot exclude|repeat|possible|represent|may be due to|'
            + 'could be secondary to).*(?=anterior.*ische|ische)',
            '(?=st|t).*(?=suggest|consistent with|consider|cannot rule out|'
            + 'cannot exclude|possible|represent|may be due to|'
            + 'could be secondary to).*anterior.*ische',
            'ische.*(?=anterior.*t|anterior.*st)',
        ],
        'ISCIN': [
            'inferior ische',
            'inferior myocardial ische',
            'inferior.*(?=st|t).*(?=suggest|consistent with|consider|cannot rule out|'
            + 'cannot exclude|repeat|possible|represent|may be due to|'
            + 'could be secondary to).*(?=inferior.*ische|ische)',
            '(?=st|t).*(?=suggest|consistent with|consider|cannot rule out|'
            + 'cannot exclude|possible|represent|may be due to|'
            + 'could be secondary to).*inferior.*ische',
            'ische.*(?=inferior.*t|inferior.*st)',
        ],
        'ISCIL': [
            'inferolateral ische',
            'inferolateral myocardial ische',
            'inferolateral.*(?=st|t).*(?=suggest|consistent with|consider|cannot rule out|'
            + 'cannot exclude|repeat|possible|represent|may be due to|'
            + 'could be secondary to).*(?=inferolateral.*ische|ische)',
            '(?=st|t).*(?=suggest|consistent with|consider|cannot rule out|'
            + 'cannot exclude|possible|represent|may be due to|'
            + 'could be secondary to).*inferolateral.*ische',
            'ische.*(?=inferolateral.*t|inferolateral.*st)',
        ],
        'ABQRS': [
            'qrs abnormal',
            'abnormal qrs',
        ],
        'PVC': [
            'ventricular premature complex',
            'ventricular premature contraction',
            'ventricular premature beat',
            'premature ventricular complex',
            'premature ventricular contraction',
            'premature ventricular beat',
        ],
        'STD_': [
            'st segment depression',
            'st depression',
        ],
        'VCLVH': [
            'voltage criteria for left ventricular hypertrophy',
            'borderline left ventricular hypertrophy',
        ],
        'QWAVE': [
            'q wave',
            'q in',
        ],
        'LOWT': [
            'low amplitude t',
            'low t',
        ],
        'NT_': [
            't change',
            't wave change',
            't wave flat'
            '!st-t',
        ],
        'PAC': [
            'atrial premature complex',
            'atrial premature contraction',
            'atrial premature beat',
            'premature atrial complex',
            'premature atrial contraction',
            'premature atrial beat',
        ],
        'LPR': [
            'pr interval prolong',
            'p-r interval prolong',
            'prolonged p-r interval',
            'long p-r interval',
            'p-r interval long',
            'pr interval long',
        ],
        'INVT': [
            't wave inversion',
            'inverted t',
            't waves are inverted',
            't wave inverted',
            't waves inverted',
        ],
        'LVOLT': [
            'low qrs voltage',
            'low qrs amplitude',
            'low limb lead qrs voltage',
            'low amplitude',
            'low voltage',
        ],
        'HVOLT': [
            'high qrs voltage',
            'high voltage',
            'prominent limb lead qrs voltage',
            'prominent qrs voltage',
            'prominent voltage',
        ],
        'TAB_': [
            'abnormal t',
            't wave abnormal',
            't abnormal',
            't flatten',
            't wave flatten',
            'flat t',
            '!st-t & nonspecific & non-specific',
        ],
        'STE_': [
            'st segment elevation',
            'st elevation',
        ],
        'PRC(S)': [
            'premature complex',
            'premature contraction',
            'premature beat',
            '!atrial & ventricular',
        ],
        'SR': [
            'sinus rhythm',
        ],
        'AFIB': [
            'atrial fibrillation',
            'afib',
        ],
        'AFLT': [
            'atrial flutter',
        ],
        'STACH': [
            'sinus tachycardia',
            'sinus tachycardic',
            'tachycardia',
            'tachycardic',
            '!supraventricular tachycardia & supraventricular tachycardic',
        ],
        'SVTAC': [
            'supraventricular tachycardia',
            'supraventricular tachycardic',
            '!paroxysmal supraventricular tachycardia & paroxysmal supraventricular tachycardic',
        ],
        'PSVT': [
            'paroxysmal supraventricular tachycardia',
            'paroxysmal supraventricular tachycardic',
        ],
        'SBRAD': [
            'sinus bradycardia',
            'sinus bradycardic',
            'bradycardia',
            'bradycardic',
        ],
        'SARRH': [
            'sinus arrhythmia',
            'arrhythmia',
            '!supraventricular arrhythmia',
        ],
        'SVARR': [
            'supraventricular arrhythmia',
        ],
        'PACE': [
            'pacemaker',
        ],
        'BIGU': [
            'bigeminal',
            'bigeminy',
        ],
        'TRIGU': [
            'trigeminal',
            'trigeminy',
        ],
    }

    code_to_superclass = {
        'NORM': 0,
        'LAFB': 1,
        'LPFB': 1,
        '1AVB': 1,
        '2AVB': 1,
        '3AVB': 1,
        'IVCD': 1,
        'WPW': 1,
        'IRBBB': 1,
        'ILBBB': 1,
        'CRBBB': 1,
        'CLBBB': 1,
        'LAO/LAE': 2,
        'RAO/RAE': 2,
        'LVH': 2,
        'RVH': 2,
        'SEHYP': 2,
        'IMI': 3,
        'ILMI': 3,
        'IPMI': 3,
        'IPLMI': 3,
        'INJIN': 3,
        'INJIL': 3,
        'AMI': 3,
        'ASMI': 3,
        'ALMI': 3,
        'INJAS': 3,
        'INJAL': 3,
        'INJLA': 3,
        'LMI': 3,
        'PMI': 3,
        'NDT': 4,
        'DIG': 4,
        'LNGQT': 4,
        'ANEUR': 4,
        'EL': 4,
        'NST_': 4,
        'ISC_': 4,
        'ISCAL': 4,
        'ISCAS': 4,
        'ISCLA': 4,
        'ISCAN': 4,
        'ISCIN': 4,
        'ISCIL': 4,
    }

    def get_form_index():
        return list(codebook.keys()).index('ABQRS')

    def get_superclass(
        y: np.array,
        codebook: Dict[str, List[str]] = codebook,
        code_to_superclass: Dict[str, int] = code_to_superclass,
    ):
        """
        Convert encoded diagnostic labels to their superclass.
        Note that 0:NORM, 1:CD, 2:HYP, 3:MI, 4:STTC by default.

        Args:
            y: np.array = encoded labels
            codebook: Dict[str, List[str]] = corresponding codebook for y
                the order of the encoded labels (y) should be matched with codebook
            code_to_superclass: Dict[str, int] = mapping directory to superclass
        """
        res = np.zeros((len(y), len(np.unique(list(code_to_superclass.values())))))
        total = res.copy()
        for i, code in enumerate(codebook.keys()):
            if code in code_to_superclass:
                """consider the shape of y is (N, 71)"""
                res[:, code_to_superclass[code]] += y[:, i]
                total[:, code_to_superclass[code]] += np.where(y[:,i] == 0, y[:,i], 1)

        res = np.divide(res, total, out=np.zeros_like(res), where=total!=0)
        return res

    def encode_scp_codes(scp_codes: pd.Series, codebook: Dict[str, List[str]] = codebook):
        scp_codes = scp_codes.apply(lambda x: eval(x))
        y = dict()
        w = dict()
        for code in codebook.keys():
            w[code] = scp_codes.apply(lambda x: 1.0 if code not in x else x[code] / 100 if x[code] != 0 else 1.0).to_list()
            y[code] = scp_codes.apply(lambda x: 1.0 if code in x else 0.0).to_list()

        return pd.DataFrame(y).values, pd.DataFrame(w).values

    def decode_scp_codes(multi_hot_code: np.array, codebook: Dict[str, List[str]] = codebook):
        decoded = dict()
        for i, code in enumerate(codebook.keys()):
            if multi_hot_code[i] > 0:
                decoded[code] = int(multi_hot_code[i] * 100)

        return decoded

    def extract_scp_codes(notes: pd.Series, codebook: Dict[str, List[str]] = codebook) -> pd.DataFrame:
        y = dict()
        for i, (code, keywords) in enumerate(codebook.items()):
            including = r''
            excluding = r''

            for key in keywords:
                if '&' in key:
                    keys = key.split('&')
                    keys = [s.strip() for s in keys]

                    if not key.startswith('!'):
                        for k in keys:
                            including += f'(?=.*{k})'
                        including += '|'
                    else:
                        keys[0] = keys[0].replace('!', '').strip()
                        for k in keys:
                            excluding  += f'{k}|'
                else:
                    if not key.startswith('!'):
                        including += f'{key}|'
                    else:
                        key = key.split('!')[-1].strip()
                        excluding += key
                        excluding += '|'

            including = including[:-1]
            true_or_false = notes.str.contains(including)
            if len(excluding) > 0:
                excluding = excluding[:-1]
                try:
                    true_or_false &= ~notes.str.contains(excluding)
                except:
                    breakpoint()
                    # check notes[notes.isnull()]

            y[code] = true_or_false.to_list()

        return pd.DataFrame(y).astype(float)

    ptbxl_database = pd.read_csv(os.path.join(ptbxl_dir, 'ptbxl_database.csv'))
    df = ptbxl_database[ptbxl_database['validated_by_human']]
    df = df[df['age'].notnull()]
    df['report'] = df['report'].map(lambda x: x.replace('ekg', 'ecg').replace('.', '').strip())

    pts = pd.read_csv(os.path.join(mimic_dir, 'PATIENTS.csv'))
    adms = pd.read_csv(os.path.join(mimic_dir, 'ADMISSIONS.csv'))

    if os.path.exists('results/ecgnotes_with_hadm_id.csv'):
        notes = pd.read_csv('results/ecgnotes_with_hadm_id.csv')
    else:
        notes = pd.read_csv(os.path.join(mimic_dir, 'NOTEEVENTS.csv'))
        notes = notes[notes['CATEGORY'] == 'ECG']
        notes = notes[notes['HADM_ID'].notnull()]
        notes['TEXT'] = notes['TEXT'].map(lambda x: x.lower().replace('\n', ' ').replace('.', '').strip())
        notes = notes[notes['TEXT'].notnull()]
        pattern = r'since|compared'
        notes['report'] = notes.apply(lambda x: re.split(pattern, x['TEXT'])[0], axis=1)
        notes = notes[notes['report'].str.strip() != ""]
        if not os.path.exists('results'):
            os.mkdir('results')
        notes.to_csv('results/ecgnotes_with_hadm_id.csv', index=False)

    adms = adms[adms['HADM_ID'].isin(notes.HADM_ID.unique())]
    merged = pd.merge(adms, pts[['SUBJECT_ID', 'DOB', 'GENDER']], on='SUBJECT_ID', how='left')

    def calculate_age(birth: datetime, now: datetime):
        age = now.year - birth.year
        if now.month < birth.month:
            age -= 1
        elif (now.month == birth.month) and (now.day < birth.day):
            age -= 1

        return age

    merged['AGE'] = merged.apply(
        lambda x: calculate_age(
            datetime.strptime(x['DOB'], '%Y-%m-%d %H:%M:%S'),
            datetime.strptime(x['ADMITTIME'], '%Y-%m-%d %H:%M:%S')
        ), axis=1
    )

    merged = merged[merged['AGE'] < 300]

    df.rename(columns={'age': 'AGE', 'sex': 'GENDER'}, inplace=True)
    df['GENDER'] = df.apply(lambda x: 'M' if x['GENDER'] == 0 else 'F', axis=1)
    df['AGE'] = df.apply(lambda x: int(x['AGE']), axis=1)

    df['AGE'] = df.apply(lambda x: int(x['AGE'] / 10) * 10, axis=1)
    merged['AGE'] = merged.apply(lambda x: int(x['AGE'] / 10) * 10, axis=1)

    uniques = np.unique(
        np.array(
            list(df[['AGE', 'GENDER']].to_records(index=False))
        )
    )

    # def construct_sim_matrix(ptbxl:list, mimic_iii:list, metric='bleu', subword=False, eps=1e-6):
    #     if not getattr(construct_sim_matrix, 'tokenizer', None):
    #         construct_sim_matrix.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    #     if metric == 'bleu':
    #         if subword:
    #             ptbxl = list(map(lambda x: construct_sim_matrix.tokenizer.tokenize(x), ptbxl))
    #             mimic_iii = list(map(lambda x: [construct_sim_matrix.tokenizer.tokenize(x)], mimic_iii))
    #             weights = (0.5, 0.5)
    #         else:
    #             ptbxl = list(map(lambda x: x.split(), ptbxl))
    #             mimic_iii = list(map(lambda x: [x.split()], mimic_iii))
    #             weights = (0.5, 0.5)
    #         scores = []
    #         for m in mimic_iii:
    #             s = [sentence_bleu(m, ptbxl[i], weights=weights) for i in range(len(ptbxl))]
    #             s = list(map(lambda x: x if x >= eps else 0, s))
    #             scores.append(s)
    #     elif metric == 'bert':
    #         pass

    #     return np.array(scores).T

    skipped = 0
    pending = 0
    mapped_total = []
    total_mimic_candidates = []
    for i, (age, gender) in enumerate(uniques):
        print(f"{i+1}/{len(uniques)}")
        hadms = merged[(merged['AGE'] == age) & (merged['GENDER'] == gender)]
        ptbxl = df[(df['AGE'] == age) & (df['GENDER'] == gender)]
        if not hadms.empty:
            mimic = pd.merge(hadms, notes[['HADM_ID', 'CHARTDATE', 'TEXT', 'report']], on='HADM_ID', how='left')
            key = list(
                mimic.groupby(mimic['HADM_ID'])['CHARTDATE'].min().items()
            )
            mimic = mimic[
                mimic[['HADM_ID', 'CHARTDATE']].agg(tuple, axis=1).isin(key)
            ]

            survivors = []
            if mimic.duplicated(['HADM_ID']).any():
                duplicated = mimic[mimic.duplicated(['HADM_ID'], keep=False)]
                for dup_hadm_id in duplicated['HADM_ID'].unique():
                    victim = duplicated[duplicated['HADM_ID'] == dup_hadm_id]
                    survivor = victim[
                        victim['TEXT'].str.lower().str.contains('no previous tracing|no previous report|tracing #1')
                        & ~victim['TEXT'].str.lower().str.contains('compared to tracing #1|compared tracing #1')
                    ]
                    if survivor.empty:
                        survivor = victim[victim['TEXT'].str.lower().str.contains('tracing #\d+')]
                        if survivor.empty:
                            skipped += 1
                            continue
                        else:
                            tracing = [re.findall(r'tracing #\d+', text)[-1] for text in survivor.TEXT]
                            idx = np.argmin(tracing)
                            survivor = survivor.iloc[[idx]]
                    elif len(survivor) > 1:
                        skipped += 1
                        continue

                    survivors.append(survivor)

            if len(survivors) > 0:
                survivors = pd.concat(survivors)
                mimic = pd.concat([mimic.drop_duplicates(['HADM_ID'], keep=False), survivors])
            pending += len(mimic)

            total_mimic_candidates.append(mimic)

            mimic_y = extract_scp_codes(mimic['report']).values
            ptbxl_y, ptbxl_w = encode_scp_codes(ptbxl['scp_codes'])
            mimic_super = get_superclass(mimic_y)
            ptbxl_super = get_superclass(ptbxl_y)

            #################################################################################################################################
            # calc hamming distance considering superclass

            # method #0
            # distance_matrix = np.array(
            #     [[distance.hamming(ptbxl_y[i], mimic_y[j], ptbxl_w[i]) for j in range(len(mimic_y))] for i in range(len(ptbxl_y))]
            # )
            # distance_matrix += np.array(
            #     [[distance.hamming(ptbxl_super[i], mimic_super[j]) for j in range(len(mimic_super))] for i in range(len(ptbxl_super))]
            # )

            # method #1
            ptbxl_y = np.concatenate((ptbxl_y, ptbxl_super), axis=-1)
            ptbxl_w = np.concatenate((ptbxl_w, np.ones(ptbxl_super.shape)), axis=-1)
            mimic_y = np.concatenate((mimic_y, mimic_super), axis=-1)
            distance_matrix = np.array(
                [[distance.hamming(ptbxl_y[i], mimic_y[j], ptbxl_w[i]) for j in range(len(mimic_y))] for i in range(len(ptbxl_y))]
            )
            #################################################################################################################################

            form_idx = get_form_index()
            mask = np.zeros(distance_matrix.shape)
            for i in range(len(mask)):
                for j in range(len(mask[0])):
                    mask[i][j] = (
                        1 if (
                            not set(np.where(mimic_y[j, form_idx:])[0]).issubset(set(np.where(ptbxl_y[i, form_idx:])[0]))
                            or not mimic_y[j].any()
                        )
                        else 0
                    )
            distance_matrix = np.ma.array(distance_matrix, mask=mask).filled(fill_value=100)

            _ptbxl = [f'p{i}' for i in range(len(ptbxl_y))]
            _mimic = [f'm{i}' for i in range(len(mimic_y))]

            G = nx.Graph()
            G.add_nodes_from(_ptbxl, bipartite=0, label='ptbxl')
            G.add_nodes_from(_mimic, bipartite=1, label='mimic')

            for i, p in enumerate(_ptbxl):
                for j, m in enumerate(_mimic):
                    G.add_edge(p, m, weight=distance_matrix[i][j])

            min_matched = mx.minimum_weight_full_matching(G)
            p = []
            m = []
            distances = []
            for k, v in min_matched.items():
                if not 'p' in k:
                    continue
                p.append(int(k[1:]))
                m.append(int(v[1:]))
                distances.append(distance_matrix[p[-1]][m[-1]])

            mapped = mimic.iloc[m][['SUBJECT_ID', 'HADM_ID', 'CHARTDATE', 'report']]
            mapped['mimic_scp_codes'] = [str(decode_scp_codes(y)) for y in mimic_y[m]]
            mapped['ptbxl_id'] = ptbxl.iloc[p]['ecg_id'].to_list()
            mapped['ptbxl_report'] = ptbxl.iloc[p]['report'].to_list()
            mapped['ptbxl_scp_codes'] = ptbxl.iloc[p]['scp_codes'].to_list()
            mapped['distance'] = distances

            mapped_total.append(mapped)

            # ptbxl_notes = ptbxl['report'].to_list()
            # mimic_notes = mimic['report'].to_list()
            # sim_matrix = construct_sim_matrix(ptbxl_notes, mimic_notes, subword=False)


    # total_mimic_candidates = pd.concat(total_mimic_candidates)
    # total_mimic_candidates.to_csv('total_mimic_candidates.csv', index=False)

    mapped_total = pd.concat(mapped_total)
    mapped_total.rename(columns={'ptbxl_id': 'ecg_id', 'report': 'mimic_report'}, inplace=True)
    mapped_ptbxl = pd.merge(ptbxl_database, mapped_total, on="ecg_id", how="inner")
    mapped_ptbxl = mapped_ptbxl[mapped_ptbxl["distance"] < 100]

    if not os.path.exists('results'):
        os.mkdir('results')
    mapped_ptbxl.to_csv(os.path.join('results', 'mapped_ptbxl.csv'), index=False)

    return mapped_ptbxl
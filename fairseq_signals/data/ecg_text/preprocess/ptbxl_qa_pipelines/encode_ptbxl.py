import os
import re
import math

import numpy as np
import pandas as pd
import wfdb
import neurokit2 as nk

def encode_ptbxl(ptbxl_dir, ptbxl_database):
    pd.set_option('mode.chained_assignment', None)

    leads = ['i','ii','iii','avr','avl','avf','v1','v2','v3','v4','v5','v6']

    lead_positions_patterns = r'v leads|chest|limb|anterior|antero\-?\s?lateral|antero\-?\s?septal|inferior|infero\-?\s?lateral|infero\-?\s?septal|lateral|high\-?\s?lateral|precordial|peripheral|standard|lateral\-?\s?chest'
    lead_patterns = r'(((v?\d)|(lead)?\s?(?<=[^A-Za-z])iii(?=[^A-Za-z])|(lead)?\s?(?<=[^A-Za-z])ii(?=[^A-Za-z])|(lead)?\s?(?<=[^A-Za-z])i(?=[^A-Za-z])|(lead)?\s?avr|(lead)?\s?avl|(lead)?\s?avf|v leads|all leads|chest( lead(s)?)?|limb( lead(s)?)?|anterior( lead(s)?)?|antero\-?\s?lateral( lead(s)?)?|antero\-?\s?septal( lead(s)?)?|inferior( lead(s)?)?|infero\-?\s?lateral( lead(s)?)?|infero\-?\s?septal( lead(s)?)?|lateral( lead(s)?)?|high\-?\s?lateral( lead(s)?)?|precordial( lead(s)?)?|prepheral( lead(s)?)?|standard( lead(s)?)?|infero\-lateral( lead(s)?)?|lateral\-?\s?chest( lead(s)?)?)\s?([/\),\.\-\&\s])?\s?(and)?\s?)+'
    positions_to_leads = {
        'chest': ['v1', 'v2', 'v3', 'v4', 'v5', 'v6'],
        'vleads': ['v1', 'v2', 'v3', 'v4', 'v5', 'v6'],
        'limb': ['i', 'ii', 'iii', 'avr', 'avl', 'avf'],
        'anterior': ['v3', 'v4'],
        'anterolateral': ['i', 'avl', 'v3', 'v4', 'v5', 'v6'],
        'anteroseptal': ['v1', 'v2', 'v3', 'v4'],
        'inferior': ['ii', 'iii', 'avf'],
        'inferolateral': ['i', 'ii', 'iii', 'avl', 'avf', 'v5', 'v6'],
        'inferoseptal': ['ii', 'iii', 'avf', 'v1', 'v2'],
        'lateral': ['i', 'avl', 'v5', 'v6'],
        'highlateral': ['i', 'avl'],
        'precordial': ['v1', 'v2', 'v3', 'v4', 'v5', 'v6'],
        "prephreal": ['i', 'ii', 'iii', 'avr', 'avl', 'avf'],
        'standard': ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6'],
        'lateralchest': ['i', 'avl', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
    }

    scp_to_parse_lead = ['NDT', 'NST_', 'DIG', 'STD_', 'VCLVH', 'QWAVE', 'LOWT', 'NT_', 'INVT', 'LVOLT', 'HVOLT', 'TAB_', 'STE_']

    #XXX backup
    # lead patterns for each form statement
    # [0]: first priority patterns. if not matched, pass to [1]
    # [1]: if [1][0] is matched, try matching [1][1]
    # [2]: if any patterns are not matched, try matching [2]
    # patterns = {
    #     'NDT': [
    #         r'(t wave|t wve|t change neg|t-change neg|t abnormal|t sinus rhythm abnormal|t flat|t flach|flat t|'
    #         r't biphasic|biphasic t|t change|t-change|t flattened|t negative|t neg|neg t|'
    #         r't term\. neg|t high|high t|t wave flat|t wave changes).* in (' + lead_patterns +'),?'
    #         r'( (neg|t neg|inferior flattened|biphasic|t flattened|flat t|flach) in (' + lead_patterns + '))?',
    #         (
    #             r't waves are.* in these leads',
    #             r'(low limb lead voltage)|(st segments are depressed in (' + lead_patterns + '))'
    #         ),
    #         [
    #             r'((st segment depression|st depression|(st segments are.* depressed)|st lowering|'
    #             r'st-lowering|st reduction|st-senkung|st-thinking).* (in|above) ('
    #             + lead_patterns + r'))|((st depression|st-lowering|st lowering|st reduction) ('
    #             + lead_patterns + r'))',
    #             r'((st elevation|st-elevation).* (in|over|discrete) (' + lead_patterns + r'))|'
    #             r'((st elevation|st-elevation|st-hebung in) (' + lead_patterns + r'))',
    #             r'((st-t wave change|st-t change|t change).* in (' + lead_patterns + '))|((' +
    #             lead_positions_patterns + ') (st-t wave|st.t|t) changes)|(st-t changes ' + lead_patterns + ')',
    #             r'(((t changes|t-changes|t- changes):? (neg|high|flat|biphas|excessive|biphas).* ('
    #             + lead_patterns + r'))(, (flat|high|neg) in (' + lead_patterns + r'))?)|'
    #             r'((t-changes in|t-changes|t abnormal in) (' + lead_patterns + r'))',
    #         ]
    #     ],
    #     'NST_': [
    #         r'((st-t wave change|st-t change|t change).* in (' + lead_patterns + '))|((' +
    #         lead_positions_patterns + ') (st-t wave|st.t|t) (changes|abnormal))|(st-t changes ' + lead_patterns + ')',
    #         (

    #         ),
    #         [
    #             r'((st segment depression|st depression|(st segments are.* depressed)|st lowering|'
    #             r'st-lowering|st reduction|st-senkung|st-thinking).* (in|above) ('
    #             + lead_patterns + r'))|((st depression|st-lowering|st lowering|st reduction) ('
    #             + lead_patterns + r'))',
    #             r'((st elevation|st-elevation).* (in|over|discrete) (' + lead_patterns + r'))|'
    #             r'((st elevation|st-elevation|st-hebung in) (' + lead_patterns + r'))',
    #             r'(((t changes|t-changes|t- changes):? (neg|high|flat|biphas|excessive|biphas).* ('
    #             + lead_patterns + r'))(, (flat|high|neg) in (' + lead_patterns + r'))?)|'
    #             r'((t-changes in|t-changes|t abnormal in) (' + lead_patterns + r'))',
    #             r'(t wave flattening.* in (' + lead_patterns + r'))|'
    #             r'(t wave changes in (' + lead_patterns + r'))',
    #             r'(t wave|t wve|t change neg|t-change neg|t abnormal|t sinus rhythm abnormal|t flat|t flach|flat t|'
    #             r't biphasic|biphasic t|t changes|t-changes|t flattened|t negative|t neg|neg t|'
    #             r't term\. neg|t high|high t|t wave flat|t wave changes).* in (' + lead_patterns +'),?'
    #             r'( (neg|t neg|inferior flattened|biphasic|t flattened|flat t|flach) in (' + lead_patterns + '))?',
    #         ]
    #     ],
    #     'DIG': [
    #         r'digitalis.* (in|trough|changes) (' + lead_patterns + ')',
    #         (

    #         ),
    #         [
    #             r'((st segment depression|st depression|(st segments are.* depressed)|st lowering|'
    #             r'st-lowering|st reduction|st-senkung|st-thinking).* (in|above) ('
    #             + lead_patterns + r'))|((st depression|st-lowering|st lowering|st reduction) ('
    #             + lead_patterns + r'))',
    #             r'((st elevation|st-elevation).* (in|over|discrete) (' + lead_patterns + r'))|'
    #             r'((st elevation|st-elevation|st-hebung in) (' + lead_patterns + r'))',
    #         ]
    #     ],
    #     'STD_': [
    #         r'((st segment depression|st decrease|st depression|(st segments are.* depressed)|st lowering|'
    #         r'st-lowering|st reduction|st-senkung|st-thinking).* (in|above) ('
    #         + lead_patterns + r'))|((st depression|st-lowering|st lowering|st reduction) ('
    #         + lead_patterns + r'))',
    #         (

    #         ),
    #         [

    #         ]
    #     ],
    #     'VCLVH': [
    #         r'(voltages are high in (' + lead_patterns + r'))|((' 
    #         + lead_patterns + r')\s?voltages suggest possible lv hypertrophy)|'
    #         r'(left ventricular hypertrophy are satisfied in (' + lead_patterns +'))|'
    #         r'(voltages in (' + lead_patterns + r')\s?are at upper limit)|'
    #         r'(voltages in (' + lead_patterns + r')\s?of left ventricular hypertrophy)|'
    #         r'(r wave height in (' +lead_patterns + r')\s?suggests the possibility of left ventricular hypertrophy)|'
    #         r'((' + lead_positions_patterns + r') voltages of left ventricular hypertrophy)',
    #         (

    #         ),
    #         [

    #         ]
    #     ],
    #     'QWAVE': [
    #         r'(q wave.* in (' + lead_patterns + '))|(q in (' + lead_patterns + r'))|'
    #         r'(small (' + lead_positions_patterns + ') q waves noted)',
    #         (

    #         ),
    #         [

    #         ]
    #     ],
    #     'LOWT': [
    #         r't waves are low.* in (' + lead_patterns + ')',
    #         (
    #             r't waves are.*low.* in these leads',
    #             r'st segments are depressed in (' + lead_patterns + ')'
    #         ),
    #         [

    #         ]
    #     ],
    #     'NT_': [
    #         r'(t wave flattening.* in (' + lead_patterns + r'))|'
    #         r'(t wave changes in (' + lead_patterns + r'))',
    #         (

    #         ),
    #         [

    #         ]
    #     ],
    #     'INVT': [
    #         r'((t waves are inverted|t wave inversion|t waves inverted) in ('
    #         + lead_patterns + r'))|'
    #         r'(t wave flattening.* and inverted in (' + lead_patterns + r'))',
    #         (
    #             r't waves are inverted.* in these leads',
    #             r'st segments are depressed in (' + lead_patterns + ')'
    #         ),
    #         [

    #         ]
    #     ],
    #     'LVOLT': [
    #         r'(low limb lead voltage|peripheral low voltage|peripheral low-voltage|'
    #         r'peripheral low tension|low voltage in (' + lead_patterns + '))',
    #         (

    #         ),
    #         r''
    #     ],
    #     'HVOLT': [
    #         r'(high v lead voltages)|'
    #         r'(voltages in (' + lead_patterns + r')\s?are at upper limit)|'
    #         r'(voltages are high in (' + lead_patterns + r'))',
    #         (

    #         ),
    #         [

    #         ]
    #     ],
    #     'TAB_': [
    #         r'(((t changes|t-changes|t- changes):? (neg|high|flat|biphas|excessive|biphas).* ('
    #         + lead_patterns + r'))(, (flat|high|neg) in (' + lead_patterns + r'))?)|'
    #         r'((t-changes in|t-changes|t abnormal in) (' + lead_patterns + r'))',
    #         (

    #         ),
    #         [

    #         ]
    #     ],
    #     'STE_': [
    #         r'((st elevation|st-elevation).* (in|over|discrete) (' + lead_patterns + r'))|'
    #         r'((st elevation|st-elevation|st-hebung in) (' + lead_patterns + r'))',
    #         (

    #         ),
    #         [

    #         ]
    #     ]
    # }
    patterns = {
        'NDT': [
            r"(t wave(s)?|t waves are|t wve|t-change(s)?(:)?|t change(s)?(:)?|flat t|biphasic t|"
            + r"neg t|high t)\s*"
            + r"(are)?\s*"
            + r"(neg in|flattening in|flattening or slight inversion in|flat in|changes in|"
            + r"inversion in|low or flat in|generally low and are flat in|low in|flattened in|"
            + r"now generally flatter and are slightly inverted in|inverted in|"
            + r"slightly inverted in|low or flat in|flat or slightly inverted in|"
            + r"now slightly inverted in|now inverted in|biphasic in|abnormal in|"
            + r"abnormality in|sinus rhythm abnormal in|flat|flach in|in|biphas.|"
            + r"neg\. in|biphasic|negative in|neg in|term. neg in|high in|)\s*"
            + r"(" + lead_patterns + r")",
            (
            ),
            [
            ]
        ],
        'NST_': [
            r"((st-t wave changes|st-t changes|t changes)\s*"
            + r"(are marked in|persist in|in|)\s*"
            + r"(" + lead_patterns + r"))|"
            + r"((" + lead_positions_patterns + r")\s*"
            + r"(st-t wave changes|st-t changes))",
            (
            ),
            [
            ]
        ],
        'DIG': [
            r"(digitalis)\s*"
            + r"(t pointed in|change in|changes|change trough|change r trough|in|)\s*"
            + r"(" + lead_patterns + r")",
            (
            ),
            [
            ]
        ],
        'STD_': [
            r"(st segment(s)?|st(\-)?st-senkung|st-thinking)\s*"
            + r"(are)?\s*"
            + r"(depression in|depression and t wave flattening in|depressed in|"
            + r"depressed and t wave.{0,15} in|depression|depression above|lowering in|"
            + r"lowering|reduction in|reduction discrete in|in)\s*"
            + r"(" + lead_patterns + r")",
            (
            ),
            [

            ]
        ],
        'VCLVH': [
            r"(voltages are high in (" + lead_patterns + r") suggesting lvh)|"
            + r"((" + lead_patterns + r") voltages suggest possible lv hypertrophy)|"
            + r"(left ventricular hypertrophy are satisfied in (" + lead_patterns + r"))|"
            + r"(voltages in (" + lead_patterns + r") are at upper limit)|"
            + r"(voltages in (" + lead_patterns + r") of left ventricular hypertrophy)|"
            + r"(r wave height in (" + lead_patterns + r") suggests the possibility of left ventricular hypertrophy)",
            (
            ),
            [

            ]
        ],
        'QWAVE': [
            r"((q wave(s)?|q)\s*"
            + r"(are)?\s*"
            + r"(present in|in|(and|,).{0,55} in)\s*"
            + r"(" + lead_patterns + r"))|"
            + r"((" + lead_positions_patterns +r") q waves noted)",
            (
            ),
            [
            ]
        ],
        'LOWT': [
            r"(t waves are)\s*"
            + r"(low or flat in|low in)\s*"
            + r"(" + lead_patterns + r")",
            (
            ),
            [
            ]
        ],
        'NT_': [
            r"(t wave)\s*"
            + r"(flattening in|flattening persists in|flattening or slight inversion in|"
            + r"flattening or inversion in|changes in)\s*"
            + r"(" + lead_patterns + r")",
            (
            ),
            [
            ]
        ],
        'INVT': [
            r"(t wave(s)?)\s*"
            + r"(are)?\s*"
            + r"(inverted in|inversion in|flattening in.{0,30} and inverted in)\s*"
            + r"(" + lead_patterns + r")",
            (
            ),
            [
            ]
        ],
        'LVOLT': [
            r"(low limb lead voltage|peripheral low voltage|peripheral low-voltage|"
            + r"peripheral low tension|low voltage in (" + lead_patterns + r"))",
            (
            ),
            [
            ]
        ],
        'HVOLT': [
            r"(high v lead voltages)|"
            + r"(voltages in (" + lead_patterns + r")\s*are at upper limit)|"
            + r"(voltages are high in (" + lead_patterns + r"))",
            (
            ),
            [
            ]
        ],
        'TAB_': [
            # r'(((t changes|t-changes|t- changes):? (neg|high|flat|biphas|excessive|biphas).* ('
            # + lead_patterns + r'))(, (flat|high|neg) in (' + lead_patterns + r'))?)|'
            # r'((t-changes in|t-changes|t abnormal in) (' + lead_patterns + r'))',
            r"(t(\-)?\s?change(s)?(:)?|t abnormal)\s*"
            + r"(negative in|neg in|neg t|neg\. t in|neg\. in|high in|high t in|"
            + r"flat in|biphas|biphas\.|biphas\. in|excessive|in|)\s*"
            + r"(" + lead_patterns + r")",
            (
            ),
            [
            ]
        ],
        'STE_': [
            # r'((st elevation|st-elevation).* (in|over|discrete) (' + lead_patterns + r'))|'
            # r'((st elevation|st-elevation|st-hebung in) (' + lead_patterns + r'))',
            r"(st\-?\s?elevation|st-hebung)\s*"
            + r"(in|discrete in|over|discrete|)\s*"
            + r"(" + lead_patterns + r")",
            (
            ),
            [
            ]
        ]
    }


    ptbxl_database['baseline_drift'] = ptbxl_database['baseline_drift'].fillna('').str.lower().values
    ptbxl_database['static_noise'] = ptbxl_database['static_noise'].fillna('').str.lower().values
    ptbxl_database['burst_noise'] = ptbxl_database['burst_noise'].fillna('').str.lower().values
    ptbxl_database['electrodes_problems'] = ptbxl_database['electrodes_problems'].fillna('').str.lower().values
    ptbxl_database['extra_beats'] = ptbxl_database['extra_beats'].fillna('').str.lower().values

    def parse_lead_positions(str: str, stmt=None):
        def _refine(s):
            s = s.strip()
            s = s.replace('!', '1')
            s = s.replace('11', 'ii')
            s = s.replace('111', 'iii')
            s = s.replace('alf', 'avf')
            if s.isdigit():
                s = 'v' + s
            if (
                any([p in s for p in leads])
                or 'all' in s
                or re.search(lead_positions_patterns, s)
            ):
                return s
            else:
                return ''
        parsed_leads = re.split(r';|,|\.|\?|\)| and |\s', str)
        for _ in range(2):
            for i, p in enumerate(parsed_leads):
                if p.endswith('-'):
                    parsed_leads[i] = parsed_leads[i] + parsed_leads[i+1]
                elif p.startswith('-'):
                    parsed_leads[i] = parsed_leads[i-1] + parsed_leads[i]
        parsed_leads = list(map(_refine, parsed_leads))

        parsed = []
        for l in parsed_leads:
            if l.isdigit():
                l = 'v' + l

            if x := re.search(lead_positions_patterns, l):
                x = x.group().replace('-', '').replace(' ', '')
                parsed.extend(positions_to_leads[x])
            elif not '-' in l:
                if l in leads:
                    parsed.append(l)
                elif 'alle' in l:
                    parsed = leads.copy()
                    break
                elif ' ' in l:
                    l = l.split()
                    for _l in l:
                        if _l in leads:
                            parsed.append(_l)
                        elif 'alle' in _l:
                            parsed = leads.copy()
                            break
            else:
                l = re.split('--|-', l)
                l = list(map(lambda x: x.strip(), l))
                l = list(map(lambda x: 'v' + x if x.isdigit() else x, l))
                start = leads.index(l[0])
                end = leads.index(l[1]) + 1
                parsed.extend(leads[start:end])

        parsed = list(set(parsed))
        parsed = np.array(
            list(map(lambda x: leads.index(x), parsed))
        )
        res = np.zeros(len(leads)).astype(np.int8)
        if len(parsed) == 0:
            return None

        res[parsed] = 1
        return res

    def parse_for_es(str, id=None):
        if len(str) > 0 and str[0].isdigit():
            if not (
                (len(str) > 1) and (
                    str[1:].startswith('es')
                    or str[1:].startswith('ves')
                    or str[1:].startswith('sves')
                )
            ) and not (
                (len(str) > 2) and (
                    str[2:].startswith('es')
                    or str[2:].startswith('ves')
                    or str[2:].startswith('sves')
                )
            ):
                str = 'es' + str

        leads = parse_lead_positions(str)
        if isinstance(leads, float):
            leads = None

        es_dict = {
            'es': -1,
            'ves': -1,
            'sves': -1,
        }

        es_list = []
        str = re.split(',|\(|\)|;', str)
        str = list(map(lambda x: x.strip(), str))
        for s in str:
            if 'es' in s:
                es_list.append(s)

        if 'alles' in es_list:
            es_list.remove('alles')

        for es in es_list:
            if es[0].isdigit():
                if es[1].isdigit():
                    es_dict[es[2:]] = int(es[:2])
                else:
                    es_dict[es[1:]] = int(es[0])
            elif es[-1].isdigit():
                if es[-2].isdigit():
                    es_dict[es[:-2]] = int(es[-2:])
                else:
                    es_dict[es[:-1]] = int(es[-1])
            else:
                es_dict[es] = 0

        es = np.array(list(es_dict.values())).astype(np.int8)

        output = [leads if x >= 0 else None for x in es]
        output.extend(es)
        assert len(output) == 6, (
            str, output
        )
        return output

    ptbxl_database['DRIFT'] = ptbxl_database.apply(lambda x: parse_lead_positions(x['baseline_drift']), axis=1)
    ptbxl_database['STATIC'] = ptbxl_database.apply(lambda x: parse_lead_positions(x['static_noise']), axis=1)
    ptbxl_database['BURST'] = ptbxl_database.apply(lambda x: parse_lead_positions(x['burst_noise']), axis=1)
    ptbxl_database['ELECT'] = ptbxl_database.apply(lambda x: parse_lead_positions(x['electrodes_problems']), axis=1)
    noises = ['DRIFT', 'STATIC', 'BURST', 'ELECT']
    any_noise = []
    for i in range(12):
        idx = ptbxl_database.apply(lambda _: False, axis=1)
        for n in noises:
            idx |= ptbxl_database.apply(lambda x: x[n][i] if isinstance(x[n], np.ndarray) else 0, axis=1)
        any_noise.append(np.array(idx.astype(np.int8).tolist()))
    any_noise = np.stack(any_noise, axis=-1)
    any_noise = pd.Series(list(any_noise), index=ptbxl_database.index)
    any_noise = any_noise.apply(lambda x: None if x.sum() == 0 else x)
    ptbxl_database['ANY_NOISE'] = any_noise

    EXTRA = ptbxl_database.apply(lambda x: parse_for_es(x['extra_beats'], x['ecg_id']), axis=1)
    ptbxl_database['ES'] = pd.Series([x[0] for x in EXTRA], index=EXTRA.index)
    ptbxl_database['VES'] = pd.Series([x[1] for x in EXTRA], index=EXTRA.index)
    ptbxl_database['SVES'] = pd.Series([x[2] for x in EXTRA], index=EXTRA.index)
    ptbxl_database['ES_COUNT'] = pd.Series([x[3] for x in EXTRA], index=EXTRA.index)
    ptbxl_database['VES_COUNT'] = pd.Series([x[4] for x in EXTRA], index=EXTRA.index)
    ptbxl_database['SVES_COUNT'] = pd.Series([x[5] for x in EXTRA], index=EXTRA.index)
    extra = ['ES', 'VES', 'SVES']
    idx = ptbxl_database.apply(lambda _: False, axis=1)
    for e in extra:
        idx |= ptbxl_database[e].apply(lambda x: True if isinstance(x, np.ndarray) else False)

    ptbxl_database['ANY_EXTRA'] = False
    ptbxl_database['ANY_EXTRA'][idx] = True

    ptbxl_database['PACE'] = ptbxl_database['scp_codes'].str.contains("'PACE'", regex=False)
    # ptbxl_database["PACE"] = ptbxl_database["pacemaker"].notnull()

    ptbxl_database['MI_NAN'] = (
        ptbxl_database['infarction_stadium1'].isna()
        & ptbxl_database['infarction_stadium2'].isna()
    )
    ptbxl_database['MI_UNK'] = ptbxl_database['infarction_stadium1'] == 'unknown'
    ptbxl_database['MI_STAGE1'] = (
        (
            (ptbxl_database['infarction_stadium2'] == 'Stadium I')
            | (ptbxl_database['infarction_stadium2'] == 'Stadium I-II')
        ) 
        | (
            (
                (ptbxl_database['infarction_stadium1'] == 'Stadium I')
                | (ptbxl_database['infarction_stadium1'] == 'Stadium I-II')
            )
            & (ptbxl_database['infarction_stadium2'].isna())
        )
    )
    ptbxl_database['MI_STAGE2'] = (
        (
            (ptbxl_database['infarction_stadium2'] == 'Stadium II')
            | (ptbxl_database['infarction_stadium2'] == 'Stadium II-III')
        )
        | (
            (
                (ptbxl_database['infarction_stadium1'] == 'Stadium II')
                | (ptbxl_database['infarction_stadium1'] == 'Stadium II-III')
            )
            & (ptbxl_database['infarction_stadium2'].isna())
        )
    )
    ptbxl_database['MI_STAGE3'] = (
        (
            ptbxl_database['infarction_stadium2'] == 'Stadium III'
        )
        | (
            (ptbxl_database['infarction_stadium1'] == 'Staidum III')
            & (ptbxl_database['infarction_stadium2'].isna())
        )
    )

    ptbxl_database['report'] = ptbxl_database.apply(lambda x: x['report'].replace('4.46', ''), axis=1)
    ptbxl_database['report'] = ptbxl_database.apply(lambda x: ' '.join((x['report'] + '.').split()), axis=1)

    def parse(str, pattern):
        def _parse(stmt):
            parsed = np.zeros((12,), dtype=int)
            if stmt is not None:
                for x in re.finditer(lead_patterns, stmt):
                    x = x.group()
                    parsed_ = parse_lead_positions(x, str)
                    if parsed_ is not None:
                        parsed |= parsed_
            return parsed

        statements = None
        if parsed := re.search(pattern[0], str):
            statements = parsed.group()
        elif len(pattern[1]) > 0:
            if re.search(pattern[1][0], str):
                if parsed := re.search(pattern[1][1], str):
                    statements = parsed.group()

        if statements is not None:
            return _parse(statements)
        elif len(pattern[2]) > 0:
            parsed_leads = np.zeros((12,), dtype=int)
            for alternative in pattern[2]:
                if parsed := re.search(alternative, str):
                    parsed_leads |= _parse(parsed.group())
            return parsed_leads
        else:
            return np.zeros((12,), dtype=int)

    # s = "sinus rhythm left type nonspecific abnormal t."
    # p = patterns['NDT']
    # z = parse(s, p)
    # breakpoint()

    for scp_code, pattern in patterns.items():
        print(scp_code)
        ptbxl_database[scp_code] = ptbxl_database.apply(
            lambda x: parse(x['report'], pattern) if scp_code in eval(x['scp_codes']) else None, axis=1
        )

    scp_statements = pd.read_csv(os.path.join(ptbxl_dir, 'scp_statements.csv'))
    diagnostics = scp_statements[scp_statements['diagnostic'] == 1]['Unnamed: 0'].to_list()
    diagnostics.remove('NORM')
    diagnostic_superclass = ['STTC', 'MI', 'HYP', 'CD']
    ptbxl_database[diagnostic_superclass] = 0

    def find_superclass(scp_codes):
        superclass = dict()
        for ds in diagnostic_superclass:
            superclass[ds] = False
        for scp_code in re.finditer(r'\'[a-zA-Z]*\'', scp_codes):
            scp_code = scp_code.group()
            scp_code_without_quotation = scp_code.strip("'")
            if scp_code_without_quotation in diagnostics:
                superclass[
                    scp_statements[scp_statements['Unnamed: 0'] == scp_code_without_quotation]['diagnostic_class'].iloc[0]
                ] = True
        return list(superclass.values())
    superclass = ptbxl_database.apply(lambda x: find_superclass(x['scp_codes']), axis=1)

    for i, ds in enumerate(diagnostic_superclass):
        ptbxl_database[ds] = superclass.apply(lambda x: x[i])

    subclass = scp_statements['Unnamed: 0'].to_list()
    subclass.remove('NORM')
    subclass.remove('SR')

    forms = scp_statements[scp_statements['form'] == 1]['Unnamed: 0'].to_list()
    forms_with_leads = ['NDT', 'NST_', 'DIG', 'STD_', 'VCLVH', 'QWAVE', 'LOWT', 'NT_', 'INVT', 'LVOLT', 'HVOLT', 'TAB_', 'STE_']
    rhythms = scp_statements[scp_statements['rhythm'] == 1]['Unnamed: 0'].to_list()
    # rhythms.remove("NORM")

    idx = ptbxl_database.apply(lambda _: False, axis=1)
    for s in subclass:
        idx |= ptbxl_database['scp_codes'].str.contains("'" + s + "'", regex=False)
    ptbxl_database['ANY_SCP_CODES'] = False
    ptbxl_database['ANY_SCP_CODES'][idx] = True

    idx = ptbxl_database.apply(lambda _: False, axis=1)
    for d in diagnostics:
        idx |= ptbxl_database['scp_codes'].str.contains("'" + d + "'", regex=False)
    ptbxl_database['ANY_DX_INC'] = False
    ptbxl_database['ANY_DX_INC'][idx] = True

    def _search(pattern, string, default_value=None):
        matched = re.search(pattern, string)
        if matched is not None:
            return matched.group()
        return default_value

    idx = ptbxl_database.apply(lambda _: False, axis=1)
    for d in diagnostics:
        likelihood = ptbxl_database['scp_codes'].apply(
            lambda x: _search(r'(?<=\'' + d + r'\': )\d+\.\d', x, default_value=-1)
        ).astype(float)
        idx |= ((likelihood == 0) | (likelihood == 100))
    ptbxl_database['ANY_DX_EXC'] = False
    ptbxl_database['ANY_DX_EXC'][idx] = True

    idx = ptbxl_database.apply(lambda _: False, axis=1)
    for f in forms:
        idx |= ptbxl_database['scp_codes'].str.contains("'" + f + "'", regex=False)
    ptbxl_database['ANY_FORM'] = False
    ptbxl_database['ANY_FORM'][idx] = True

    idx = ptbxl_database.apply(lambda _: False, axis=1)
    for r in rhythms:
        idx |= ptbxl_database['scp_codes'].str.contains("'" + r + "'", regex=False)
    ptbxl_database['ANY_RHYTHM'] = False
    ptbxl_database['ANY_RHYTHM'][idx] = True

    def extract_numeric_features(ecg, lead):
        clean_ecg = nk.ecg_clean(ecg[lead], sampling_rate=500)
        _, r_peaks = nk.ecg_peaks(clean_ecg, sampling_rate=500)
        if len(r_peaks['ECG_R_Peaks']) == 0:
            r_peaks = {'ECG_R_Peaks': [0], 'sampling_rate': 500}

        if len(r_peaks['ECG_R_Peaks']) < 7:
            print('Too few peaks detected in lead II: ' + str(len(r_peaks['ECG_R_Peaks'])) + ' Cannot calculate numeric features.')
            rr_interval = None
            p_duration = None
            pr_interval = None
            qrs_duration = None
            qt_interval = None
            qt_corrected = None
        else:
            _, waves_dwt = nk.ecg_delineate(clean_ecg, r_peaks, sampling_rate=500)

            p_onsets = waves_dwt['ECG_P_Onsets']
            p_offsets = waves_dwt['ECG_P_Offsets']    
            q_peaks = waves_dwt['ECG_Q_Peaks']
            r_peaks = r_peaks['ECG_R_Peaks']
            r_onsets = waves_dwt['ECG_R_Onsets']
            r_offsets = waves_dwt['ECG_R_Offsets']
            t_offsets = waves_dwt['ECG_T_Offsets']

            rr_interval = []
            for j, _ in enumerate(r_peaks[:-1]):
                rrint = (r_peaks[j+1] - r_peaks[j]) / 500
                if rrint >= 2.:
                    continue
                rr_interval.append(rrint)

            p_duration = []
            n = 0
            for p_onset, p_offset in zip(p_onsets, p_offsets):
                if any(np.isnan([p_onset, p_offset])):
                    continue
                p_duration.append((p_offset - p_onset + 1) / 500)
                n += 1
            if n == 0:
                p_duration = None

            pr_interval = []
            n = 0
            for p_onset, r_onset in zip(p_onsets, r_onsets):
                if any(np.isnan([p_onset, r_onset])):
                    continue
                pr_interval.append((r_onset - p_onset + 1) / 500)
                n += 1
            if n == 0:
                pr_interval = None

            qrs_duration = []
            n = 0
            for r_onset, r_offset in zip(r_onsets, r_offsets):
                if any(np.isnan([r_onset, r_offset])):
                    continue
                qrs_duration.append((r_offset - r_onset + 1) / 500)
                n += 1
            if n == 0:
                qrs_duration = None

            qt_interval = []
            n_qt = 0
            qt_corrected = []
            n_qtc = 0
            for j, (q_peak, t_offset) in enumerate(zip(q_peaks, t_offsets)):
                if any(np.isnan([q_peak, t_offset])):
                    continue
                qt_interval.append((t_offset - q_peak + 1) / 500)
                n_qt += 1
                if j > 0:
                    qt_corrected.append(((t_offset - q_peak + 1) / 500 ) / (math.sqrt((r_peaks[j] - r_peaks[j-1]) / 500)))
                    n_qtc += 1
            if n_qt == 0:
                qt_interval = None
            if n_qtc == 0:
                qt_corrected = None

        ecg_lead_i = ecg[0]
        clean_ecg = nk.ecg_clean(ecg_lead_i, sampling_rate=500)
        _, r_peaks = nk.ecg_peaks(clean_ecg, sampling_rate=500)

        if len(r_peaks['ECG_R_Peaks']) < 7:
            lead_i_exists = False
        else:
            _, waves_dwt = nk.ecg_delineate(clean_ecg, r_peaks, sampling_rate=500)
            r_peaks_lead_i = r_peaks['ECG_R_Peaks']
            r_onsets_lead_i = waves_dwt['ECG_R_Onsets']
            lead_i_exists = True

        ecg_lead_avf = ecg[5]
        clean_ecg = nk.ecg_clean(ecg_lead_avf, sampling_rate=500)
        _, r_peaks = nk.ecg_peaks(clean_ecg, sampling_rate=500)
        if len(r_peaks['ECG_R_Peaks']) < 7:
            lead_avf_exists = False
        else:
            _, waves_dwt = nk.ecg_delineate(clean_ecg, r_peaks, sampling_rate=500)
            r_peaks_lead_avf = r_peaks['ECG_R_Peaks']
            r_onsets_lead_avf = waves_dwt['ECG_R_Onsets']
            lead_avf_exists = True

        if not (lead_i_exists and lead_avf_exists):
            print('Too few peaks detected in lead I or aVF: ' + str(len(r_peaks['ECG_R_Peaks'])) +' Cannot calculate heart axis.')
            return {
                'rr_interval': rr_interval,
                'p_duration': p_duration,
                'pr_interval': pr_interval,
                'qrs_duration': qrs_duration,
                'qt_interval': qt_interval,
                'qt_corrected': qt_corrected,
                'qrs_lead_i': None,
                'qrs_lead_avf': None,
                'r_axis': None,
            }

        # make all peaks aligned between lead i and avf
        i_idx = []
        avf_idx = []
        for j, r_peak_lead_i in enumerate(r_peaks_lead_i):
            for k, r_peak_lead_avf in enumerate(r_peaks_lead_avf):
                if abs(r_peak_lead_i - r_peak_lead_avf) <= 50:
                    i_idx.append(j)
                    avf_idx.append(k)
                    break

        r_peaks_lead_i = [int(x) if not np.isnan(x) else x for x in np.array(r_peaks_lead_i)[i_idx].tolist()]
        r_onsets_lead_i = [int(x) if not np.isnan(x) else x for x in np.array(r_onsets_lead_i)[i_idx].tolist()]

        r_peaks_lead_avf = [int(x) if not np.isnan(x) else x for x in np.array(r_peaks_lead_avf)[avf_idx].tolist()]
        r_onsets_lead_avf = [int(x) if not np.isnan(x) else x for x in np.array(r_onsets_lead_avf)[avf_idx].tolist()]

        r_axis = []
        _qrs_lead_i = []
        _qrs_lead_avf = []
        n = 0
        for i in range(len(r_peaks_lead_i)):
            r_peak_lead_i = r_peaks_lead_i[i]
            r_onset_lead_i = r_onsets_lead_i[i]
            r_peak_lead_avf = r_peaks_lead_avf[i]
            r_onset_lead_avf = r_onsets_lead_avf[i]
            if any(
                np.isnan(
                    [
                        r_peak_lead_i,
                        r_onset_lead_i,
                        r_peak_lead_avf,
                        r_onset_lead_avf,
                    ]
                )
            ):
                continue

            srch_range = 50
            baseline_lead_amp_i = ecg_lead_i[r_onset_lead_i]
            r_peak_amp_lead_i = ecg_lead_i[max(0, r_peak_lead_i - srch_range) : min(len(ecg_lead_i), r_peak_lead_i + srch_range)]
            idx = np.argmax(list(map(lambda x: abs(baseline_lead_amp_i - x), r_peak_amp_lead_i)))
            r_peak_amp_lead_i = r_peak_amp_lead_i[idx] - baseline_lead_amp_i
            r_peak_idx_lead_i = max(0, r_peak_lead_i - srch_range) + idx

            if r_peak_amp_lead_i > 0:
                positive = r_peak_amp_lead_i
                negative = ecg_lead_i[max(0, r_peak_idx_lead_i - srch_range) : min(len(ecg_lead_i), r_peak_idx_lead_i + srch_range)]
                negative = min(list(map(lambda x: x - baseline_lead_amp_i, negative)))
            else:
                positive = ecg_lead_i[max(0, r_peak_idx_lead_i - srch_range) : min(len(ecg_lead_i), r_peak_idx_lead_i + srch_range)]
                positive = max(list(map(lambda x: x - baseline_lead_amp_i, positive)))
                negative = r_peak_amp_lead_i

            qrs_lead_i = positive + negative

            baseline_lead_amp_avf = ecg_lead_avf[r_onset_lead_avf]
            r_peak_amp_lead_avf = ecg_lead_avf[max(0, r_peak_lead_avf - srch_range): min(len(ecg_lead_avf), r_peak_lead_avf + srch_range)]
            idx = np.argmax(list(map(lambda x: abs(baseline_lead_amp_avf - x), r_peak_amp_lead_avf)))
            r_peak_amp_lead_avf = r_peak_amp_lead_avf[idx] - baseline_lead_amp_avf
            r_peak_idx_lead_avf = max(0, r_peak_lead_avf - srch_range) + idx

            if r_peak_amp_lead_avf > 0:
                positive = r_peak_amp_lead_avf
                negative = ecg_lead_avf[max(0, r_peak_idx_lead_avf - srch_range) : min(len(ecg_lead_avf), r_peak_idx_lead_avf + srch_range)]
                negative = min(list(map(lambda x: x - baseline_lead_amp_avf, negative)))
            else:
                positive = ecg_lead_avf[max(0, r_peak_idx_lead_avf - srch_range) : min(len(ecg_lead_avf), r_peak_idx_lead_avf + srch_range)]
                positive = max(list(map(lambda x: x - baseline_lead_amp_avf, positive)))
                negative = r_peak_amp_lead_avf

            qrs_lead_avf = positive + negative


            _r_axis = math.acos(np.dot(
                [qrs_lead_i, qrs_lead_avf], [1, 0]
            ) / np.linalg.norm([qrs_lead_i, qrs_lead_avf]))
            _r_axis *= 180 / np.pi
            if qrs_lead_avf < 0:
                _r_axis = -_r_axis

            _qrs_lead_i.append(qrs_lead_i)
            _qrs_lead_avf.append(qrs_lead_avf)
            r_axis.append(_r_axis)
            n += 1

        if n == 0:
            r_axis = None
            _qrs_lead_i = None
            _qrs_lead_avf = None

        return {
            'rr_interval': rr_interval,
            'p_duration': p_duration,
            'pr_interval': pr_interval,
            'qrs_duration': qrs_duration,
            'qt_interval': qt_interval,
            'qt_corrected': qt_corrected,
            'qrs_lead_i': _qrs_lead_i,
            'qrs_lead_avf': _qrs_lead_avf,
            'r_axis': r_axis,
        }

    rr_intervals = []
    p_durations = []
    pr_intervals = []
    qrs_durations = []
    qt_intervals = []
    qt_correcteds = []
    r_axes = []
    sizes = []

    lead = 1 # lead II
    for i, ecg_id in enumerate(ptbxl_database['ecg_id'].to_list()):
        print(f'{i} / {len(ptbxl_database)}')
        ecg, info = wfdb.rdsamp(
            os.path.join('/home/jwoh/ecg/data/ptb-xl/ptbxl/records500', f'{int(ecg_id / 1000) * 1000 :05d}', f'{ecg_id:05d}_hr')
        )
        ecg = ecg.T
        sizes.append(ecg.shape[-1])

        features = extract_numeric_features(ecg, lead)

        metadata = ptbxl_database[ptbxl_database['ecg_id'] == ecg_id].iloc[0]
        def is_clean_lead(_metadata, lead):
            return not (
                (isinstance(_metadata['DRIFT'], np.ndarray) and _metadata['DRIFT'][lead] == 1)
                or (isinstance(_metadata['STATIC'], np.ndarray) and _metadata['STATIC'][lead] == 1)
                or (isinstance(_metadata['BURST'], np.ndarray) and _metadata['BURST'][lead] == 1)
                or (isinstance(_metadata['ELECT'], np.ndarray) and _metadata['ELECT'][lead] == 1)
                or (isinstance(_metadata['ES'], np.ndarray) and _metadata['ES'][lead] == 1)
                or (isinstance(_metadata['VES'], np.ndarray) and _metadata['VES'][lead] == 1)
                or (isinstance(_metadata['SVES'], np.ndarray) and _metadata['SVES'][lead] == 1)
            )
        is_clean_lead_ii = is_clean_lead(metadata, lead=1)
        is_clean_lead_i = is_clean_lead(metadata, lead=0)
        is_clean_lead_avf = is_clean_lead(metadata, lead=5)

        if not is_clean_lead_ii:
            features['rr_interval'] = None
            features['p_duration'] = None
            features['pr_interval'] = None
            features['qrs_duration'] = None
            features['qt_interval'] = None
            features['qt_corrected'] = None
        
        if (not is_clean_lead_i) or (not is_clean_lead_avf):
            features['r_axis'] = None

        rr_intervals.append(features['rr_interval'])
        p_durations.append(features['p_duration'])
        pr_intervals.append(features['pr_interval'])
        qrs_durations.append(features['qrs_duration'])
        qt_intervals.append(features['qt_interval'])
        qt_correcteds.append(features['qt_corrected'])

        if features['r_axis'] is not None:
            qrs_lead_i = np.mean(features['qrs_lead_i'])
            qrs_lead_avf = np.mean(features['qrs_lead_avf'])
            inferred_r_axis = math.acos(np.dot(
                [qrs_lead_i, qrs_lead_avf], [1, 0]
            ) / np.linalg.norm([qrs_lead_i, qrs_lead_avf]))
            inferred_r_axis *= 180 / np.pi
            if qrs_lead_avf < 0:
                inferred_r_axis = -inferred_r_axis
            r_axes.append(inferred_r_axis)
        else:
            r_axes.append(None)

    ptbxl_database['size'] = sizes
    ptbxl_database['rr_interval'] = rr_intervals
    ptbxl_database['max_rr_interval'] = ptbxl_database['rr_interval'].apply(lambda x: max(x) if isinstance(x, list) else None)
    ptbxl_database['min_rr_interval'] = ptbxl_database['rr_interval'].apply(lambda x: min(x) if isinstance(x, list) else None)
    ptbxl_database['p_duration'] = p_durations
    ptbxl_database['max_p_duration'] = ptbxl_database['p_duration'].apply(lambda x: max(x) if isinstance(x, list) else None)
    ptbxl_database['min_p_duration'] = ptbxl_database['p_duration'].apply(lambda x: min(x) if isinstance(x, list) else None)
    ptbxl_database['pr_interval'] = pr_intervals
    ptbxl_database['max_pr_interval'] = ptbxl_database['pr_interval'].apply(lambda x: max(x) if isinstance(x, list) else None)
    ptbxl_database['min_pr_interval'] = ptbxl_database['pr_interval'].apply(lambda x: min(x) if isinstance(x, list) else None)
    ptbxl_database['qrs_duration'] = qrs_durations
    ptbxl_database['max_qrs_duration'] = ptbxl_database['qrs_duration'].apply(lambda x: max(x) if isinstance(x, list) else None)
    ptbxl_database['min_qrs_duration'] = ptbxl_database['qrs_duration'].apply(lambda x: min(x) if isinstance(x, list) else None)
    ptbxl_database['qt_interval'] = qt_intervals
    ptbxl_database['max_qt_interval'] = ptbxl_database['qt_interval'].apply(lambda x: max(x) if isinstance(x, list) else None)
    ptbxl_database['min_qt_interval'] = ptbxl_database['qt_interval'].apply(lambda x: min(x) if isinstance(x, list) else None)
    ptbxl_database['qt_corrected'] = qt_correcteds
    ptbxl_database['max_qt_corrected'] = ptbxl_database['qt_corrected'].apply(lambda x: max(x) if isinstance(x, list) else None)
    ptbxl_database['min_qt_corrected'] = ptbxl_database['qt_corrected'].apply(lambda x: min(x) if isinstance(x, list) else None)
    ptbxl_database['r_axis'] = r_axes

    ptbxl_database['HEART_AXIS_NORM'] = ptbxl_database['r_axis'].apply(lambda x: (-30 <= x) and (x < 90) if not pd.isna(x) else None)
    ptbxl_database['HEART_AXIS_LEFT'] = ptbxl_database['r_axis'].apply(lambda x: (-90 <= x) and (x < -30) if not pd.isna(x) else None)
    ptbxl_database['HEART_AXIS_RIGHT'] = ptbxl_database['r_axis'].apply(lambda x: (90 <= x) and (x <= 180) if not pd.isna(x) else None)
    ptbxl_database['HEART_AXIS_EXTR'] = ptbxl_database['r_axis'].apply(lambda x: (-180 <= x) and (x <= -90) if not pd.isna(x) else None)

    ptbxl_database["MAX_RR_INTERVAL_LOW"] = ptbxl_database["max_rr_interval"].apply(lambda x: x < 0.6 if x is not None else None)
    ptbxl_database["MAX_RR_INTERVAL_NORM"] = ptbxl_database["max_rr_interval"].apply(lambda x: 0.6 <= x and x <= 1.0 if x is not None else None)
    ptbxl_database["MAX_RR_INTERVAL_HIGH"] = ptbxl_database["max_rr_interval"].apply(lambda x: 1.0 < x if x is not None else None)
    ptbxl_database["MIN_RR_INTERVAL_LOW"] = ptbxl_database["min_rr_interval"].apply(lambda x: x < 0.6 if x is not None else None)
    ptbxl_database["MIN_RR_INTERVAL_NORM"] = ptbxl_database["min_rr_interval"].apply(lambda x: 0.6 <= x and x <= 1.0 if x is not None else None)
    ptbxl_database["MIN_RR_INTERVAL_HIGH"] = ptbxl_database["min_rr_interval"].apply(lambda x: 1.0 < x if x is not None else None)

    ptbxl_database["MAX_P_DURATION_LOW"] = ptbxl_database["max_p_duration"].apply(lambda x: None)
    ptbxl_database["MAX_P_DURATION_NORM"] = ptbxl_database["max_p_duration"].apply(lambda x: x <= 0.12 if x is not None else None)
    ptbxl_database["MAX_P_DURATION_HIGH"] = ptbxl_database["max_p_duration"].apply(lambda x: 0.12 < x if x is not None else None)
    ptbxl_database["MIN_P_DURATION_LOW"] = ptbxl_database["min_p_duration"].apply(lambda x: None)
    ptbxl_database["MIN_P_DURATION_NORM"] = ptbxl_database["min_p_duration"].apply(lambda x: x <= 0.12 if x is not None else None)
    ptbxl_database["MIN_P_DURATION_HIGH"] = ptbxl_database["min_p_duration"].apply(lambda x: 0.12 < x if x is not None else None)

    ptbxl_database["MAX_PR_INTERVAL_LOW"] = ptbxl_database["max_pr_interval"].apply(lambda x: x < 0.12 if x is not None else None)
    ptbxl_database["MAX_PR_INTERVAL_NORM"] = ptbxl_database["max_pr_interval"].apply(lambda x: 0.12 <= x and x <= 0.2 if x is not None else None)
    ptbxl_database["MAX_PR_INTERVAL_HIGH"] = ptbxl_database["max_pr_interval"].apply(lambda x: 0.2 < x if x is not None else None)
    ptbxl_database["MIN_PR_INTERVAL_LOW"] = ptbxl_database["min_pr_interval"].apply(lambda x: x < 0.12 if x is not None else None)
    ptbxl_database["MIN_PR_INTERVAL_NORM"] = ptbxl_database["min_pr_interval"].apply(lambda x: 0.12 <= x and x <= 0.2 if x is not None else None)
    ptbxl_database["MIN_PR_INTERVAL_HIGH"] = ptbxl_database["min_pr_interval"].apply(lambda x: 0.2 < x if x is not None else None)

    ptbxl_database["MAX_QRS_DURATION_LOW"] = ptbxl_database["max_qrs_duration"].apply(lambda x: x < 0.06 if x is not None else None)
    ptbxl_database["MAX_QRS_DURATION_NORM"] = ptbxl_database["max_qrs_duration"].apply(lambda x: 0.06 <= x and x <= 0.11 if x is not None else None)
    ptbxl_database["MAX_QRS_DURATION_HIGH"] = ptbxl_database["max_qrs_duration"].apply(lambda x: 0.11 < x if x is not None else None)
    ptbxl_database["MIN_QRS_DURATION_LOW"] = ptbxl_database["min_qrs_duration"].apply(lambda x: x < 0.06 if x is not None else None)
    ptbxl_database["MIN_QRS_DURATION_NORM"] = ptbxl_database["min_qrs_duration"].apply(lambda x: 0.06 <= x and x <= 0.11 if x is not None else None)
    ptbxl_database["MIN_QRS_DURATION_HIGH"] = ptbxl_database["min_qrs_duration"].apply(lambda x: 0.11 < x if x is not None else None)

    ptbxl_database["MAX_QT_INTERVAL_LOW"] = ptbxl_database["max_qt_interval"].apply(lambda x: x < 0.33 if x is not None else None)
    ptbxl_database["MAX_QT_INTERVAL_NORM"] = ptbxl_database["max_qt_interval"].apply(lambda x: 0.33 <= x and x <= 0.43 if x is not None else None)
    ptbxl_database["MAX_QT_INTERVAL_HIGH"] = ptbxl_database["max_qt_interval"].apply(lambda x: 0.43 < x if x is not None else None)
    ptbxl_database["MIN_QT_INTERVAL_LOW"] = ptbxl_database["min_qt_interval"].apply(lambda x: x < 0.33 if x is not None else None)
    ptbxl_database["MIN_QT_INTERVAL_NORM"] = ptbxl_database["min_qt_interval"].apply(lambda x: 0.33 <= x and x <= 0.43 if x is not None else None)
    ptbxl_database["MIN_QT_INTERVAL_HIGH"] = ptbxl_database["min_qt_interval"].apply(lambda x: 0.43 < x if x is not None else None)

    qtc_range = (0.33 / math.sqrt(1.0), 0.43 / math.sqrt(0.6))
    ptbxl_database["MAX_QT_CORRECTED_LOW"] = ptbxl_database["max_qt_corrected"].apply(lambda x: x < qtc_range[0] if x is not None else None)
    ptbxl_database["MAX_QT_CORRECTED_NORM"] = ptbxl_database["max_qt_corrected"].apply(lambda x: qtc_range[0] <= x and x <= qtc_range[1] if x is not None else None)
    ptbxl_database["MAX_QT_CORRECTED_HIGH"] = ptbxl_database["max_qt_corrected"].apply(lambda x: qtc_range[1] < x if x is not None else None)
    ptbxl_database["MIN_QT_CORRECTED_LOW"] = ptbxl_database["min_qt_corrected"].apply(lambda x: x < qtc_range[0] if x is not None else None)
    ptbxl_database["MIN_QT_CORRECTED_NORM"] = ptbxl_database["min_qt_corrected"].apply(lambda x: qtc_range[0] <= x and x <= qtc_range[1] if x is not None else None)
    ptbxl_database["MIN_QT_CORRECTED_HIGH"] = ptbxl_database["min_qt_corrected"].apply(lambda x: qtc_range[1] < x if x is not None else None)

    if not os.path.exists('results'):
        os.mkdir('results')
    ptbxl_database.to_pickle(os.path.join('results', 'encoded_ptbxl.pkl'))

    return ptbxl_database
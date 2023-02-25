import argparse
import os

import pickle
import pandas as pd

# from ptbxl_qa_pipelines.map_to_mimiciii import map_to_mimiciii
from ptbxl_qa_pipelines.encode_ptbxl import encode_ptbxl
from ptbxl_qa_pipelines.instantiate_template import instantiate_template
from ptbxl_qa_pipelines.manifest import manifest

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", type=str, default=None,
        help="specific task from the whole pipeline. "
            "if not set, run the whole steps."
    )

    parser.add_argument(
        '--ptbxl-dir', type=str,
        help='directory containing ptbxl_database.csv and scp_statements.csv.'
        ' We recommend you to first translate `report` in ptbxl_database.csv'
        ' (or you can download ptbxl_database_translated.csv from ...'
        ' and rename to "ptbxl_database.csv") '
        ' and then run this pipeline.'
    )
    parser.add_argument(
        '--ptbxl-data-dir', type=str, default=None,
        help='directory containing ptbxl data (records500/**/*.mat). '
        'if not given, try to infer from --ptbxl-dir'
    )
    parser.add_argument(
        '--mimic-dir', metavar='DIR',
        help='directory containing MIMIC-III .csv files'
    )
    parser.add_argument(
        '--template-dir', metavar='DIR',
        help='directory containing type1QA_template.csv file'
    )
    parser.add_argument(
        '--tokenize', action="store_true",
        help='whether to tokenize questions and answers'
    )
    parser.add_argument(
        '--answer_encode', type=str, default='multi-label',
        help='answer type. should be one of ["multi-label", "text"]'
    )

    parser.add_argument(
        '--ptbxl-db', type=str, default='results/mapped_ptbxl.csv',
        help='path to ptbxl_database.csv'
    )
    parser.add_argument(
        '--encoded-ptbxl', type=str, default='results/encoded_ptbxl.pkl',
        help='path to encoded_ptbxl.pkl'
    )
    parser.add_argument(
        '--sampled-data', type=str, default='results/sampled_data.pkl',
        help='path to sampled_data.pkl'
    )
    parser.add_argument(
        '--derived-grounding-data', type=str, default='results/derived_grounding_data.pkl',
        help='path to derived_grounding_data.pkl'
    )
    parser.add_argument(
        '--independent-grounding-data', type=str, default='results/independent_grounding_data.pkl',
        help='path to independent_grounding_data.pkl'
    )

    parser.add_argument(
        "--valid-percent",
        default=0.1,
        type=float,
        metavar="D",
        help="percentage of ecg to use as validation",
    )
    parser.add_argument(
        "--test-percent",
        default=0.1,
        type=float,
        metavar="D",
        help="percentage of ecg to use as test",
    )
    parser.add_argument(
        "--dest", default=".", type=str, metavar="DIR", help="output directory"
    )
    parser.add_argument("--seed", default=42, type=int, metavar="N", help="random seed")

    return parser

def main(args):
    task = args.task
    
    if args.ptbxl_data_dir is None:
        args.ptbxl_data_dir = args.ptbxl_dir

    assert task in [
        None,
        # 'map_to_mimiciii',
        'encode_ptbxl',
        'instantiate_template',
        'manifest'
    ]
    if task is None:
        # print('[1] Map PTB-XL to MIMIC-III')
        # mapped_ptbxl = map_to_mimiciii(args.ptbxl_dir, args.mimic_dir)
        ptbxl_database = pd.read_csv(os.path.join(args.ptbxl_dir, "ptbxl_database.csv"))
        ptbxl_database = ptbxl_database[ptbxl_database["validated_by_human"]]
        ptbxl_database["report"] = (
            ptbxl_database["report"].map(lambda x: x.replace("ekg", "ecg").replace(".", ""))
        )
        print('[2] Encode PTB-XL database')
        encoded_ptbxl = encode_ptbxl(args.ptbxl_dir, ptbxl_database=ptbxl_database)
        print('[3] Instantiate templates based on the encoded PTB-XL')

        sampled_data, derived_grounding_data, independent_grounding_data = instantiate_template(
            ptbxl_dir=args.ptbxl_dir,
            ptbxl_data_dir=args.ptbxl_data_dir,
            template_dir=args.template_dir,
            encoded_ptbxl=encoded_ptbxl,
            valid_percent=args.valid_percent,
            test_percent=args.test_percent,
            seed=args.seed,
            tokenize=args.tokenize,
            answer_encode=args.answer_encode
        )
        print('[4] Prepare manifest for the sampled data')
        manifest(sampled_data, derived_grounding_data, independent_grounding_data, args.dest)
    # elif task == 'map_to_mimiciii':
    #     map_to_mimiciii(args.ptbxl_dir, args.mimic_dir)
    elif task == 'encode_ptbxl':
        ptbxl_database = pd.read_csv(os.path.join(args.ptbxl_dir, "ptbxl_database.csv"))
        ptbxl_database = ptbxl_database[ptbxl_database["validated_by_human"]]
        ptbxl_database["report"] = (
            ptbxl_database["report"].map(lambda x: x.replace("ekg", "ecg").replace(".", ""))
        )

        encode_ptbxl(args.ptbxl_dir, ptbxl_database)
    elif task == 'instantiate_template':
        encoded_ptbxl = pd.read_pickle(args.encoded_ptbxl)

        instantiate_template(
            ptbxl_dir=args.ptbxl_dir,
            ptbxl_data_dir=args.ptbxl_data_dir,
            template_dir=args.template_dir,
            encoded_ptbxl=encoded_ptbxl,
            valid_percent=args.valid_percent,
            test_percent=args.test_percent,
            seed=args.seed,
            tokenize=args.tokenize,
            answer_encode=args.answer_encode
        )
    elif task == 'manifest':
        with open(args.sampled_data, 'rb') as f:
            sampled_data = pickle.load(f)
        with open(args.derived_grounding_data, "rb") as f:
            derived_grounding_data = pickle.load(f)
        with open(args.independent_grounding_data, "rb") as f:
            independent_grounding_data = pickle.load(f)
        manifest(sampled_data, derived_grounding_data, independent_grounding_data, args.dest)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
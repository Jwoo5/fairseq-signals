import argparse
import os
import glob
import wfdb
import numpy as np
import scipy.io
from scipy.interpolate import CubicSpline

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", metavar="DIR",
                       help="root directory containing mat files to pre-process")
    parser.add_argument(
        "--sample-rate",
        default=500,
        type=int,
        help="if set, data must be sampled by this sampling rate to be processed"
    )
    parser.add_argument(
        "--resample",
        default=False,
        action='store_true',
        help='if set, resample data to have a sample rate of --sample-rate'
    )
    parser.add_argument(
        "--separate-leads",
        default=False,
        action='store_true',
        help='if set, save each lead individually with a suffix indicating the lead'
    )
    parser.add_argument("--dest", type=str, metavar="DIR",
                       help="output directory")

    return parser

def align_segments(segments_list, threshold=50):
    # segments_list: list of 3-tuples where each 3-tuple is composed of (onset, peak, offset)
    if not segments_list:
        return []
    segments_list = sorted(segments_list, key=lambda x: x[1])

    grouped = []
    current_group = [segments_list[0]]
    base_value = segments_list[0][1]

    for i in range(1, len(segments_list)):
        if abs(segments_list[i][1] - base_value) < threshold:
            current_group.append(segments_list[i])
        else:
            grouped.append(current_group)
            current_group = [segments_list[i]]
            base_value = segments_list[i][1]
    
    grouped.append(current_group)
    # aligned = [(min(x, key=lambda x: x[0])[0], max(x, key=lambda x: x[2])[2]) for x in grouped]
    aligned = [
        (int(np.mean([y[0] for y in x])), int(np.mean([y[2] for y in x]))) for x in grouped
    ]

    return aligned

def main(args):
    label_dict = {
        "Aberrant conduction.": 0,
        "Atrial extrasystole, type: allorhythmic pattern.": 1,
        "Atrial extrasystole, type: bigemini.": 2,
        "Atrial extrasystole, type: quadrigemini.": 3,
        "Atrial extrasystole, type: single PAC.": 4,
        "Atrial extrasystole: SA-nodal extrasystole.": 5,
        "Atrial extrasystole: left atrial.": 6,
        "Atrial extrasystole: low atrial.": 7,
        "Atrial extrasystole: undefined.": 8,
        "BIpolar ventricular pacing.": 9,
        "Biventricular pacing.": 10,
        "Complete left bundle branch block.": 11,
        "Complete right bundle branch block.": 12,
        "Early repolarization syndrome.": 13,
        "Electric axis of the heart: horizontal.": 14,
        "Electric axis of the heart: left axis deviation.": 15,
        "Electric axis of the heart: normal.": 16,
        "Electric axis of the heart: right axis deviation.": 17,
        "Electric axis of the heart: vertical.": 18,
        "I degree AV block.": 19,
        "III degree AV-block.": 20,
        "Incomplete left bundle branch block.": 21,
        "Incomplete right bundle branch block.": 22,
        "Ischemia: anterior wall.": 23,
        "Ischemia: apical.": 24,
        "Ischemia: inferior wall.": 25,
        "Ischemia: lateral wall.": 26,
        "Ischemia: posterior wall.": 27,
        "Ischemia: septal.": 28,
        "Left anterior hemiblock.": 29,
        "Left atrial hypertrophy.": 30,
        "Left atrial overload.": 31,
        "Left ventricular hypertrophy.": 32,
        "Left ventricular overload.": 33,
        "Non-specific intravintricular conduction delay.": 34,
        "Non-specific repolarization abnormalities: anterior wall.": 35,
        "Non-specific repolarization abnormalities: apical.": 36,
        "Non-specific repolarization abnormalities: inferior wall.": 37,
        "Non-specific repolarization abnormalities: lateral wall.": 38,
        "Non-specific repolarization abnormalities: posterior wall.": 39,
        "Non-specific repolarization abnormalities: septal.": 40,
        "P-synchrony.": 41,
        "Rhythm: Atrial fibrillation.": 42,
        "Rhythm: Atrial flutter, typical.": 43,
        "Rhythm: Irregular sinus rhythm.": 44,
        "Rhythm: Sinus arrhythmia.": 45,
        "Rhythm: Sinus bradycardia.": 46,
        "Rhythm: Sinus rhythm.": 47,
        "Rhythm: Sinus tachycardia.": 48,
        "Right atrial hypertrophy.": 49,
        "Right atrial overload.": 50,
        "Right ventricular hypertrophy.": 51,
        "STEMI: anterior wall.": 52,
        "STEMI: apical.": 53,
        "STEMI: inferior wall.": 54,
        "STEMI: lateral wall.": 55,
        "STEMI: septal.": 56,
        "Scar formation: anterior wall.": 57,
        "Scar formation: apical.": 58,
        "Scar formation: inferior wall.": 59,
        "Scar formation: lateral wall.": 60,
        "Scar formation: posterior wall.": 61,
        "Scar formation: septal.": 62,
        "Sinoatrial blockade, undefined.": 63,
        "UNIpolar atrial pacing.": 64,
        "UNIpolar ventricular pacing.": 65,
        "Undefined ischemia/scar/supp.NSTEMI: anterior wall.": 66,
        "Undefined ischemia/scar/supp.NSTEMI: apical.": 67,
        "Undefined ischemia/scar/supp.NSTEMI: inferior wall.": 68,
        "Undefined ischemia/scar/supp.NSTEMI: lateral wall.": 69,
        "Undefined ischemia/scar/supp.NSTEMI: posterior wall.": 70,
        "Undefined ischemia/scar/supp.NSTEMI: septal.": 71,
        "Ventricular extrasystole, localisation: IVS, middle part.": 72,
        "Ventricular extrasystole, localisation: LV, undefined.": 73,
        "Ventricular extrasystole, localisation: LVOT, LVS.": 74,
        "Ventricular extrasystole, localisation: RVOT, anterior wall.": 75,
        "Ventricular extrasystole, localisation: RVOT, antero-septal part.": 76,
        "Ventricular extrasystole, morphology: polymorphic.": 77,
        "Ventricular extrasystole, type: couplet.": 78,
        "Ventricular extrasystole, type: intercalary PVC.": 79,
        "Ventricular extrasystole, type: single PVC.": 80,
        "Wandering atrial pacemaker.": 81,
    }

    dir_path = os.path.realpath(args.root)
    dest_path = os.path.realpath(args.dest)
    
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    
    for fname in glob.iglob(os.path.join(dir_path, "*.dat")):
        fname = os.path.splitext(fname)[0]

        basename = os.path.basename(fname)
        ecg, header = wfdb.rdsamp(fname)
        ecg = ecg.T
        data = {}

        sample_rate = header["fs"]
        resampled = False
        if (
            args.sample_rate is not None
            and sample_rate != args.sample_rate
        ):
            if args.resample:
                resampled = True
                length = ecg.shape[-1]
                sample_size = int(length * (args.sample_rate / sample_rate))
                xs = np.linspace(0, length, sample_size)
                f = CubicSpline(np.arange(length), ecg)
                ecg = f(xs)
                sample_rate = args.sample_rate
                length = sample_size
            else:
                print(f"Skipped {fname} due to the following reason: Not a desired sampling rate")
                continue
        
        label = np.zeros(len(label_dict)).astype(int)
        label_idx = [label_dict[x] for x in header["comments"][3:]]
        label[label_idx] = 1
        data["label"] = label
        data["curr_sample_rate"] = sample_rate

        p_waves = []
        qrs_waves = []
        t_waves = []
        for i, lead_name in enumerate(header["sig_name"]):
            ann = wfdb.rdann(fname, extension=lead_name)
            if resampled:
                resample_rate = sample_rate / header["fs"]
                ann.sample = [round(x * resample_rate) for x in ann.sample]

            waves = {
                "p": [],
                "qrs": [],
                "t": []
            }
            on = None
            for t, symbol in zip(ann.sample, ann.symbol):
                if symbol == "(":
                    on = t
                elif symbol == ")":
                    off = t
                    if on is None:
                        on = key_t
                    waves[key].append((on, key_t, off))
                    on = None
                else:
                    key_t = t
                    if symbol in ["p", "t"]:
                        key = symbol
                    else:
                        key = "qrs"
            p_waves.append(waves["p"])
            qrs_waves.append(waves["qrs"])
            t_waves.append(waves["t"])
        
        aligned_p_waves = align_segments(sum(p_waves, []), threshold=int(0.2 * sample_rate))
        aligned_qrs_waves = align_segments(sum(qrs_waves, []), threshold=int(0.2 * sample_rate))
        aligned_t_waves = align_segments(sum(t_waves, []), threshold=int(0.2 * sample_rate))
        if not args.separate_leads:
            data["feats"] = ecg
            data["p"] = np.zeros(len(data["feats"][0])).astype(int)
            data["qrs"] = np.zeros(len(data["feats"][0])).astype(int)
            data["t"] = np.zeros(len(data["feats"][0])).astype(int)

            for onset, offset in aligned_p_waves:
                assert (offset - onset) <= int(0.3 * sample_rate)
                data["p"][onset:offset+1] = 1
            for onset, offset in aligned_qrs_waves:
                assert (offset - onset) <= int(0.3 * sample_rate)
                data["qrs"][onset:offset+1] = 1
            for onset, offset in aligned_t_waves:
                assert (offset - onset) <= int(0.4 * sample_rate)
                data["t"][onset:offset+1] = 1

            data["none"] = np.ones(len(data["feats"][0])).astype(int) - data["p"] - data["qrs"] - data["t"]
            segments = np.stack((data["p"], data["qrs"], data["t"], data["none"]), axis=1)
            assert segments.sum() == len(data["feats"][0])
            data["segment_label"] = np.argmax(segments, axis=1)

            any_type = data["p"] + data["qrs"] + data["t"]
            min_idx = np.where(any_type)[0].min()
            max_idx = np.where(any_type)[0].max()
            mask = np.ones(len(data["feats"][0])).astype(bool)
            mask[min_idx:max_idx+1] = False
            data["segment_mask"] = mask

            scipy.io.savemat(os.path.join(dest_path, basename + ".mat"), data)
        else:
            for i, lead_name in enumerate(header["sig_name"]):
                data["feats"] = ecg[i]
                data["p"] = np.zeros(len(data["feats"])).astype(int)
                data["qrs"] = np.zeros(len(data["feats"])).astype(int)
                data["t"] = np.zeros(len(data["feats"])).astype(int)

                for onset, peak, offset in p_waves[i]:
                    data["p"][onset:offset+1] = 1
                for onset, peak, offset in qrs_waves[i]:
                    assert (offset - onset) <= int(0.3 * sample_rate)
                    data["qrs"][onset:offset+1] = 1
                for onset, peak, offset in t_waves[i]:
                    assert (offset - onset) <= int(0.4 * sample_rate)
                    data["t"][onset:offset+1] = 1

                data["none"] = np.ones(len(data["feats"])).astype(int) - data["p"] - data["qrs"] - data["t"]
                segments = np.stack((data["p"], data["qrs"], data["t"], data["none"]), axis=1)
                assert segments.sum() == len(data["feats"])
                data["segment_label"] = np.argmax(segments, axis=1)

                any_type = data["p"] + data["qrs"] + data["t"]
                min_idx = np.where(any_type)[0].min()
                max_idx = np.where(any_type)[0].max()
                mask = np.ones(len(data["feats"])).astype(bool)
                mask[min_idx:max_idx+1] = False
                data["segment_mask"] = mask

                scipy.io.savemat(os.path.join(dest_path, basename + f"_{lead_name}.mat"), data)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
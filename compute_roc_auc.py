"""Recompute the paper's empirical OvR ROC/AUC from frozen probability CSVs."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path, PureWindowsPath
from hashlib import sha256
import numpy as np
from reference.roc_math import roc_binary, macro_curve

CLASSES=('defect','normal','nut')


def calculate(prediction_file):
    with prediction_file.open(encoding='utf-8-sig',newline='') as handle:
        rows=list(csv.DictReader(handle))
    if len(rows)!=629:raise ValueError('Expected 629 frozen test predictions.')
    truth=np.array([int(r['true_id']) for r in rows])
    scores=np.array([[float(r['prob_'+name]) for name in CLASSES] for r in rows])
    if [int((truth==i).sum()) for i in range(3)]!=[129,306,194]:
        raise ValueError('Test class counts differ from the paper.')
    if not np.isfinite(scores).all() or (scores<0).any() or (scores>1).any():
        raise ValueError('Invalid probability values.')
    if np.max(np.abs(scores.sum(1)-1))>0.0015:
        raise ValueError('Probability sums exceed the original float16 rounding tolerance.')
    keys=[(PureWindowsPath(r['image']).parent.name,PureWindowsPath(r['image']).name,r['true_class']) for r in rows]
    if len(set(keys))!=629:raise ValueError('Repeated test samples.')
    if any(CLASSES[y]!=r['true_class'] for y,r in zip(truth,rows,strict=True)):
        raise ValueError('Class label and true_id disagree.')
    curves=[roc_binary(truth==i,scores[:,i]) for i in range(3)]
    x,y,mean=macro_curve(curves)
    return curves,(x,y,mean),sorted(keys)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,required=True,help='CSV with classifier,input,run_dir columns.')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('Use a new/empty output directory.')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with args.runs.open(encoding='utf-8-sig',newline='') as handle:runs=list(csv.DictReader(handle))
    summary=[];reference=None;sources=[]
    for r in runs:
        run=Path(r['run_dir'])
        file=run/'test_predictions.csv'
        before=sha256(file.read_bytes()).hexdigest()
        curves,macro,keys=calculate(file)
        if reference is None:reference=keys
        if keys!=reference:raise ValueError('Test identities or labels differ across runs.')
        slug=''.join(c if c.isalnum() else '_' for c in r['classifier']+'_'+r['input'])
        for name,curve in zip((*CLASSES,'macro'),(*curves,macro),strict=True):
            with (args.output_dir/f'{slug}_{name}.csv').open('w',newline='',encoding='utf-8') as h:
                w=csv.writer(h);w.writerow(['FPR','TPR']);w.writerows(zip(curve[0],curve[1],strict=True))
        summary.append({'Classifier':r['classifier'],'Input':r['input'],'Test N':629,
                        **{name+' AUC':curves[i][-1] for i,name in enumerate(CLASSES)},
                        'Macro OvR AUC':macro[-1]})
        assert before==sha256(file.read_bytes()).hexdigest()
        sources.append({'classifier':r['classifier'],'input':r['input'],'predictions_sha256':before})
    with (args.output_dir/'auc_summary.csv').open('w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    (args.output_dir/'provenance.json').write_text(json.dumps({'sources':sources,'probabilities_renormalized':False,
        'definition':'Mean of three class-specific OvR AUCs; no smoothing or threshold tuning.'},indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()

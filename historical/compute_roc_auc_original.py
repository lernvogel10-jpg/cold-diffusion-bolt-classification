"""Read-only ROC/AUC analysis of frozen test predictions, separate from EditaPlot rendering."""
from pathlib import Path, PureWindowsPath
from zipfile import ZipFile
from hashlib import sha256
from collections import Counter
import csv
import json
import sys
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
SNAPSHOT = OUT.parent / 'source_audit.json'
SOURCE_MD = ROOT / 'COLD_DIFFUSION_RESNET18_EXPERIMENT_SUMMARY.md'
snapshot = json.loads(SNAPSHOT.read_text(encoding='utf-8'))
prior = json.loads((ROOT / 'tmp/section43_comparison_20261017/edit_report.json').read_text(encoding='utf-8'))
CLASSES = ['defect', 'normal', 'nut']
COUNTS = [129, 306, 194]
N = 629
models = snapshot['models']
digest = lambda p: sha256(Path(p).read_bytes()).hexdigest()
source_word = Path(snapshot['source'])
frozen_hashes = {str(source_word): digest(source_word), str(SOURCE_MD): digest(SOURCE_MD)}

def area(x, y):
    return float(np.sum(np.diff(x) * (y[:-1] + y[1:]) / 2.0))

def roc_binary(positive, scores):
    """All unique score thresholds, with ties grouped and an initial (0,0) vertex."""
    positive = np.asarray(positive, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind='stable')
    sorted_scores, truth = scores[order], positive[order]
    last_at_threshold = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores)-1]
    tp = np.cumsum(truth, dtype=np.int64)[last_at_threshold]
    fp = last_at_threshold + 1 - tp
    fpr = np.r_[0.0, fp / np.count_nonzero(~positive)]
    tpr = np.r_[0.0, tp / np.count_nonzero(positive)]
    threshold = np.r_[np.inf, sorted_scores[last_at_threshold]]
    auc_value = area(fpr, tpr)
    # Independent Mann-Whitney concordance check, including half-credit for ties.
    pos = scores[positive, None]
    neg = scores[~positive][None, :]
    concordance = float((np.count_nonzero(pos > neg) + .5*np.count_nonzero(pos == neg)) / (pos.size*neg.size))
    assert abs(auc_value-concordance) < 1e-12, (auc_value, concordance)
    assert np.all(np.diff(fpr) >= 0) and np.all(np.diff(tpr) >= 0)
    assert np.array_equal([fpr[0], tpr[0], fpr[-1], tpr[-1]], [0, 0, 1, 1])
    return fpr, tpr, threshold, auc_value

def limits_on_grid(x, y, grid):
    """Exact left/right values of an empirical piecewise-linear ROC, preserving vertical jumps."""
    unique_x, first = np.unique(x, return_index=True)
    last = np.r_[first[1:]-1, len(x)-1]
    left, right = y[first], y[last]
    result_left, result_right = [], []
    for gx in grid:
        index = np.searchsorted(unique_x, gx)
        if index < len(unique_x) and unique_x[index] == gx:
            result_left.append(left[index])
            result_right.append(right[index])
        else:
            assert 0 < index < len(unique_x)
            fraction = (gx-unique_x[index-1]) / (unique_x[index]-unique_x[index-1])
            value = right[index-1] + fraction*(left[index]-right[index-1])
            result_left.append(value)
            result_right.append(value)
    return np.column_stack([result_left, result_right]).ravel()

def macro_curve(class_curves):
    grid = np.unique(np.concatenate([curve[0] for curve in class_curves]))
    x = np.repeat(grid, 2)
    y = np.mean([limits_on_grid(curve[0], curve[1], grid) for curve in class_curves], axis=0)
    expected_auc = float(np.mean([curve[3] for curve in class_curves]))
    assert abs(area(x, y)-expected_auc) < 1e-12
    return x, y, expected_auc

def write_csv(path, headers, data):
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(data)

for positive, scores, expected in [([1,0],[.9,.1],1),([1,0],[.1,.9],0),([1,0],[.5,.5],.5),([0,0,1,1],[.1,.4,.35,.8],.75)]:
    assert abs(roc_binary(positive, scores)[3]-expected) < 1e-12

# Confirm that these frozen runs still correspond to current Word Table 3, not an old table.
ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
with ZipFile(source_word) as archive:
    body = ET.fromstring(archive.read('word/document.xml')).find('w:body', ns)
visible = lambda e: ''.join(t.text or '' for t in e.findall('.//w:t', ns))
elements = list(body)
caption_index = next(i for i,e in enumerate(elements) if visible(e).startswith('Table 3.'))
table_rows = elements[caption_index+1].findall('w:tr', ns)[1:]
assert len(table_rows) == 12
for row, old in zip(table_rows, prior['rows'], strict=True):
    assert [visible(c) for c in row.findall('w:tc', ns)[2:]] == old['values_percent']

sources = []
records = []
curves = {}
raw_curve_rows = []
sample_reference = None
classification_keys = ['accuracy', 'weighted_precision', 'weighted_recall', 'weighted_f1']
for old in prior['rows']:
    model, variant = old['classifier'], old['input']
    metrics_path = Path(old['source_metrics'])
    prediction_path = metrics_path.with_name('test_predictions.csv')
    frozen_hashes[str(metrics_path)] = digest(metrics_path)
    frozen_hashes[str(prediction_path)] = digest(prediction_path)
    assert frozen_hashes[str(metrics_path)] == old['source_sha256']
    with prediction_path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        headers = list(reader.fieldnames)
        prediction_rows = list(reader)
    assert len(prediction_rows) == N
    assert headers == ['image','true_id','true_class','predicted_id','predicted_class','confidence','prob_defect','prob_normal','prob_nut']
    sample_keys = [(PureWindowsPath(r['image']).parent.name, PureWindowsPath(r['image']).name) for r in prediction_rows]
    assert len(set(sample_keys)) == N
    assert all(key[0] == r['true_class'] for key,r in zip(sample_keys,prediction_rows,strict=True))
    lookup = {key: r for key,r in zip(sample_keys,prediction_rows,strict=True)}
    if sample_reference is None:
        sample_reference = sorted(sample_keys)
        reference_labels = {key: lookup[key]['true_class'] for key in sample_reference}
    assert set(sample_keys) == set(sample_reference)
    prediction_rows = [lookup[key] for key in sample_reference]
    assert all(r['true_class'] == reference_labels[key] for key,r in zip(sample_reference,prediction_rows,strict=True))
    truth = np.array([int(r['true_id']) for r in prediction_rows])
    predicted = np.array([int(r['predicted_id']) for r in prediction_rows])
    probabilities = np.array([[float(r['prob_'+c]) for c in CLASSES] for r in prediction_rows])
    assert set(truth) == {0,1,2} and set(predicted).issubset({0,1,2})
    assert all(CLASSES[y] == r['true_class'] and CLASSES[p] == r['predicted_class'] for y,p,r in zip(truth,predicted,prediction_rows,strict=True))
    assert [int(np.count_nonzero(truth == i)) for i in range(3)] == COUNTS
    assert np.all(np.isfinite(probabilities)) and np.all((probabilities >= 0) & (probabilities <= 1))
    sum_deviation = float(np.max(np.abs(probabilities.sum(axis=1)-1)))
    # Saved inference probabilities are float16-rounded; preserve them, do not renormalize.
    assert sum_deviation <= 0.0015, sum_deviation
    chosen = probabilities[np.arange(N), predicted]
    assert np.all(chosen == probabilities.max(axis=1))
    assert np.max(np.abs(chosen-np.array([float(r['confidence']) for r in prediction_rows]))) < 1e-8
    confusion = np.zeros((3,3), dtype=int)
    np.add.at(confusion, (truth, predicted), 1)
    raw_metrics = json.loads(metrics_path.read_text(encoding='utf-8-sig'))
    assert np.array_equal(confusion, np.array(raw_metrics['confusion_matrix']))
    support, predicted_count = confusion.sum(axis=1), confusion.sum(axis=0)
    precision = np.divide(np.diag(confusion), predicted_count, out=np.zeros(3), where=predicted_count != 0)
    recall = np.diag(confusion)/support
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(3), where=precision+recall != 0)
    classification = [float(np.trace(confusion)/N), float(precision@support/N), float(recall@support/N), float(f1@support/N)]
    assert [f'{100*v:.2f}' for v in classification] == old['values_percent']
    assert all(abs(computed-raw_metrics[k]) < 1e-12 for k,computed in zip(classification_keys,classification,strict=True))
    class_curves = []
    for index, class_name in enumerate(CLASSES):
        curve = roc_binary(truth == index, probabilities[:,index])
        class_curves.append(curve)
        curves[(model,variant,class_name)] = curve
        raw_curve_rows.extend([model,variant,class_name,float(x),float(y),'inf' if np.isinf(t) else float(t)] for x,y,t in zip(*curve[:3],strict=True))
    mx,my,macro_auc = macro_curve(class_curves)
    curves[(model,variant,'macro')] = (mx,my,macro_auc)
    record = {'Classifier':model, 'Input':variant, 'Test N':N, **{c+' AUC':class_curves[i][3] for i,c in enumerate(CLASSES)}, 'Macro OvR AUC':macro_auc}
    records.append(record)
    sources.append({
        'classifier':model,'input':variant,'predictions':str(prediction_path),
        'predictions_sha256':frozen_hashes[str(prediction_path)],
        'metrics':str(metrics_path),'metrics_sha256':frozen_hashes[str(metrics_path)],
        'samples':N, 'class_counts':dict(zip(CLASSES,COUNTS,strict=True)),
        'max_probability_sum_abs_deviation':sum_deviation,
        'probabilities_renormalized':False,
        'prediction_argmax_verified':True,
        'confusion_matrix_and_all_table3_metrics_verified':True,
        'class_auc_pairwise_checks_passed':True,
        'macro_curve_area_matches_mean_class_auc':True,
        'sample_keys_and_true_labels_identical_across_all_runs':True,
    })

summary_headers = ['Classifier','Input','Test N',*[c+' AUC' for c in CLASSES],'Macro OvR AUC']
write_csv(OUT/'auc_summary.csv',summary_headers,[[r[k] for k in summary_headers] for r in records])
write_csv(OUT/'classwise_empirical_roc_long.csv',['Classifier','Input','Class','FPR','TPR','Threshold'],raw_curve_rows)
write_csv(OUT/'auc_comparison_macro.csv',['Classifier','Original Macro OvR AUC','Remapped Macro OvR AUC'],[[model,*[next(r for r in records if r['Classifier']==model and r['Input']==variant)['Macro OvR AUC'] for variant in ['Original','Remapped']]] for model in models])
curve_exports = []
preview_data = {}
slugs = {model: ''.join(c.lower() if c.isalnum() else '_' for c in model).strip('_') for model in models}
for model in models:
    preview_data[model] = {}
    for name in ['macro',*CLASSES]:
        pair = [curves[(model,variant,name)] for variant in ['Original','Remapped']]
        # Exact common-knot display; preserve both limits of every vertical jump.
        grid = np.unique(np.concatenate([curve[0] for curve in pair]))
        fpr = np.repeat(grid,2)
        y_original = limits_on_grid(pair[0][0],pair[0][1],grid)
        y_remapped = limits_on_grid(pair[1][0],pair[1][1],grid)
        auc_values = [curve[-1] for curve in pair]
        assert abs(area(fpr,y_original)-auc_values[0]) < 1e-12
        assert abs(area(fpr,y_remapped)-auc_values[1]) < 1e-12
        file = OUT / f'roc_{slugs[model]}_{name}.csv'
        write_csv(file,['FPR','Original TPR','Remapped TPR'],zip(fpr,y_original,y_remapped,strict=True))
        curve_exports.append({'classifier':model,'curve':name,'file':str(file),'sha256':digest(file),'rows':len(fpr)})
        preview_data[model][name] = {'fpr':fpr.tolist(),'Original':y_original.tolist(),'Remapped':y_remapped.tolist(),'Original AUC':auc_values[0],'Remapped AUC':auc_values[1]}

for path,expected in frozen_hashes.items():
    assert digest(path) == expected, f'Input changed: {path}'
manifest = {
    'purpose':'Supplement Table 3 with test-set empirical classwise OvR ROC and macro OvR AUC, without retraining or retesting.',
    'source_word':str(source_word),'source_word_sha256':frozen_hashes[str(source_word)],
    'source_md':str(SOURCE_MD),'source_md_sha256':frozen_hashes[str(SOURCE_MD)],
    'sample_unit':'same 629 held-out classifier-test samples in all 12 prediction files',
    'class_order':CLASSES,'class_counts':dict(zip(CLASSES,COUNTS,strict=True)),
    'classification_probabilities_used':'saved per-class probability, never hard predicted label or maximum confidence alone',
    'positive_class_definition':'one named class versus the other two classes combined',
    'roc_definition':'all distinct saved scores as thresholds; ties grouped; initial (0,0); straight empirical segments; no smoothing or dropped points',
    'auc_definition':'trapezoidal area of empirical ROC, independently checked by positive-negative concordance with 0.5 credit for ties',
    'macro_auc_definition':'unweighted arithmetic mean of defect/normal/nut OvR AUCs',
    'macro_curve_definition':'pointwise arithmetic mean of the three empirical piecewise-linear ROC functions on their union of FPR knots; both left/right limits retained at vertical jumps; its area equals the macro AUC exactly within 1e-12',
    'probability_sum_policy':'allow float16 rounding deviations <=0.0015; preserve saved class scores without renormalization',
    'input_column_dispositions':{
        'image':'support_only: sample key alignment', 'true_id':'support_only: OvR label calculation',
        'true_class':'support_only: class-order check', 'predicted_id':'support_only: recheck Table 3/confusion matrix',
        'predicted_class':'support_only: consistency check', 'confidence':'support_only: maximum probability check; never used as OvR score',
        'prob_defect':'primary upstream score for defect ROC', 'prob_normal':'primary upstream score for normal ROC', 'prob_nut':'primary upstream score for nut ROC'},
    'sources':sources,'curve_exports':curve_exports,'auc_results':records,
    'numpy_version':np.__version__,'python_version':sys.version,
    'independent_auc_checks':36,'macro_curve_area_checks':12,
    'source_files_unchanged':True,'models_retrained':False,'new_inference_performed':False,
    'not_computed':['confidence intervals','DeLong tests','p-values','optimized threshold','PR/AUPRC','bootstrap'],
    'interpretation_limits':'single seed 43 and fixed inference noise; Stage 1 used all 6199 class-encoded watermark targets including the classifier validation/test targets; these are not full-pipeline strictly independent generalization estimates',
    'preview_only':True,'origin_render_performed':False,
}
(OUT/'roc_auc_report.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'preview_data.json').write_text(json.dumps(preview_data,ensure_ascii=False),encoding='utf-8')
lines = ['# 表3补充 ROC与AUC分析','','本报告从已保存的12组测试集预测计算，没有重新训练、生成图像或执行模型推理。所有组采用相同629个样本，defect/normal/nut分别为129/306/194张。','','## 计算口径','','分别以defect、normal、nut为正类，其他两类为负类计算OvR ROC。采用各类别预测概率，不使用最大置信度或硬分类标签。Macro OvR AUC为三个类别AUC的算术平均，取值为0–1。AUC同时经过经验ROC梯形面积及正负样本排序两种独立方法核对。原始float16舍入概率保持不变，没有重新归一化、平滑或阈值优化。','','## 宏平均AUC比较','','| Classifier | Original | Remapped |','|---|---:|---:|']
for model in models:
    pair = [next(r for r in records if r['Classifier']==model and r['Input']==v)['Macro OvR AUC'] for v in ['Original','Remapped']]
    lines.append(f'| {model} | {pair[0]:.4f} | {pair[1]:.4f} |')
lines += ['','每类AUC和完整精度数值见auc_summary.csv；全部经验阈值坐标见classwise_empirical_roc_long.csv。各roc_*.csv为可供后续EditaPlot绘图的已计算坐标，FPR及TPR均为0–1。宏平均ROC通过各类别经验折线的逐点平均构建，保留竖直跳跃，不用任意密集网格平滑；曲线面积与各类别AUC均值一致。','','## 解释范围','','这是单次seed43和固定推理噪声下、选定六种分类器的结果。Stage 1预训练使用全部6199张目标水印图，包含当前分类验证/测试样本的目标，且水印方向由真实类别确定。因此AUC不能解释为全流程严格独立的泛化性能。本次不计算置信区间或显著性检验，也不修改Word、MD或原始实验文件。','','## 方法参考','','- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html','- https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html','']
(OUT/'ROC_AUC_ANALYSIS.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({'output':str(OUT),'class_counts':dict(zip(CLASSES,COUNTS,strict=True)),'validation_passed':True,'independent_auc_checks':36,'results':records},ensure_ascii=False,indent=2))

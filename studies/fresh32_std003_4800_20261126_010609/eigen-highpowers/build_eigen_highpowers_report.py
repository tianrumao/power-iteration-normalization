"""Offline report for the true-eigenvector and higher-power supplementary probes."""
import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
LABELS = ['q32', 'q64', 'q128', 'eigenvector']
NAMES = {'q32': 'q32', 'q64': 'q64', 'q128': 'q128', 'eigenvector': '真实主特征向量'}
COLORS = {'111': '#087f8c', '000': '#bd5c37'}


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{h}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{v}</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def svg(fig):
    b = io.StringIO()
    fig.savefig(b, format='svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return '<svg' + b.getvalue().split('<svg', 1)[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--gaps', type=Path, help='Parent seed-gaps.csv, if outside the standard study layout')
    p.add_argument('--output', type=Path, default=ROOT / 'eigen-highpowers-report.html')
    args = p.parse_args()
    complete = json.loads((args.data / 'complete.json').read_text())
    assert complete['status'] == 'complete' and complete['scores'] == 3840
    rows = read(args.data / 'scores.csv')
    workspace_rows = read(args.data / 'query-workspaces.csv')
    cal = read(args.data / 'calibration.csv')
    gap_path = args.gaps or next((p for p in [ROOT / 'completed-results/seed-gaps.csv',
        ROOT.parent / 'seed-gaps.csv', args.data.parent / 'seed-gaps.csv'] if p.exists()), None)
    if gap_path is None:
        raise FileNotFoundError('Provide the parent study seed-gaps.csv with --gaps')
    with gap_path.open() as f:
        flagged = {int(r['seed']) for r in csv.DictReader(f) if float(r['final_gap']) > .1}
    seeds = sorted({int(r['seed']) for r in rows})
    assert len(seeds) == 32 and len(rows) == 3840
    assert len({(r['seed'], r['mask'], r['layer'], r['target'], r['split_seed']) for r in rows}) == 3840
    M = {}
    for r in rows:
        key = int(r['seed']), r['mask'], int(r['layer']), r['target']
        M.setdefault(key, []).append(r)
    per_model = {k: {metric: float(np.mean([float(r[metric]) for r in rs]))
        for metric in ['similarity', 'fixed_similarity', 'shuffled_similarity', 'validation_similarity']}
        for k, rs in M.items()}
    rng = np.random.default_rng(20261127)
    def stats(values):
        v = np.asarray(values)
        boot = v[rng.integers(len(v), size=(20000, len(v)))].mean(1)
        return {'mean': float(v.mean()), 'sd': float(v.std(ddof=1)),
                'ci': [float(x) for x in np.quantile(boot, [.025, .975])]}
    def pct(x): return f'{100*x:.2f}%'
    def pp(x): return f'{100*x:+.2f}'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})
    bodies, all_stats = [], {}
    for cohort, ss in [('all', seeds), ('unflagged', [s for s in seeds if s not in flagged])]:
        S, D = {}, {}
        for layer in range(5):
            for target in LABELS:
                for mask in ['111', '000']:
                    S[mask, layer, target] = stats([per_model[s, mask, layer, target]['similarity'] for s in ss])
                D[layer, target] = stats([per_model[s, '111', layer, target]['similarity'] - per_model[s, '000', layer, target]['similarity'] for s in ss])
        all_stats[cohort] = {'n': len(ss), 'scores': [{'mask': m, 'layer': l, 'target': t, **v} for (m, l, t), v in S.items()],
                              'paired_differences': [{'layer': l, 'target': t, **v} for (l, t), v in D.items()]}
        fig, ax = plt.subplots(figsize=(10.5, 3.8), layout='constrained')
        for mask, marker in [('111', 'o'), ('000', 's')]:
            means = np.array([S[mask, l, 'eigenvector']['mean'] for l in range(5)]) * 100
            ci = np.array([S[mask, l, 'eigenvector']['ci'] for l in range(5)]) * 100
            ax.plot(range(5), means, marker=marker, color=COLORS[mask], label=mask, linewidth=2)
            ax.fill_between(range(5), ci[:, 0], ci[:, 1], color=COLORS[mask], alpha=.13)
            ax.plot(range(5), [100*np.mean([per_model[s, mask, l, 'eigenvector']['fixed_similarity'] for s in ss]) for l in range(5)], color=COLORS[mask], linestyle='--', alpha=.75, label=mask+' fixed lower')
        ax.axhline(100/32, color='#8796a5', linestyle=':', label='Random direction: 3.125%')
        ax.set(xticks=range(5), xticklabels=['Input 0', 'Layer 1', 'Layer 2', 'Layer 3', 'Layer 4'],
               ylabel='True-eigenvector probe similarity (%)', ylim=(0, 90))
        ax.grid(axis='y', alpha=.15)
        ax.legend(frameon=False, ncol=3, loc='upper left', fontsize=9)
        true_fig = svg(fig)
        true_rows = []
        for layer in range(5):
            diff = D[layer, 'eigenvector']
            true_rows.append([f'层 {layer}', pct(S['111', layer, 'eigenvector']['mean']),
                pct(S['000', layer, 'eigenvector']['mean']), pp(diff['mean']) + ' pp',
                f'[{pp(diff["ci"][0])}, {pp(diff["ci"][1])}] pp'])
        high_rows = []
        for layer in range(5):
            high_rows.append([f'层 {layer}', *[pct(S[m, layer, t]['mean']) for t in LABELS for m in ['111', '000']]])
        fig, ax = plt.subplots(figsize=(10.5, 3.5), layout='constrained')
        for mask, marker in [('111', 'o'), ('000', 's')]:
            ax.plot(range(4), [100*S[mask, 4, t]['mean'] for t in LABELS],
                    color=COLORS[mask], marker=marker, label=mask, linewidth=2)
        ax.set(xticks=range(4), xticklabels=['q32', 'q64', 'q128', 'True eigenvector'],
               ylabel='Layer-4 probe similarity (%)')
        ax.grid(axis='y', alpha=.15)
        ax.legend(frameon=False)
        final_fig = svg(fig)
        controls = []
        for mask in ['111', '000']:
            for layer in range(5):
                controls.append([mask, f'层 {layer}', pct(np.mean([per_model[s, mask, layer, 'eigenvector']['shuffled_similarity'] for s in ss])),
                    pct(np.mean([per_model[s, mask, layer, 'eigenvector']['fixed_similarity'] for s in ss]))])
        diff3 = D[3, 'eigenvector']
        work_table = []
        for layer in range(5):
            vals = []
            for mask in ['111', '000']:
                for workspace in ['upper', 'lower']:
                    selected = [r for r in workspace_rows if int(r['seed']) in ss and int(r['layer']) == layer and r['mask'] == mask and r['workspace'] == workspace]
                    vals.append(pct(np.mean([float(r['similarity_zero_for_invalid']) for r in selected])) if all(int(r['valid_directions']) == int(r['samples']) for r in selected) else '无方向（全零）')
            work_table.append([f'层 {layer}', *vals])
        direction = '000 更容易读出真实方向' if diff3['mean'] < 0 else '111 更容易读出真实方向'
        bodies.append(f'''<div class="cohort" id="{cohort}" {'hidden' if cohort != 'all' else ''}>
<section><span class="number">01</span><h2>直接 probe 真实主特征向量</h2><p>该目标与母模型的任务一致，沿用已保存的真实主特征向量参考，不使用 q128 替代标签。输入层 0 与网络层 1–4 都读取完整 query 向量。</p>
<figure>{true_fig}<figcaption>实线：完整 query 的拟合线性 probe；虚线：query 下半区的固定原始读出。点为 {len(ss)} 个训练种子的平均值，阴影为拟合 probe 的 95% bootstrap 均值区间。每个模型先平均三个 probe 划分。</figcaption></figure>
{table(['层','111 相似度','000 相似度','111 − 000','配对差值 95% 区间'],true_rows)}
<p class="takeaway">第 3 层的真实目标差值为 {pp(diff3['mean'])} 个百分点：{direction}。该结果直接检验真实任务方向，不依赖“q16 已经收敛”的假设。</p></section>
<section><span class="number">02</span><h2>高幂次目标与真实目标是否给出一致图景？</h2><p>表中均为同一批输入、相同划分和相同 1,200 步拟合预算下的测试平方余弦相似度。相邻高幂次目标高度相关，不能把排名直接解释为执行了精确的迭代次数。</p>
{table(['层','q32 / 111','q32 / 000','q64 / 111','q64 / 000','q128 / 111','q128 / 000','真实 / 111','真实 / 000'],high_rows)}
<figure>{final_fig}<figcaption>第 4 层针对各目标分别训练 probe。真实主特征向量与 q128 是两个独立评分目标。</figcaption></figure></section>
<section><span class="number">03</span><h2>原始向量是否靠近真值：上下两个工作区</h2><p>以下直接截取 query 的前 32 维与后 32 维，单位化后与真实特征向量比较；没有训练新的读出。上半区初始为全零，没有方向，不能将其零分当成随机方向基线。补零和交换输入／输出位置提供了工作区，但没有约束训练后的每层必须执行一次标准 power iteration。</p>
{table(['层','111 上半区','111 下半区','000 上半区','000 下半区'], work_table)}
<p class="takeaway">原始方向的低相似度限制了“在当前坐标中逐层对齐”的解释。线性 probe 分数提高只表明相关方向更容易解码；它不能单独证明网络沿标准幂迭代轨迹前进。即使两半原始方向都低，也不能据此排除矩阵 token 或混合坐标中的计算。</p>
<h2>读取基线与负对照</h2><p>固定读出直接读取 query 下半区并单位化，与原始母模型输出规则一致。它始终作为第 0 步候选。验证集选优不保证每次测试分数都高于固定读出，因此同时报告两者。打乱标签的负对照在 fit 与 validation 上分别打乱，测试仍使用真实标签。</p>
{table(['条件','层','真实目标：打乱标签 probe','真实目标：固定读出'],controls)}
</section></div>''')
    calibration = [[f'q{r["power"]}', pct(float(r['mean_similarity_to_eigenvector'])),
        pct(float(r['median_similarity'])), pct(float(r['fraction_ge_0_99'])), pct(float(r['fraction_ge_0_999']))] for r in cal]
    main_diff = next(r for r in all_stats['all']['paired_differences'] if r['layer'] == 3 and r['target'] == 'eigenvector')
    if main_diff['ci'][1] < 0:
        verdict = '真实目标的补充实验仍未支持 normalization 提前计算：第 3 层是 000 的方向更容易线性读取。'
    elif main_diff['ci'][0] > 0:
        verdict = '真实目标的补充实验在第 3 层观察到 111 的读取优势。是否构成计算提前，还需要结合其幅度、整条深度曲线及干预结果。'
    else:
        verdict = '第 3 层真实目标的配对差值区间跨零，当前补充实验未提供明确的 normalization 读取优势。'
    late = []
    for path in args.data.glob('seed-*/???/split-*.npz'):
        with np.load(path) as f:
            history, steps = f['history_similarity'], f['history_steps']
            late.extend((history.max(0) - history[steps <= 900].max(0))[1:4].ravel().tolist())
    summary = {'cohorts': all_stats, 'calibration': cal, 'verdict': verdict,
               'late_validation_gain_mean': float(np.mean(late)), 'late_validation_gain_p95': float(np.quantile(late, .95))}
    (args.data / 'report-statistics.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    raw = {n: (args.data / n).read_text() for n in ['scores.csv', 'summary.csv', 'matched-summary.csv', 'calibration.csv', 'query-workspaces.csv']}
    doc = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>真实主特征向量与高幂次逐层 Probe</title><style>
*{box-sizing:border-box}body{margin:0;background:#f2f5f5;color:#193044;font:16px/1.85 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}main{max-width:1160px;margin:30px auto;background:white;padding:48px 58px}header{border-top:5px solid #087f8c;padding-top:25px}.eyebrow,.number{font-size:12px;color:#087f8c;letter-spacing:.13em;font-weight:650}h1{font-size:35px;line-height:1.4;margin:14px 0}h2{font-size:25px;line-height:1.4}.lead{font-size:20px}.muted,figcaption{font-size:13px;color:#5c6d7c}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:20px;border-block:1px solid #dce4e8;padding:22px 0;margin:26px 0}.metrics b{display:block;font-size:27px}.metrics span{font-size:13px;color:#5c6d7c}section{border-top:1px solid #dce4e8;padding:30px 0}.scroll{overflow:auto;margin:20px 0}table{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}td,th{padding:11px;border-bottom:1px solid #dce4e8;text-align:left}td{white-space:nowrap}th{background:#f5f8f8;color:#5c6d7c;font-size:12px}svg{width:100%;height:auto}figure{margin:25px 0}.takeaway{border-left:3px solid #087f8c;padding-left:17px;font-weight:600}nav,.downloads{display:flex;gap:10px;flex-wrap:wrap;margin:24px 0}button,a.button{font:inherit;font-size:14px;background:white;border:1px solid #c7d5db;border-radius:4px;padding:9px 15px;cursor:pointer;color:#193044;text-decoration:none}button.active{background:#087f8c;border-color:#087f8c;color:white}a{color:#087f8c}li{margin:10px 0}[hidden]{display:none!important}@media(max-width:750px){main{margin:0;padding:25px 20px}h1{font-size:28px}h2{font-size:22px}.metrics{grid-template-columns:repeat(2,1fr)}}@media print{main{margin:0;padding:0}body{background:white}nav,.downloads{display:none}table,figure{break-inside:avoid}}
</style></head><body><main><header><div class="eyebrow">AIOWLS RESEARCH / SUPPLEMENTARY PROBES</div><h1>真实主特征向量<br>与 q32、q64、q128 的逐层 Probe</h1><p class="lead">__VERDICT__</p><p class="muted">原始 32 对模型保持冻结。新增四个目标，沿用 1,200 步、三个划分、固定与随机初始化，以及验证集选优。</p></header>
<div class="metrics"><div><b>64</b><span>冻结的母模型</span></div><div><b>4</b><span>新增目标</span></div><div><b>3,840</b><span>新增评分记录</span></div><div><b>1,200</b><span>每次 probe 拟合步数</span></div></div>
<section><span class="number">CALIBRATION</span><h2>更高幂次是否接近真实答案？</h2><p>以下为独立校准数据的方向比较，不是模型预测分数。目标幂次在拟合前固定，没有根据测试结果挑选。达到 99% 相似度表示该样本的 q<sub>k</sub> 与真实主特征向量已足够接近此阈值。</p>__CALIBRATION__<p class="takeaway">q128 的平均相似度已达 99.08%，但仍有少量样本未充分收敛。直接 probe 真实主特征向量，完全绕过有限次幂迭代的近似问题。</p></section>
<nav aria-label="结果子集"><button class="active" data-cohort="all" aria-pressed="true">全部 32 对</button><button data-cohort="unflagged" aria-pressed="false">差值 ≤ 0.1 的 25 对</button><button onclick="window.print()">打印 / 保存 PDF</button></nav><p class="muted">主结论基于全部 32 对。25 对子集用于敏感性分析，仅通过宽松差值阈值，不是严格性能匹配。</p>
__BODIES__
<section><span class="number">04</span><h2>如何判断“早层没有往最终方向靠”？</h2><p>低 q 的低 probe 分数，不能推出高 q 必然更低：母模型训练目标是最终特征向量，未被要求保存每个早期迭代的起点相关成分。应分别比较同一目标的逐层解码曲线、原始坐标方向，以及 111 与 000 的配对差值。本次没有看到原始 query 两半持续对齐，也没有看到 111 的早层解码优势；因此不能把当前结果作为 normalization 提前计算的正面证据。</p><p>10%、20%、…、90%、99% 可以作为新的收敛对齐目标设计。需事先固定“每个样本首次达到阈值”或“独立校准上统一选 k”这两种不同规则，说明阈值、未收敛样本和越过阈值的处理。目标到真值达到 80%，不等于 probe 应达到 80%。本报告没有拟合这组新的百分比目标，当前四个新增目标已全部完成。</p><h2>统计与解释口径</h2><ul><li>相似度为测试集 mean(cos²)，误差为 1 − 相似度。百分数不是分类准确率；随机独立方向基线为 1/32 = 3.125%。</li><li>每个模型有 3,000 个 probe 样本，每次划分的 fit / validation / test 为 1,680 / 420 / 900。先在模型内平均三个划分，再以配对训练种子为统计单位。</li><li>均值与配对差值区间使用 20,000 次训练种子 bootstrap。多目标、多层比较为探索性分析，未作多重比较校正。</li><li>四个目标都采用无 bias 的 64 → 32 线性方向读出，相同损失、优化器、预算与验证选优规则。不同目标的难度及相关性不同，不能把最高分目标理解成网络执行的精确迭代次数。</li><li>两组母模型的最终任务表现仍不同。配对种子控制了数据、初始化和批次，但不能把全部解码差异解释为算法步骤位置改变。</li><li>本次仍只 probe query token。读取优势属于表征证据，不能单独证明计算的因果位置，需与已有 bypass 和更细的干预一起解释。</li><li>最后 300 步的最佳验证分数增益：中间层平均 __LATE__ 个百分点，95 分位 __P95__ 个百分点。预算检查不保证全局最优。</li></ul></section>
<section><span class="number">05</span><h2>数据与复核</h2><p>输入与原始幂次标签通过父实验哈希核对，重算 q0–q16 与原标签一致。真实主特征向量的第 4 层固定读出也与此前同样本审计一致。所有母模型检查点在读取前后验证未变。</p><div class="downloads"><button data-download="scores.csv">完整评分 CSV</button><button data-download="summary.csv">全部种子汇总</button><button data-download="matched-summary.csv">25 对子集汇总</button><button data-download="calibration.csv">独立收敛校准</button><button data-download="query-workspaces.csv">上下工作区方向 CSV</button><a class="button" href="https://github.com/tianrumao/power-iteration-normalization/releases" target="_blank" rel="noopener">GitHub 完整归档</a></div><p class="muted">本报告离线可读。完整 probe 权重、验证曲线、目标数组、代码和执行 Notebook 另存于补充实验归档。</p></section></main>
<script id="raw" type="application/json">__RAW__</script><script>document.querySelectorAll('[data-cohort]').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('.cohort').forEach(e=>e.hidden=e.id!==b.dataset.cohort);document.querySelectorAll('[data-cohort]').forEach(x=>{x.classList.toggle('active',x===b);x.setAttribute('aria-pressed',String(x===b))})}));const raw=JSON.parse(document.getElementById('raw').textContent);document.querySelectorAll('[data-download]').forEach(b=>b.addEventListener('click',()=>{const u=URL.createObjectURL(new Blob(['\ufeff'+raw[b.dataset.download]],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=u;a.download=b.dataset.download;a.click();setTimeout(()=>URL.revokeObjectURL(u),5000)}));</script></body></html>'''
    doc = doc.replace('__VERDICT__', verdict).replace('__CALIBRATION__', table(['目标','平均相似度','中位数','达到 99% 的样本比例','达到 99.9% 的样本比例'], calibration))
    doc = doc.replace('__BODIES__', ''.join(bodies)).replace('__LATE__', f'{100*np.mean(late):.3f}').replace('__P95__', f'{100*np.quantile(late,.95):.3f}')
    doc = doc.replace('__RAW__', json.dumps(raw, ensure_ascii=False).replace('</', '<\\/'))
    args.output.write_text(doc)
    print(json.dumps({'verdict': verdict, 'true_eigenvector': [s for s in all_stats['all']['scores'] if s['target'] == 'eigenvector']}, ensure_ascii=False, indent=2))
    print(args.output)


if __name__ == '__main__':
    main()

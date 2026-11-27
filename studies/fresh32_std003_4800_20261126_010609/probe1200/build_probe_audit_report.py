"""Build an offline report for the probe-budget and target-definition audit."""
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


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{h}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{v}</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def image(fig):
    buf = io.StringIO()
    fig.savefig(buf, format='svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return '<svg' + buf.getvalue().split('<svg', 1)[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, default=ROOT / 'probe-convergence-report.html')
    args = p.parse_args()
    new = read(args.data / 'power-probes.csv')
    old = read(ROOT / 'completed-results/power-probes.csv')
    target = read(args.data / 'same-example-target-audit.csv')
    conv = read(args.data / 'convergence.csv')
    audit = json.loads((args.data / 'audit-complete.json').read_text())
    assert len(new) == len(old) == len(conv) == 7680 and len(target) == 192
    def avg(rows, key):
        return float(np.mean([float(r[key]) for r in rows]))
    def select(rows, mask, layer, power):
        return [r for r in rows if r['mask'] == mask and int(r['layer']) == layer and int(r['power']) == power]
    def mean(rows, mask, layer, power):
        return avg(select(rows, mask, layer, power), 'similarity')
    keys = ('seed', 'mask', 'layer', 'power', 'split_seed')
    O = {tuple(r[k] for k in keys): r for r in old}
    assert all(abs(float(r['fixed_similarity']) - float(O[tuple(r[k] for k in keys)]['fixed_similarity'])) < 1e-10 for r in new)
    original_cap = sum(r['best_step'] == '300' for r in old if int(r['layer']) in (1, 2, 3))
    target_table = []
    for mask in ('111', '000'):
        rr = [r for r in target if r['mask'] == mask]
        pp = select(new, mask, 4, 16)
        target_table.append([mask, f'{100*avg(rr,"same_examples_fixed_vs_true_eigenvector"):.2f}%',
            f'{100*avg(rr,"same_examples_fixed_vs_q16"):.2f}%', f'{100*avg(pp,"similarity"):.2f}%',
            f'{100*avg(rr,"same_examples_q16_vs_true_eigenvector"):.2f}%',
            f'{sum(r["best_step"]=="0" and r["restart"]=="fixed" for r in pp)}/96'])
    qrows = []
    for layer in range(5):
        row = [f'层 {layer}']
        for mask in ('111', '000'):
            o, n = mean(old, mask, layer, 16), mean(new, mask, layer, 16)
            row.extend([f'{100*o:.2f}%', f'{100*n:.2f}%', f'{100*(n-o):+.2f} pp'])
        qrows.append(row)
    best_rows = []
    powers = (1, 2, 3, 4, 6, 8, 11, 16)
    for layer in range(5):
        row = [f'层 {layer}']
        for mask in ('111', '000'):
            winner = max(powers, key=lambda k: mean(new, mask, layer, k))
            row.append(f'q{winner}：{100*mean(new,mask,layer,winner):.2f}%')
        best_rows.append(row)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(10, 3.8), layout='constrained')
    colors = {'111': '#087f8c', '000': '#bd5c37'}
    for mask in ('111', '000'):
        for rows, style, label in ((old, '--', '300'), (new, '-', '1200')):
            ax.plot(range(5), [100*mean(rows, mask, l, 16) for l in range(5)], style,
                    color=colors[mask], marker='o' if mask == '111' else 's', label=f'{mask}, {label} steps')
    ax.set(xticks=range(5), xlabel='Layer (0 = input)', ylabel='q16 test squared-cosine similarity (%)', ylim=(0, 75))
    ax.grid(axis='y', alpha=.15)
    ax.legend(frameon=False, ncol=2)
    qfig = image(fig)
    fig, ax = plt.subplots(figsize=(10, 3.6), layout='constrained')
    times = np.arange(0, 1201, 25)
    for mask in ('111', '000'):
        curves = []
        for folder in sorted(args.data.glob(f'seed-*/{mask}')):
            for path in sorted(folder.glob('split-*.npz')):
                f = np.load(path)
                hs, steps = f['history_similarity'], f['history_steps']
                curves.append([hs[steps <= t].max(0)[1:4].mean() for t in times])
        ax.plot(times, 100*np.mean(curves, axis=0), color=colors[mask], label=mask)
    ax.axvline(300, color='#64748b', linestyle='--', linewidth=1)
    ax.set(xlabel='Probe update budget', ylabel='Best validation similarity (%)', title='Intermediate layers 1-3, all eight targets')
    ax.grid(axis='y', alpha=.15)
    ax.legend(frameon=False)
    convergence_fig = image(fig)
    qdiff = mean(new, '111', 3, 16) - mean(new, '000', 3, 16)
    cap = audit['intermediate_layer_at_cap']
    late = audit['late_validation_gain_mean']
    summary = {'q16_layer4': {m: mean(new, m, 4, 16) for m in ('111', '000')},
               'layer3_q16_on_minus_off': qdiff, 'intermediate_at_cap_300': original_cap,
               'intermediate_at_cap_1200': cap, 'maps': 4608,
               'late_validation_gain_mean': late, 'target_table': target_table}
    (args.data / 'report-summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    text = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Linear probe 预算与目标核对</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f2f5f5;color:#193044;font:16px/1.85 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}}main{{max-width:1100px;margin:30px auto;padding:48px 58px;background:white}}h1{{font-size:34px;line-height:1.4}}h2{{font-size:24px;margin-top:36px}}h3{{font-size:19px}}.eyebrow{{color:#087f8c;font-size:12px;letter-spacing:.12em}}.lead{{font-size:20px}}.callout{{border-left:3px solid #087f8c;padding-left:18px}}.scroll{{overflow:auto}}table{{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}}td,th{{padding:11px;border-bottom:1px solid #dce4e8;text-align:left;white-space:nowrap}}th{{color:#5c6d7c;background:#f5f8f8;font-size:12px;white-space:normal}}svg{{width:100%;height:auto}}section{{border-top:1px solid #dce4e8;padding:15px 0}}.muted{{color:#5c6d7c;font-size:13px}}a{{color:#087f8c}}code{{font-size:.9em}}@media(max-width:700px){{main{{padding:25px 20px;margin:0}}h1{{font-size:28px}}}}@media print{{body{{background:white}}main{{margin:0;padding:0}}figure,table{{break-inside:avoid}}}}</style></head><body><main>
<div class="eyebrow">AIOWLS RESEARCH / PROBE AUDIT</div><h1>为什么第 4 层 q16 分数<br>不等于最终任务分数？</h1><p class="lead">主要原因是目标不同。把 probe 从 300 步延长到 1,200 步后，第 4 层 q16 的分数几乎不变，大部分划分仍选择原始固定读出；中间层的读取分数则需要以长预算版本重新评估。</p>
<p class="muted">32 对原模型全部冻结；64 个模型的 probe、3 个数据划分、8 个目标、5 个层全部重新拟合，共 7,680 条评分。旧版本完整保留。</p>
<section><h2>1. 在完全相同的样本上比较</h2><p>对同一个最终输出向量，分别用真实主特征向量 u₁ 和 q16 = normalize((XXᵀ)¹⁶a) 评分。以下所有列都使用同一批 probe 测试样本，消除了母模型测试池与 probe 测试池不同的影响。</p>
{table(['条件','固定输出对真实 u₁','固定输出对 q16','1,200 步 probe 对 q16','q16 本身对真实 u₁','q16 选择固定读出第 0 步'],target_table)}
<p class="callout">原始固定读出始终作为第 0 步候选。大部分划分仍选择它，具体数量见表。少量划分选择了验证集分数更高的后续权重，其测试集分数可能略降：验证集选优不保证每一次测试分数都不低于固定读出。这种微小差异解释不了对真实主特征向量与对 q16 的大幅分数差。</p>
<p>母模型只优化真实主特征向量，没有被训练成精确输出第 16 次幂迭代方向。有限次迭代仍含非主特征方向成分，而母模型可能已经抑制其中一部分。即使一个完美模型输出 u₁，它对 q16 的分数也只有 cos²(u₁,q16)，不必等于 100%。</p>
<p>校准数据中 q16 对真实 u₁ 的平均相似度为 77.68%；它本身就不是最终标签。平方余弦相似度也不是“答对样本占比”。</p></section>
<section><h2>2. 300 步是否足够？</h2><p>旧预算中，第 1–3 层的 {original_cap}/4,608 个 probe 最佳权重落在 300 步上限（{100*original_cap/4608:.1f}%），因此进行了全部重跑。1,200 步版本中，落在新上限的为 {cap}/4,608（{100*cap/4608:.1f}%）。</p>
<figure>{convergence_fig}</figure><p class="muted">曲线在每个预算下，对固定/随机初始化及此前的验证检查点取最佳，再平均第 1–3 层与 8 个目标。竖线为旧预算。测试集不参与选权重。</p>
<p>最后 300 步（900 → 1,200）的最佳验证相似度增益，平均为 {100*late:.3f} 个百分点，95 分位为 {100*audit['late_validation_gain_p95']:.3f} 个百分点。该检查描述剩余预算敏感性，不证明找到了所有线性 probe 的全局最优。</p>
<h3>q16 的完整预算对照</h3>{table(['层','111：300 步','111：1,200 步','变化','000：300 步','000：1,200 步','变化'],qrows)}<figure>{qfig}</figure>
<p>第 4 层没有因增加 probe 步数而获得更高的 q16 测试分数，不能把它与最终任务分数的差距归因于只跑了 300 步。中间层发生的变化则说明，旧预算的细小早层差值不应作过强解释。</p></section>
<section><h2>3. 长预算下，每层最高的目标</h2>{table(['层','111 最高目标','000 最高目标'],best_rows)}<p>层 3 的 q16 相似度差（111 − 000）为 {100*qdiff:+.2f} 个百分点。最高目标的排名仍是描述性指标，不能直接当作已执行的幂迭代次数。</p></section>
<section><h2>4. 可复核性与归档</h2><p>母模型的 selected.pt 只读加载，未重新训练。所有 probe 输入与幂次标签均对照父实验保存的 SHA-256 检查通过；同样本的原始第 4 层 q16 固定读出也通过一致性检查。</p><p>新版本还保存了 32 个 probe 数据池：输入和幂次标签恢复出的数组与原哈希一致，主特征向量参考由同一矩阵重新计算并明确标注来源。原始母模型训练数组未因本次分析重新生成。</p><p>数据文件：<code>budget-comparison.csv</code>、<code>convergence.csv</code>、<code>same-example-target-audit.csv</code>、<code>power-probes.csv</code>、各模型的 <code>split-*.npz</code> 与 <code>verified-probe-data/</code>。</p><p><a href="https://github.com/tianrumao/power-iteration-normalization/releases/tag/fresh32-4800-20261126">父实验完整归档</a></p></section></main></body></html>'''
    args.output.write_text(text)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(args.output)


if __name__ == '__main__':
    main()

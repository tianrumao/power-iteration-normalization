"""Offline report of matched pilot endpoints and measured generalization gaps."""
import argparse
import csv
import io
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+str(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def svg(fig,path):
    fig.savefig(path,dpi=180,bbox_inches='tight',facecolor='white')
    b=io.StringIO();fig.savefig(b,format='svg',bbox_inches='tight',facecolor='white');plt.close(fig)
    return '<svg'+b.getvalue().split('<svg',1)[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,default=ROOT/'matched-pilot-report.html');a=p.parse_args()
    repaired=a.data/'condition_specific';shared=a.data/'shared_lr001'
    if not repaired.exists():repaired=shared
    result=json.loads((repaired/'pilot-complete.json').read_text())
    audit=json.loads((repaired/'generalization-audit/complete.json').read_text())
    integrity=json.loads((repaired/'generalization-audit/leakage-and-numerics.json').read_text())
    rows=read(repaired/'generalization-audit/split-scores.csv')
    gaps=read(repaired/'generalization-audit/train-test-gaps.csv')
    paired=read(repaired/'generalization-audit/paired-split-differences.csv')
    old=read(shared/'generalization-audit/split-scores.csv')
    assert len(rows)==16 and len(gaps)==4 and len(paired)==8 and audit['status']=='complete'
    assert all(int(r['samples'])==(65536 if r['split']=='train' else 8192) for r in rows)
    values={(int(r['seed']),r['mask'],r['split']):float(r['mean_similarity']) for r in rows}
    steps={(int(r['seed']),r['mask']):int(r['checkpoint_step']) for r in rows}
    seeds=sorted({int(r['seed']) for r in rows});assert seeds==[9003,9004]
    assert all(x['shared_matrices']==0 for r in integrity['records'] for x in r['overlaps'])
    assert all(l['finite_coordinates'] for r in integrity['records'] for m in r['models'] for l in m['layers'])
    pct=lambda v:f'{100*float(v):.2f}%'
    pp=lambda v:f'{100*float(v):+.2f}'
    main_rows=[]
    for seed in seeds:
        for mask in ['111','000']:
            g=next(r for r in gaps if int(r['seed'])==seed and r['mask']==mask)
            main_rows.append([seed,mask,steps[seed,mask],*[pct(values[seed,mask,s]) for s in ['train','selection_validation','confirmation_validation','test']],pp(g['train_minus_test'])+' pp'])
    match_rows=[]
    for seed in seeds:
        ds={s:values[seed,'111',s]-values[seed,'000',s] for s in ['selection_validation','confirmation_validation','test']}
        match_rows.append([seed,*[pp(ds[s])+' pp' for s in ds], '≤1 pp' if abs(ds['test'])<=.01 else ('≤2.5 pp' if abs(ds['test'])<=.025 else '未匹配')])
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','font.size':11})
    fig,ax=plt.subplots(figsize=(9,2.9),layout='constrained')
    z=np.array([[100*(values[s,'111',split]-values[s,'000',split]) for split in ['selection_validation','confirmation_validation','test']] for s in seeds])
    im=ax.imshow(z,cmap='BrBG',vmin=-10,vmax=10,aspect='auto')
    ax.set(xticks=range(3),xticklabels=['Selection validation','Independent confirmation','Held-out test'],yticks=range(2),yticklabels=[f'Seed {s}' for s in seeds],title='Final-task difference: 111 minus 000')
    for i in range(2):
        for j in range(3):ax.text(j,i,f'{z[i,j]:+.2f} pp',ha='center',va='center',color='#193044',fontsize=13)
    fig.colorbar(im,ax=ax,label='Percentage points (fixed ±10 pp scale)')
    endpoint_fig=svg(fig,a.data/'matched-endpoint-differences.png')
    fig,axs=plt.subplots(1,2,figsize=(12,3.6),layout='constrained',sharey=True)
    for ax,seed in zip(axs,seeds):
        for folder,mask,color,style,label in [(shared,'111','#287f89','-','111, lr .001'),(shared,'000','#a6a6a6',':','000, lr .001 (failed)'),(repaired,'000','#b86236','-','000, lr .0005')]:
            history=json.loads((folder/f'seed-{seed}'/mask/'history.json').read_text())
            ax.plot([r['step'] for r in history],[100*r['mean_similarity'] for r in history],color=color,linestyle=style,label=label)
        ax.axhspan(74.5,75.5,color='#d9e5dc',alpha=.65)
        for mask,color in [('111','#287f89'),('000','#b86236')]:
            ax.scatter([steps[seed,mask]],[100*values[seed,mask,'selection_validation']],color=color,s=45,zorder=5)
        ax.set(title=f'Pilot seed {seed}',xlabel='Mother-model training updates',ylabel='Selection-validation similarity (%)',ylim=(0,85))
        ax.grid(axis='y',alpha=.15)
    axs[0].legend(frameon=False,fontsize=9,loc='lower right')
    curves=svg(fig,a.data/'matched-learning-curves.png')
    interval_rows=[[r['seed'],r['mask'],pp(r['train_minus_test'])+' pp',f"[{pp(r['approx_ci_low'])}, {pp(r['approx_ci_high'])}] pp"] for r in gaps]
    old_values={(int(r['seed']),r['mask'],r['split']):float(r['mean_similarity']) for r in old}
    old_rows=[[s,m,*[pct(old_values[s,m,split]) for split in ['train','selection_validation','confirmation_validation','test']]] for s in seeds for m in ['111','000']]
    raw={name:(repaired/'generalization-audit'/name).read_text() for name in ['split-scores.csv','train-test-gaps.csv','paired-split-differences.csv']}
    summary={'pilot_pairs':2,'matching_result':result,'test_gaps':[abs(values[s,'111','test']-values[s,'000','test']) for s in seeds],
        'normalization_train_minus_test':[values[s,'111','train']-values[s,'111','test'] for s in seeds],
        'test_used_for_checkpoint_selection':False,'full_cohort_started':False,'new_linear_probes_fitted':False,
        'integrity_checks':{'exact_matrix_overlap_across_splits':0,'nonfinite_coordinates_in_audited_states':0},
        'interpretation':'Two pilot pairs demonstrate achievable close final performance; they do not establish absence of overfitting or a normalization computation-depth effect.'}
    (a.data/'pilot-report-summary.json').write_text(json.dumps(summary,indent=2))
    doc='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>性能匹配试跑与泛化检查</title><style>
*{box-sizing:border-box}body{margin:0;background:#f2f5f5;color:#193044;font:16px/1.85 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}main{max-width:1200px;margin:25px auto;background:white;padding:38px 45px}header{border-top:5px solid #287f89;padding-top:20px}h1{font-size:31px;margin:10px 0}h2{font-size:23px}.lead{font-size:20px}.muted,figcaption{font-size:13px;color:#5b6e7c}.note{border-left:3px solid #b86236;background:#fff7f1;padding:12px 18px}.scroll{overflow:auto}table{border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums;width:100%}th,td{padding:10px 12px;border-bottom:1px solid #dbe4e7;text-align:left;white-space:nowrap}th{background:#f2f6f6}section{border-top:1px solid #dbe4e7;margin-top:30px;padding-top:20px}figure{margin:25px 0}svg{width:100%;height:auto}details{margin:20px 0}summary{cursor:pointer;color:#287f89}nav{display:flex;gap:9px;flex-wrap:wrap;margin:20px 0}button,a.button{font:inherit;font-size:14px;background:white;color:#193044;border:1px solid #bdced4;border-radius:4px;padding:8px 14px;cursor:pointer;text-decoration:none}a{color:#287f89}li{margin:8px 0}@media(max-width:700px){main{margin:0;padding:24px 16px}h1{font-size:26px}}@media print{nav{display:none}main{padding:0;margin:0}body{background:white}}</style></head><body><main><header><div class="muted">AIOWLS RESEARCH · 独立试跑 · SEEDS 9003 / 9004</div><h1>最终表现可以匹配<br>泛化差距仍需单独检查</h1><p class="lead">两对试跑的测试集实测差距为 0.90、0.25 个百分点，均在 1 个百分点以内。111 的训练—测试差距仍为 2.75、3.08 个百分点，不能宣称已经排除过拟合。</p><p class="note">这只是两对模型的训练方案试跑，尚未扩大为完整实验，也未对新模型拟合 linear probe。当前没有新的逐层计算位置结论。</p></header>
<section><h2>冻结检查点后的四份数据评分</h2><p>以下选择的是共同 75% 验证水平附近的检查点。它们与此前报告的 12,800 步训练终点不同，两类结果都保留。检查点选择只使用第一份验证集，随后才评分独立确认集和测试集。</p>__MAIN_TABLE__<p class="muted">训练集 65,536 个样本；选择验证集、独立确认集和测试集各 8,192 个。指标为 mean(cos²)，不是分类准确率。</p></section>
<section><h2>最终任务的配对差距</h2>__MATCH_TABLE__<figure>__ENDPOINT_FIG__<figcaption>这里只显示母模型最终任务的匹配诊断，不是逐层 probe 热力图。颜色以实测 111 − 000 差值绘制，0 为中性色，固定色标为 ±10 个百分点；数字没有人为清零。</figcaption></figure><p>匹配主门槛固定为 1 个百分点，2.5 个百分点仅作敏感性口径。小于门槛是对这批样本均值的描述，不等于总体差异已被证明为零；只有两对试跑，不能替代完整独立重复。</p></section>
<section><h2>训练—测试差距与过拟合问题</h2>__GAP_TABLE__<p>表中区间仅作当前样本分数波动的近似描述。训练样本参与了模型拟合，不能据此作严格的总体过拟合检验，也没有计入训练种子的不确定性。111 的测试表现确实接近其验证表现，但训练集分数仍更高；应如实报告泛化差距，继续检查数据量与正则化，而不能用验证分数代替测试结果。</p></section>
<section><h2>训练设置和检查点如何选择？</h2><p>模型、数据生成规则、输入编码和归一化实现沿用原实验。当前 normalizer 是前 3 个 block 后的 token 列向量 L2 单位化。两组共享初始化、数据和批次计划；按条件调整学习率：111 为 0.001，000 为 0.0005，最多 12,800 步，临近目标时每 50 步保存权重。</p><p>先在独立试跑中依次检查预先规定的共同水平 78%、75%，每组候选需处于目标 ±0.5 个百分点。选定所有检查点后，才打开独立确认验证集，并要求每对差距 ≤1 个百分点。此次共同水平为 75%，两对确认集差距为 0.20、0.59 个百分点。</p><figure>__CURVES__<figcaption>曲线仅来自选择验证集。点为最终选择的检查点；不同组所用训练步数在主表中完整列出。</figcaption></figure><details><summary>查看统一 0.001 学习率的失败试跑终点</summary>__OLD_TABLE__<p>该参数方案未达到共同性能水平，保留为诊断记录，不作为性能匹配结果。此前的约 76% 测试表现对应此表中的 111 训练终点。</p></details></section>
<section><h2>检查到的完整性证据</h2><ul><li>按 X 矩阵逐样本 SHA-256 检查，训练、选择验证、独立确认和测试之间未发现完全相同的矩阵；各集合内部也未发现完全重复矩阵。</li><li>每个冻结模型在前 128 个测试样本的输入及全部 4 层状态上，未发现非有限坐标。检查范围和数值尺度保存在 leakage-and-numerics.json。</li><li>所有评分前后均验证模型文件哈希未变。没有根据测试分数重新选择检查点。</li><li>这些检查支持“没有发现所检查的重复或数值异常”，不能证明不存在所有形式的过拟合或分布变化。</li></ul></section>
<section><h2>下一阶段的条件</h2><p>完整重跑仍需冻结训练方案、泛化差距的验收口径及原始输入的保存方式。最终测试集匹配与训练—测试泛化差距是两个独立条件，不能共用一个未经说明的阈值。后续逐层差值图将直接显示实际差值，以 0 为颜色中心。</p><p class="muted">此试跑的原始 .npy 数据已保存在 Google Drive；精确种子、生成器、数组哈希、全部检查点、验证历史及逐样本评分均已归档。紧凑归档不内嵌大型原始输入，RAW_DATA_INDEX.json 记录其持久存储位置。</p><nav><button data-download="split-scores.csv">四份数据完整评分</button><button data-download="train-test-gaps.csv">训练—测试差距</button><button data-download="paired-split-differences.csv">配对差值及区间</button><button onclick="window.print()">打印 / 保存 PDF</button></nav></section></main><script id="raw" type="application/json">__RAW__</script><script>const raw=JSON.parse(document.getElementById('raw').textContent);document.querySelectorAll('[data-download]').forEach(b=>b.onclick=()=>{const u=URL.createObjectURL(new Blob(['\\ufeff'+raw[b.dataset.download]],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=u;a.download=b.dataset.download;a.click();setTimeout(()=>URL.revokeObjectURL(u),5000)});</script></body></html>'''
    for token,value in [('__MAIN_TABLE__',table(['seed','条件','训练步数','训练集','选择验证','独立确认','测试集','训练 − 测试'],main_rows)),('__MATCH_TABLE__',table(['seed','选择验证差值','独立确认差值','测试差值','测试实测匹配'],match_rows)),('__GAP_TABLE__',table(['seed','条件','训练 − 测试','近似 95% 区间'],interval_rows)),('__ENDPOINT_FIG__',endpoint_fig),('__CURVES__',curves),('__OLD_TABLE__',table(['seed','条件','训练集','选择验证','独立确认','测试集'],old_rows)),('__RAW__',json.dumps(raw,ensure_ascii=False).replace('</','<\\/'))]:doc=doc.replace(token,value)
    a.output.write_text(doc);print(a.output)


if __name__=='__main__':main()

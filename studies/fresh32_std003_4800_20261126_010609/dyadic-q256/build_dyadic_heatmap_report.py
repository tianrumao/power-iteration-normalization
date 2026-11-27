"""Merge existing 1200-step probes with q256; display the requested dyadic heatmaps."""
import argparse
import csv
import io
import json
import hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
POWERS=[2**j for j in range(9)]


def read(path):
    with path.open() as f: return list(csv.DictReader(f))


def write(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def svg(fig):
    b=io.StringIO();fig.savefig(b,format='svg',bbox_inches='tight',facecolor='white');plt.close(fig)
    return '<svg'+b.getvalue().split('<svg',1)[1]


def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+str(v)+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in r)+'</tr>' for r in rows)+'</tbody></table></div>'


def heatmap(arrays, titles, *, difference=False, stackable=True, export=None):
    if len(arrays)>1 and stackable:
        return '<div class="pair">'+''.join('<div>'+heatmap([z],[t],difference=difference)+'</div>' for z,t in zip(arrays,titles))+'</div>'
    fig,axs=plt.subplots(1,len(arrays),figsize=(8*len(arrays),3.9),layout='constrained',squeeze=False)
    limit=max(1.,np.ceil(max(np.nanmax(np.abs(a)) for a in arrays))) if difference else 100
    for ax,z,title in zip(axs[0],arrays,titles):
        im=ax.imshow(z,cmap='BrBG' if difference else 'Blues',vmin=-limit if difference else 0,vmax=limit,aspect='auto')
        ax.set(xticks=range(9),xticklabels=POWERS,yticks=range(5),yticklabels=['Input 0','Layer 1','Layer 2','Layer 3','Layer 4'],xlabel='Power target k',title=title)
        for i in range(5):
            for j in range(9):
                ax.text(j,i,'—' if not np.isfinite(z[i,j]) else (f'{z[i,j]:+.1f}' if difference else f'{z[i,j]:.1f}'),ha='center',va='center',fontsize=10,
                    color='white' if (abs(z[i,j])>.58*limit if difference else z[i,j]>55) else '#152f42')
        ax.spines[:].set_visible(False)
    fig.colorbar(im,ax=list(axs[0]),shrink=.9,label='111 minus 000 (percentage points)' if difference else 'Mean squared cosine (%)')
    if export is not None:
        fig.savefig(export,dpi=180,bbox_inches="tight",facecolor="white")
        fig.savefig(export.with_suffix(".pdf"),bbox_inches="tight",facecolor="white")
    return svg(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--q256',type=Path,required=True)
    p.add_argument('--original',type=Path,default=ROOT/'probe1200-results')
    p.add_argument('--high',type=Path,default=ROOT/'eigen-highpowers-results')
    p.add_argument('--output',type=Path,default=ROOT/'dyadic-results')
    p.add_argument('--preview-incomplete',action='store_true',help='Development only: exclude missing q256 from chart by leaving it blank')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    rows,provenance=[],[]
    fields=['seed','mask','layer','power','split_seed','similarity','fixed_similarity','shuffled_similarity','validation_similarity','best_step','restart','source_study']
    sources=[('probe1200',a.original,[1,2,4,8,16]),('eigen-highpowers',a.high,[32,64,128]),('q256',a.q256,[256])]
    for name,folder,keep in sources:
        paths=sorted(folder.glob('seed-*/???/scores.csv'))
        if not paths and name=='q256' and a.preview_incomplete: continue
        assert len(paths)==64,(name,len(paths))
        for path in paths:
            provenance.append({'study':name,'path':path.relative_to(folder).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            for r in read(path):
                if r['power'] and int(r['power']) in keep:
                    assert tuple(int(r[k]) for k in ['fit_samples','validation_samples','test_samples'])==(1680,420,900)
                    item={k:r[k] for k in fields if k!='source_study'}
                    item['source_study']=name;rows.append(item)
    expected=8640 if not a.preview_incomplete else 7680
    assert len(rows)==expected,(len(rows),expected)
    keys=[(int(r['seed']),r['mask'],int(r['layer']),int(r['power']),int(r['split_seed'])) for r in rows]
    assert len(set(keys))==len(rows)
    expected_powers=POWERS[:-1] if a.preview_incomplete else POWERS
    assert set(keys)=={(seed,mask,layer,k,split) for seed in range(1001,1033) for mask in ['111','000'] for layer in range(5) for k in expected_powers for split in [0,1,2]}
    assert {k[0] for k in keys}==set(range(1001,1033))
    M={}
    for r in rows:
        key=int(r['seed']),r['mask'],int(r['layer']),int(r['power'])
        M.setdefault(key,[]).append(r)
    assert all(len(v)==3 for v in M.values())
    metrics=['similarity','fixed_similarity','shuffled_similarity']
    per_model={k:{m:float(np.mean([float(r[m]) for r in rr])) for m in metrics} for k,rr in M.items()}
    flagged={int(r['seed']) for r in read(ROOT/'completed-results/seed-gaps.csv') if float(r['final_gap'])>.1}
    rng=np.random.default_rng(20261127)
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','font.size':11})
    cohorts=[];summary=[]
    for cohort,seeds in [('all',list(range(1001,1033))),('unflagged',[s for s in range(1001,1033) if s not in flagged])]:
        matrices={}
        for metric in metrics:
            for mask in ['111','000']:
                matrices[metric,mask]=np.array([[100*np.mean([per_model[s,mask,l,k][metric] for s in seeds]) if (seeds[0],mask,l,k) in per_model else np.nan for k in POWERS] for l in range(5)])
        differences=matrices['similarity','111']-matrices['similarity','000']
        heatmap([matrices['similarity',m] for m in ['111','000']],[f'{m} | Fitted linear probe' for m in ['111','000']],stackable=False,export=a.output/f'dyadic-heatmaps-{cohort}.png')
        # Preserve the originally requested blue, numbered heatmaps as the primary view.
        pictures=[]
        for metric,title in [('similarity','Fitted linear probe'),('fixed_similarity','Fixed lower-query readout'),('shuffled_similarity','Shuffled-label control')]:
            pictures.append(f'<div class="metric" data-metric="{metric}" {"hidden" if metric!="similarity" else ""}>'+heatmap([matrices[metric,m] for m in ['111','000']],[f'{m} | {title}' for m in ['111','000']])+'</div>')
        deltafig=heatmap([differences],['Paired difference | fitted linear probe'],difference=True)
        for l in range(5):
            for k in POWERS:
                if (seeds[0],'111',l,k) not in per_model:continue
                delta=np.array([per_model[s,'111',l,k]['similarity']-per_model[s,'000',l,k]['similarity'] for s in seeds])
                ci=np.quantile(delta[rng.integers(len(seeds),size=(20000,len(seeds)))].mean(1),[.025,.975])
                for m in ['111','000']:
                    summary.append({'cohort':cohort,'n':len(seeds),'layer':l,'power':k,'mask':m,
                        'mean_similarity':float(np.mean([per_model[s,m,l,k]['similarity'] for s in seeds])),
                        'paired_111_minus_000':float(delta.mean()),'paired_ci_low':float(ci[0]),'paired_ci_high':float(ci[1])})
        highest=[]
        for l in range(5):
            rr=[str(l)]
            for m in ['111','000']:
                z=matrices['similarity',m][l];j=int(np.nanargmax(z));rr.extend([f'q{POWERS[j]}',f'{z[j]:.2f}%'])
            highest.append(rr)
        cohorts.append(f'<div class="cohort" id="{cohort}" {"hidden" if cohort!="all" else ""}><p class="muted">{len(seeds)} 对模型；各格先在模型内平均三个划分，再平均训练种子。</p><div class="scroll">'+''.join(pictures)+f'</div><details><summary>显示 111 − 000 的差值热力图</summary>{deltafig}<p>正数表示 111 更高；负数表示 000 更高。配对置信区间见可下载 CSV。</p></details><details><summary>显示各层最高分目标</summary>'+table(['层','111 最高目标','相似度','000 最高目标','相似度'],highest)+'<p>输入层也可能对低幂次得到高分。最高分目标受目标相关性、初始向量保留及解码难度影响，不能直接当作执行步数。</p></details></div>')
    write(a.output/'dyadic-scores.csv',rows);write(a.output/'dyadic-summary.csv',summary)
    (a.output/'dyadic-provenance.json').write_text(json.dumps({'powers':POWERS,'scores':len(rows),'reused_targets':POWERS[:-1],'newly_fitted_targets':[256],'probe_steps':1200,'mother_models_retrained':False,'sources':provenance},indent=2))
    audit=json.loads((a.output/'data-distribution-audit.json').read_text())
    calrows=[[f'q{r["power"]}',*[f'{100*r[k]:.2f}%' for k in ['mean','median','p10','p90','fraction_ge_0_99']]] for r in audit['powers']]
    actual=json.loads((a.output/'actual-probe-data-audit.json').read_text())
    sample_rows=read(a.output/'data-distribution-samples.csv')
    fig,axs=plt.subplots(1,2,figsize=(12,3.5),layout='constrained')
    for ax,k in zip(axs,[1,16]):
        v=100*np.array([float(r['similarity']) for r in sample_rows if int(r['power'])==k])
        ax.hist(v,bins=np.linspace(0,100,26),color='#4a8bbf',edgecolor='white')
        ax.axvline(v.mean(),color='#ac431e',linewidth=2,label=f'Mean {v.mean():.2f}%')
        ax.axvline(np.median(v),color='#1f675b',linestyle='--',linewidth=2,label=f'Median {np.median(v):.2f}%')
        ax.set(title=f'q{k} vs true eigenvector | calibration',xlabel='Squared cosine (%)',ylabel='Samples',xlim=(0,100));ax.legend(frameon=False,fontsize=9)
    distfig=svg(fig)
    strata=[[('谱比 λ2/λ1' if r['variable']=='spectral_ratio' else '初始平方对齐'),'低→高 '+str(r['quartile'])+ '/4',f'{r["low"]:.5f}–{r["high"]:.5f}',f'{100*r["q16_mean"]:.2f}%',f'{100*r["q16_median"]:.2f}%'] for r in audit['strata']]
    raw={name:(a.output/name).read_text() for name in ['dyadic-scores.csv','dyadic-summary.csv','data-distribution-summary.csv']}
    doc='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>q1–q256 逐层 Probe 热力图</title><style>
*{box-sizing:border-box}body{margin:0;background:#f2f5f7;color:#193044;font:16px/1.8 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}main{max-width:1580px;margin:24px auto;padding:32px 38px;background:white}header{border-top:5px solid #397bb0;padding-top:18px}h1{font-size:32px;margin:8px 0}h2{font-size:24px;margin:15px 0}p{max-width:1100px}.muted,figcaption{color:#536b7d;font-size:13px}nav{display:flex;gap:9px;flex-wrap:wrap;margin:20px 0}button,a.button{font:inherit;font-size:14px;padding:8px 15px;border:1px solid #c7d5df;border-radius:4px;background:white;color:#193044;cursor:pointer;text-decoration:none}button.active{background:#397bb0;color:white;border-color:#397bb0}section{border-top:1px solid #dbe4ea;margin-top:30px;padding-top:20px}.pair{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,580px),1fr));gap:20px}.metric svg{width:100%;height:auto}svg{width:100%;height:auto}.scroll{overflow-x:auto}details{margin:18px 0}summary{cursor:pointer;color:#28699a}table{border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}th,td{padding:9px 15px;border-bottom:1px solid #dbe4ea;text-align:left;white-space:nowrap}th{background:#f2f6f9}.note{border-left:3px solid #397bb0;padding-left:17px}a{color:#28699a}[hidden]{display:none!important}@media(max-width:700px){main{margin:0;padding:20px 15px}h1{font-size:25px}}@media print{nav{display:none}main{margin:0;padding:0}body{background:white}.metric svg{min-width:0}}</style></head><body><main><header><div class="muted">AIOWLS RESEARCH · 32 对冻结模型 · 1,200 步 Linear Probe</div><h1>q1–q256 的逐层 Probe 热力图</h1><p style="background:#fff2ed;border-left:4px solid #b44f31;padding:14px"><b>探索结果：最终性能未匹配。</b>同样本真实目标的固定读出，111 为 70.68%，000 为 77.76%，相差约 7.08 个百分点。此批结果不作为性能匹配的主比较。按验证集差距 ≤1 个百分点只有 2 对，≤2.5 个百分点只有 4 对；新的匹配训练另行记录。</p><p>目标按 2 的幂次取值：1、2、4、8、16、32、64、128、256。两张主图共用 0–100% 色标，每格标注测试集平均平方余弦相似度。幂次翻倍不代表到真值的相似度百分比均匀。</p><p class="muted">已完成的 q1–q128 probe 直接复用；仅 q256 新增拟合。母模型、输入、划分、1,200 步预算与验证集选优规则保持一致。完整网格共 8,640 条评分记录。</p></header>
<nav aria-label="结果子集"><button class="active" data-cohort="all">全部 32 对</button><button data-cohort="unflagged">旧宽松子集：≤10 个百分点（25 对）</button></nav><nav aria-label="评分方式"><button class="active" data-metric="similarity">拟合 Linear Probe</button><button data-metric="fixed_similarity">固定原始读出</button><button data-metric="shuffled_similarity">打乱标签对照</button></nav>
__COHORTS__<p class="note">主图展示“某个目标能否从该层线性读出”。输入层是必要基线；深层更像高幂次目标，不足以证明网络执行了标准 Power Method 或精确的幂次翻倍。25 对子集仅通过旧的 10 个百分点差值阈值，不是严格性能匹配；新主门槛为 1 个百分点，2.5 个百分点只用于敏感性分析。</p>
<p>固定同一个 X、更换随机初始向量 a 时，母模型的真实主特征向量目标不变，但有限幂次 q<sub>k</sub>(X,a) 会变。最终答案监督没有要求网络保留从给定 a 出发的每个中间迭代。因此，即使数据难度受控，仍需检验参考轨迹是否与模型采用的计算方式一致；“固定 X、更换 a”是可独立检验这一点的对照，本报告尚未运行该对照。</p><p class="muted">独立校准中 q128 与 q256 标签本身的平均平方余弦相似度为 __HIGH_CORRELATION__。高幂次列会越来越相似，不能仅按很小的最高分差异指定算法步数。</p><section><h2>为什么均值和中位数差这么大？</h2><p class="note">对实际保存的全部 32 个 probe 池、共 96,000 个输入也做了独立检查：零填充正确，单位范数仅有 float32 舍入误差；保存的真值方向及 q16 标签与独立谱分解计算一致。实际池的 q16 到真值相似度均值为 __ACTUAL_MEAN__，中位数为 __ACTUAL_MEDIAN__，10% 分位为 __ACTUAL_P10__。分布不均匀并非只出现在校准种子中。</p><p>下表和直方图是独立校准样本的 q<sub>k</sub> 与真实主特征向量之间的相似度，<b>不是模型或 probe 分数</b>。在相同 k 下，随机矩阵的谱比与随机初始方向不同，形成快、慢收敛样本的混合。小 k 时较高分的少数样本会抬高均值；较大 k 时慢收敛样本会拉低均值。均值与中位数的差距本身不说明实现错误，但说明固定 k 不是统一进度。</p>__CAL__<figure>__DIST__<figcaption>2,048 个独立校准样本，原生成规则与原校准种子。相同数据没有被筛选或替换。</figcaption></figure>
<details><summary>查看谱比和初始方向的分组诊断</summary>__STRATA__<p>每个变量分别按四分位分组，每组 512 个样本；这是描述性关联，不是控制其他变量后的因果效应。</p></details>
<p class="note">数据生成和标签审计：单位范数、特征向量残差，以及“谱分解直接求幂”与递推求幂的方向误差均在浮点误差量级；未发现明显数值错误。但现有随机分布的收敛难度不受控，不能把固定 q16 当成所有样本都达到同一百分比。</p>
<p>若下一步要研究统一的计算进度，可以单独设计谱比与初始对齐程度受控、分层平衡的新实验；也可以在现有分布上预先定义逐样本首次达到 10%–99% 阈值的标签。它们回答不同问题，不能将筛选后的容易样本分数当成原分布性能。本报告没有更换训练数据，也没有拟合这组百分比标签。</p></section>
<section><h2>数据下载与复核</h2><p class="muted">每个模型先平均三个划分，再以训练种子为统计单位；配对 95% 区间使用 20,000 次种子 bootstrap，多层多目标比较属于探索性分析，未作多重比较校正。随机独立方向基线是 3.125%；指标不是分类准确率。各格拟合权重仅按验证集选优；最高分目标之间可能高度相关。</p><nav><button data-download="dyadic-scores.csv">九目标完整评分 CSV</button><button data-download="dyadic-summary.csv">配对汇总和 95% 区间</button><button data-download="data-distribution-summary.csv">数据分布审计 CSV</button><button onclick="window.print()">打印 / 保存 PDF</button></nav><p class="muted">来源与哈希见 dyadic-provenance.json；新 q256 权重、验证曲线及目标数组单独归档，已有权重保留在之前的两个版本中。</p></section></main>
<script id="raw" type="application/json">__RAW__</script><script>
document.querySelectorAll('button[data-cohort]').forEach(b=>b.onclick=()=>{document.querySelectorAll('.cohort').forEach(x=>x.hidden=x.id!==b.dataset.cohort);document.querySelectorAll('button[data-cohort]').forEach(x=>x.classList.toggle('active',x===b))});
document.querySelectorAll('button[data-metric]').forEach(b=>b.onclick=()=>{document.querySelectorAll('div[data-metric]').forEach(x=>x.hidden=x.dataset.metric!==b.dataset.metric);document.querySelectorAll('button[data-metric]').forEach(x=>x.classList.toggle('active',x===b))});
const raw=JSON.parse(document.getElementById('raw').textContent);document.querySelectorAll('[data-download]').forEach(b=>b.onclick=()=>{const u=URL.createObjectURL(new Blob(['\\ufeff'+raw[b.dataset.download]],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=u;a.download=b.dataset.download;a.click();setTimeout(()=>URL.revokeObjectURL(u),5000)});
</script></body></html>'''
    doc=doc.replace('__COHORTS__',''.join(cohorts)).replace('__CAL__',table(['目标','均值','中位数','10% 分位','90% 分位','达到 99% 的比例'],calrows)).replace('__DIST__',distfig).replace('__STRATA__',table(['分组变量','四分位组','变量范围','q16 均值','q16 中位数'],strata)).replace('__RAW__',json.dumps(raw,ensure_ascii=False).replace('</','<\\/'))
    doc=doc.replace('__HIGH_CORRELATION__',f"{100*audit['target_pairwise_mean_squared_cosine'][-2][-1]:.2f}%")
    doc=doc.replace('__ACTUAL_MEAN__',f"{100*actual['q16_mean']:.2f}%").replace('__ACTUAL_MEDIAN__',f"{100*actual['q16_median']:.2f}%").replace('__ACTUAL_P10__',f"{100*actual['q16_p10']:.2f}%")
    if a.preview_incomplete:
        doc=doc.replace('完整网格共 8,640 条评分记录。','预览：q256 尚未完成，空格不是零分。')
    (ROOT/'dyadic-probe-heatmaps.html').write_text(doc)
    print('Merged',len(rows),'scores; wrote',ROOT/'dyadic-probe-heatmaps.html')


if __name__=='__main__':main()

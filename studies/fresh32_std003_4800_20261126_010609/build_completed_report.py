from pathlib import Path
import json, io, re, html, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'completed-results'
A=json.loads((DATA/'analysis.json').read_text())
OUT=ROOT/'experiment-results.html'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#334155','text.color':'#172b3a','xtick.color':'#526577','ytick.color':'#526577','svg.fonttype':'none'})
COL={'111':'#087f8c','000':'#bd5c37'}
POW=[1,2,3,4,6,8,11,16]
def svg(fig):
    f=io.StringIO(); fig.savefig(f,format='svg',bbox_inches='tight',facecolor='white'); plt.close(fig)
    return '<svg'+f.getvalue().split('<svg',1)[1]
def get(c,mask,l,p): return next(r for r in c['summary'] if (r['mask'],r['layer'],r['power'])==(mask,l,p))
def pct(x,d=2): return f'{100*x:.{d}f}%'
def pp(x,d=2): return f'{100*x:+.{d}f}'
def interval(s): return f'[{100*s["ci"][0]:+.2f}, {100*s["ci"][1]:+.2f}]'
def table(headers,rows): return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+h+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def heat(c):
    fig,axs=plt.subplots(1,2,figsize=(12,3.5),layout='constrained')
    for ax,mask in zip(axs,['111','000']):
        z=np.array([[get(c,mask,l,p)['similarity']['mean']*100 for p in POW] for l in range(5)])
        im=ax.imshow(z,cmap='Blues',vmin=0,vmax=75,aspect='auto')
        for l in range(5):
            for j in range(8): ax.text(j,l,f'{z[l,j]:.1f}',ha='center',va='center',color='white' if z[l,j]>40 else '#172b3a',fontsize=10)
            j=int(z[l].argmax()); ax.scatter(j,l,s=650,facecolors='none',edgecolors='#d98b13',linewidth=2,marker='s')
        ax.set(xticks=range(8),xticklabels=POW,yticks=range(5),yticklabels=['Input 0','Layer 1','Layer 2','Layer 3','Layer 4'],xlabel='Power target k',title=f'{mask} | '+('Normalization on' if mask=='111' else 'Normalization off'))
    fig.colorbar(im,ax=axs,label='Test squared-cosine similarity (%)',shrink=.9)
    return svg(fig)
def difference(c):
    fig,ax=plt.subplots(figsize=(10.7,3.4),layout='constrained')
    z=np.array([[(get(c,'111',l,p)['similarity']['mean']-get(c,'000',l,p)['similarity']['mean'])*100 for p in POW] for l in range(5)])
    im=ax.imshow(z,cmap='BrBG',vmin=-17,vmax=17,aspect='auto')
    for l in range(5):
        for j in range(8): ax.text(j,l,f'{z[l,j]:+.2f}',ha='center',va='center',color='white' if abs(z[l,j])>10 else '#172b3a',fontsize=10)
    ax.set(xticks=range(8),xticklabels=POW,yticks=range(5),yticklabels=['Input 0','Layer 1','Layer 2','Layer 3','Layer 4'],xlabel='Power target k')
    fig.colorbar(im,ax=ax,label='111 minus 000 (percentage points)')
    return svg(fig)
def qline(c):
    fig,ax=plt.subplots(figsize=(10.7,3.7),layout='constrained')
    for mask,mark in [('111','o'),('000','s')]:
        ss=[get(c,mask,l,16)['similarity'] for l in range(5)]
        y=np.array([s['mean'] for s in ss])*100
        ci=np.array([s['ci'] for s in ss])*100
        ax.plot(range(5),y,color=COL[mask],marker=mark,label=mask,lw=2)
        ax.fill_between(range(5),ci[:,0],ci[:,1],color=COL[mask],alpha=.12)
    ax.axhline(100/32,color='#94a3b8',ls=':',label='Random direction: 3.125%')
    ax.set(xticks=range(5),xticklabels=['Input 0','Layer 1','Layer 2','Layer 3','Layer 4'],ylabel='q16 probe similarity (%)',ylim=(0,76))
    ax.grid(axis='y',alpha=.15); ax.legend(frameon=False,ncol=3,loc='upper left')
    return svg(fig)
def bypass(c):
    fig,axs=plt.subplots(1,2,figsize=(12,3.6),sharey=True,layout='constrained')
    for ax,mode,title in zip(axs,['bypass','matched'],['Plain bypass','Length-matched bypass']):
        for mask,mark in [('111','o'),('000','s')]:
            rs=[next(r for r in c['bypass'] if (r['mask'],r['mode'],r['layer'])==(mask,mode,l)) for l in range(1,5)]
            y=np.array([r['mean'] for r in rs])*100; ci=np.array([r['ci'] for r in rs])*100
            ax.errorbar(range(1,5),y,yerr=np.maximum(0,np.array([y-ci[:,0],ci[:,1]-y])),marker=mark,color=COL[mask],label=mask,capsize=3,lw=2)
        ax.set(title=title,xticks=range(1,5),xlabel='Bypassed residual block',ylim=(0,85)); ax.grid(axis='y',alpha=.15); ax.legend(frameon=False)
    axs[0].set_ylabel('Increase in final test error (pp)')
    return svg(fig)
def gapsvg():
    fig,ax=plt.subplots(figsize=(12,3.4),layout='constrained')
    xs=[r['seed'] for r in A['gaps']]; ys=[r['gap'] for r in A['gaps']]
    for flagged in [False,True]:
        idx=[i for i,y in enumerate(ys) if (y>.1)==flagged]
        ax.scatter(idx,[ys[i] for i in idx],color='#ba4f39' if flagged else '#087f8c',marker='X' if flagged else 'o',label='Gap > 0.1' if flagged else 'Gap <= 0.1',zorder=3)
    ax.axhline(.1,color='#ba4f39',ls='--',alpha=.6)
    ax.set(xticks=range(32),xticklabels=xs,ylabel='Absolute validation-error gap',ylim=(0,.15))
    ax.tick_params(axis='x',labelrotation=65,labelsize=9); ax.grid(axis='y',alpha=.15); ax.legend(frameon=False,ncol=2)
    return svg(fig)

sections=[]
for name,c in A['cohorts'].items():
    bestrows=[]
    for l in range(5):
        bs=[next(b for b in c['best'] if b['layer']==l and b['mask']==m) for m in ['111','000']]
        bestrows.append(['输入层 0' if l==0 else f'第 {l} 层',*[f'<strong>q{b["power"]}</strong> · {pct(b["similarity"]["mean"])} ± {100*b["similarity"]["sd"]:.2f}' for b in bs],f'{c["n"]}/{c["n"]} 个种子均为同一最高目标'])
    qrows=[]
    for l in range(5):
        s=next(s for s in c['probe_contrasts'] if (s['layer'],s['power'])==(l,16))
        qrows.append([f'层 {l}',pct(get(c,'111',l,16)['similarity']['mean']),pct(get(c,'000',l,16)['similarity']['mean']),pp(s['mean'])+' pp',interval(s)+' pp'])
    prirows=[]
    for r in c['primary']:
        s=r['contrast']; rel=r['relative_sensitivity']
        prirows.append(['普通 bypass' if r['mode']=='bypass' else '长度匹配 bypass',pp(s['mean'])+' pp',interval(s)+' pp',f'{s["n"]-s["positive"]}/{s["n"]}',pp(rel['mean'])+' pp',interval(rel)+' pp'])
    matrix=[]
    for l in range(5):
        for m in ['111','000']:
            matrix.append([f'层 {l}',m,*[pct(get(c,m,l,p)['similarity']['mean']) for p in POW]])
    tests='；'.join(f'{r["mask"]}：{r["mean"]:.5f} ± {r["sd"]:.5f}' for r in c['test_error'])
    sections.append(f'''<div class="cohort" id="{name}" {'hidden' if name!='all' else ''}>
<section><div class="section-number">01</div><h2>每一层，哪一个 linear probe 最高？</h2><p>在 8 个已测目标中，按测试集平均平方余弦相似度排名。先在每个模型内平均 3 次数据划分，再跨训练种子平均。± 为训练种子之间的标准差，单位为百分点。</p>{table(['层','111：最高目标与相似度','000：最高目标与相似度','逐种子一致性'],bestrows)}
<p class="takeaway">层 3 是分歧最大的地方：111 的最高目标仍为 q1，000 的最高目标已为 q16。层 4 两者最高都是 q16。这个模式在所有种子中一致。</p>
<figure>{heat(c)}<figcaption>金色方框标出每一层最高分。共享色阶；数值为原始平方余弦相似度百分比。</figcaption></figure>
<p>但“q1 分数最高”不等于只做了一步迭代。输入层尚未经网络计算，q1 probe 已达约 68.6%，说明起始方向与低幂次目标的相关性很强。不同 q 的难度不同，且相邻目标方向相关；最高分目标只是表征的描述性指标，不能解释为网络精确执行了 k 次迭代。</p></section>
<section><div class="section-number">02</div><h2>对同一个高幂次目标，谁更早可读？</h2><p>固定看已测的最高幂次 q16，避免把不同目标的难度混在一起。正差值代表 111 更好。</p><figure>{qline(c)}<figcaption>阴影为跨训练种子的 95% bootstrap 均值区间。q16 是幂迭代方向，区别于母模型训练用的真实主特征向量。</figcaption></figure>{table(['层','111 相似度','000 相似度','111 − 000','配对差值 95% 区间'],qrows)}
<p class="takeaway">层 1、2 的优势接近零，区间均跨零；层 3、4 是 000 的 q16 读取更好。当前 probe 结果不支持 normalization 把高幂次方向更早变得容易线性读取。</p>
<h3>全部幂次的配对差值</h3><figure>{difference(c)}<figcaption>绿色为 111 更好，棕色为 000 更好，单位为百分点。层 3 的 111 优势集中在 q1、q2；q3–q16 则是 000 更好。</figcaption></figure>
<details><summary>查看所有层与目标的相似度原表</summary>{table(['层','条件',*[f'q{p}' for p in POW]],matrix)}</details></section>
<section><div class="section-number">03</div><h2>Bypass：后层是否已经不重要？</h2><p>对已训练模型逐层跳过一次<strong>残差更新</strong>，保留该处的归一化，再评估最终任务误差。长度匹配对照另外匹配正常前向传播中该层的 token 长度。ΔE = 干预后误差 − 完整模型误差，越大说明该干预越有破坏性。</p><figure>{bypass(c)}<figcaption>点为种子均值，误差棒为 95% bootstrap 区间。111 的两种 bypass 基本重合，符合归一化对长度调整的抵消作用。</figcaption></figure>
<p>两组跳过第 4 层都会造成巨大误差上升，曲线也都总体随层加深而升高。因此，不能说 111 在早层已经完成任务、最后一层基本无用。</p>
<div class="formula">C = [(ΔE₄ − ΔE₁)₁₁₁ − (ΔE₄ − ΔE₁)₀₀₀]</div><p>C &lt; 0 表示 111 的“晚层相对早层的干预代价差”较小，与<strong>相对更依赖早层</strong>的假设一致。</p>{table(['干预','C 均值','95% 配对区间','C < 0 种子数','按完整模型相似度缩放后的 C','缩放后区间'],prirows)}
<p class="takeaway">Bypass 提供了“层间依赖相对变化”的支持，尤其是长度匹配对照。但它衡量的是干预敏感性，不能单独证明早层已经算出更高级的中间结果。</p>
<p>完整模型测试误差为 {tests}。两组最终性能仍不同。把每个模型的 ΔE 除以其完整模型相似度（1 − E）后，普通 bypass 的 C 区间跨零；长度匹配的 C 仍为负。这是本报告新增的探索性尺度敏感性分析，不能替代原始对比，也不能彻底解决性能差异或干预饱和问题。</p></section>
</div>''')

gap= gapsvg()
flag=', '.join(map(str,A['flagged']))
summary_text='当前结果不支持“normalization 让高阶计算在更早层完成”的强结论。Probe 没有出现所需的早层优势；bypass 支持的是更有限的层间依赖变化。'
raw={p.name:p.read_text() for p in DATA.glob('*.csv')}
assets=''.join(sections)
document='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Normalization 与计算提前：32 对模型实验结果</title><style>
:root{--ink:#172b3a;--muted:#5c6d7c;--teal:#087f8c;--rust:#bd5c37;--line:#dce4e8;--paper:#fff;--bg:#f2f5f5}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.8 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}main{max-width:1160px;margin:32px auto;background:white;padding:52px 62px}header{border-top:5px solid var(--teal);padding-top:26px}.eyebrow{font-size:12px;letter-spacing:.15em;color:var(--teal);font-weight:700}h1{font-size:38px;line-height:1.35;max-width:800px;margin:13px 0 20px;letter-spacing:-1px}h2{font-size:26px;line-height:1.4;margin:6px 0 18px}h3{font-size:19px;margin-top:30px}p{margin:13px 0}strong{font-weight:650}.lead{font-size:20px;line-height:1.75;max-width:900px}.muted,figcaption{color:var(--muted);font-size:13px}.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:24px;margin:30px 0;padding:24px 0;border-block:1px solid var(--line)}.kpi b{display:block;font-size:30px;line-height:1.3}.kpi span{font-size:13px;color:var(--muted)}nav{display:flex;gap:10px;flex-wrap:wrap;margin:30px 0 12px}button,.button{font:inherit;font-size:14px;border:1px solid #c8d5dc;border-radius:5px;padding:9px 15px;background:white;color:var(--ink);cursor:pointer;text-decoration:none}button.active{background:var(--teal);color:white;border-color:var(--teal)}button:focus-visible,a:focus-visible{outline:3px solid #e4ac3f;outline-offset:3px}section{padding:35px 0;border-top:1px solid var(--line)}.section-number{font-size:13px;color:var(--teal);font-weight:700;letter-spacing:.12em}figure{margin:24px 0 20px}svg{display:block;width:100%;height:auto}figcaption{margin-top:9px}table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}th{text-align:left;font-size:12px;color:var(--muted);font-weight:600;background:#f5f8f8}td,th{padding:12px 13px;border-bottom:1px solid var(--line);vertical-align:top}td:not(:first-child){white-space:nowrap}.scroll{overflow:auto;margin:22px 0}.takeaway{border-left:3px solid var(--teal);padding:2px 0 2px 17px;font-weight:600}.formula{font:18px/1.7 ui-monospace,monospace;padding:16px 0}.sources{display:flex;gap:12px;flex-wrap:wrap}.sources a,a{color:var(--teal)}details{margin:24px 0}summary{cursor:pointer;color:var(--teal);font-weight:600}li{margin:10px 0}.verdict{font-size:20px;font-weight:650}footer{border-top:1px solid var(--line);padding-top:22px;font-size:13px;color:var(--muted)}[hidden]{display:none!important}.controls-note{font-size:13px;color:var(--muted)}@media(max-width:750px){main{margin:0;padding:26px 20px}h1{font-size:29px}h2{font-size:23px}.kpis{grid-template-columns:repeat(2,1fr)}.lead{font-size:18px}td,th{padding:10px 8px}}@media print{body{background:white}main{margin:0;padding:0;max-width:none}nav,.downloads{display:none}section{break-inside:auto}figure,table{break-inside:avoid}h2,h3{break-after:avoid}button{display:none}a{color:inherit;text-decoration:none}.cohort[hidden]{display:none!important}}
</style></head><body><main><header><div class="eyebrow">AIOWLS RESEARCH / POWER ITERATION</div><h1>Normalization 是否让计算<br>在更早层完成？</h1><p class="lead">__SUMMARY__</p><p class="muted">本轮固定预算终点分析 · 32 对配对模型 · 111 / 000 · 初始化标准差 0.03 · 每模型 4,800 步</p></header>
<div class="kpis"><div class="kpi"><b>32 / 32</b><span>模型对与分析均已完成</span></div><div class="kpi"><b>7,680</b><span>Linear probe 测试评分</span></div><div class="kpi"><b>512</b><span>普通与长度匹配 bypass 结果</span></div><div class="kpi"><b>7 / 32</b><span>最终验证误差差值 > 0.1</span></div></div>
<nav aria-label="结果子集"><button class="active" data-cohort="all" aria-pressed="true">全部 32 对</button><button data-cohort="gap_le_01" aria-pressed="false">差值 ≤ 0.1 的 25 对</button><button onclick="window.print()">打印 / 存为 PDF</button></nav><p class="controls-note">切换子集会更新下方最高 probe、热图、q16 曲线、bypass 与统计表。25 对仅通过宽松差值阈值，不能称为最终性能严格匹配。</p>
__COHORTS__
<section><div class="section-number">04</div><h2>最终差值分布</h2><p>差值 = |111 最终验证误差 − 000 最终验证误差|。超过 0.1 的种子为 <strong>__FLAG__</strong>。所有种子都纳入主分析，不因差值较大删除。</p><figure>__GAP__<figcaption>每个点对应一个训练种子；红色叉号标记差值 > 0.1。</figcaption></figure></section>
<section><div class="section-number">05</div><h2>对“normalization 可以提前学习”的判断</h2><p class="verdict">如果“提前”指深度方向上的计算提前：目前不能确认，probe 的主要模式甚至指向相反方向。</p><ol><li><strong>早层读取优势没有出现。</strong> q16 在层 1、2 的 111 优势只有 +0.14、+0.06 个百分点，配对区间均跨零；层 3 为 −5.97 个百分点，层 4 为 −5.01 个百分点。</li><li><strong>最高 probe 反映不同的表征分配。</strong> 层 3 的 111 更容易读出 q1，000 更容易读出 q16；这支持 111 更保留低幂次方向，而不是更早呈现高幂次方向。最高目标模式在 32 个种子中完全一致。</li><li><strong>Bypass 支持有限的相对前移信号。</strong> 111 的“晚层相对早层的代价差”较小，长度匹配对照尤为明显。但两组都高度依赖最后一层，且最终性能不同，不能把该负对比直接等同于计算提前。</li><li><strong>这轮没有测训练时间上的“更早学会”。</strong> 当前 probe 只比较 4,800 步终点。若要判断 111 是否在较少训练步数就学会同样内容，需要对训练过程的多个检查点运行相同 probe。</li></ol><h3>适合汇报的一句话</h3><p class="takeaway">在当前固定预算的 32 对模型中，normalization 改变了中间表征与层间干预敏感性，但没有使高幂次方向在更早层更易线性读取。Bypass 的相对前移信号尚不足以建立“计算提前完成”的结论。</p><h3>下一步最有区分力的实验</h3><p>保留当前结果，另外比较最终任务性能更接近的模型，预先确定“提前”的判据：同一高幂次目标在更浅层达到同一读取水平，并与降低的晚层增量依赖相互印证。若研究训练速度，则对多个训练检查点重复 probe。当前结果不应通过只挑某个有利目标或删掉差值较大的种子来改写结论。</p></section>
<section><div class="section-number">06</div><h2>实验口径与解释边界</h2><ul><li><strong>母模型：</strong>4 个残差块，d = 32，n = 64；111 在前 3 个块后做 token 列向量的 L2 单位化，000 关闭这 3 个内部归一化。这里不是带可学习平移/缩放的标准 LayerNorm。末层方向读出两组相同。</li><li><strong>配对与预算：</strong>种子 1001–1032；同一对使用相同数据、初始化、批次及学习率计划；初始化 SD = 0.03，学习率基值 0.0005，float64，每个模型固定 4,800 步。已完成模型直接读取检查点；本轮只补完剩余训练与分析。</li><li><strong>任务与 probe：</strong>母模型拟合 XXᵀ 的主特征向量；probe 目标为 q<sub>k</sub> = normalize((XXᵀ)<sup>k</sup>a)，k ∈ {1, 2, 3, 4, 6, 8, 11, 16}。q16 仍不等于真实主特征向量。</li><li><strong>数据：</strong>每个种子训练 65,536、验证 2,048、母模型测试 4,096；probe 独立数据池 3,000。每次划分 fit / validation / test = 1,680 / 420 / 900，划分种子 0、1、2。</li><li><strong>Linear probe：</strong>每层 query 的完整 64 维特征映射到 32 维方向，无 bias；平方余弦方向损失；300 次更新，固定与随机两种初始化，以验证集选权重。逐样本正比例缩放及按特征 RMS 缩放不改变线性方向读出的函数类。</li><li><strong>指标：</strong>相似度 s = mean(cos²)，越高越好；误差 E = 1 − s，越低越好。随机独立方向基线为 1/32 = 3.125%；校正相似度为 (s − 1/32)/(1 − 1/32)，此仿射变换不会改变逐层目标排名。</li><li><strong>负对照：</strong>fit 与 validation 标签分别打乱，test 保留真实标签。q16 的打乱标签分数各层约 3.11%–3.15%，接近随机基线；真实读取超过负对照，说明方向信息可读，但不确定网络采用了哪种内部算法。</li><li><strong>统计：</strong>先在每个模型内平均 3 个 split，再以训练种子为独立配对单位，不能把 7,680 条评分当作独立模型。均值区间为 20,000 次 percentile bootstrap；差值按同一种子配对。区间是探索性、未作多重比较校正。逐层最高目标是测试集上的描述性排名。</li><li><strong>干预边界：</strong>bypass 会改变中间状态及后续输入分布，长度匹配仅控制 token 长度；最终误差上界及两组基础性能差异仍影响 ΔE。负的 C 不能单独定位某个算法步骤。</li></ul></section>
<section class="downloads"><div class="section-number">07</div><h2>数据与复核</h2><p>报告离线可用。下方 CSV 直接来自本轮结果，下载不会重新训练或改变实验。</p><div class="sources"><a class="button" href="https://drive.google.com/drive/folders/11QFFvJWo3oaUXCOJyN2E_2DhR6Fzw-DL" target="_blank" rel="noopener">完整 Drive 结果</a><button data-download="power-probes.csv">完整 probe CSV</button><button data-download="bypass.csv">Bypass CSV</button><button data-download="probe-summary.csv">Probe 汇总 CSV</button><button data-download="seed-gaps.csv">种子差值 CSV</button></div><p class="muted">数据来源：fresh32_std003_4800_20261126_010609 运行的 power-probes.csv、bypass.csv、probe-summary.csv、matched-probe-summary.csv 与 seed-gaps.csv。全部评分、两种 bypass 及两个子集均已核对。</p></section>
<footer>AIOwls Research · 固定预算模型对照报告。文中结论限定于当前结构、数据分布、初始化与训练配方。</footer></main>
<script id="data" type="application/json">__RAWDATA__</script><script>
document.querySelectorAll('[data-cohort]').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('.cohort').forEach(e=>e.hidden=e.id!==b.dataset.cohort);document.querySelectorAll('[data-cohort]').forEach(x=>{x.classList.toggle('active',x===b);x.setAttribute('aria-pressed',String(x===b))})}));
const raw=JSON.parse(document.getElementById('data').textContent);document.querySelectorAll('[data-download]').forEach(b=>b.addEventListener('click',()=>{const u=URL.createObjectURL(new Blob(['\ufeff'+raw[b.dataset.download]],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=u;a.download=b.dataset.download;a.click();setTimeout(()=>URL.revokeObjectURL(u),5000)}));
</script></body></html>'''
document=document.replace('__SUMMARY__',summary_text).replace('__COHORTS__',assets).replace('__FLAG__',flag).replace('__GAP__',gap).replace('__RAWDATA__',json.dumps(raw,ensure_ascii=False).replace('</','<\\/'))
OUT.write_text(document)
print(OUT)
print('HTML bytes:',OUT.stat().st_size)

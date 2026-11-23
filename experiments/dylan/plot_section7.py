from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from docx import Document
from docx.table import Table
from docx.shared import Inches
root=Path(__file__).resolve().parents[2]
base=root/'experiments/dylan/results'
data=json.loads((base/'math-similarity/result.json').read_text())
folder=base/'section7-figures';folder.mkdir(parents=True,exist_ok=True)
def matrix(metric,projection):
 if metric=='RMI': return np.array(data['diagnostics'][projection]['rmi'])
 return np.array(data['similarities'][projection][metric])
for title,metrics,slug in [('RMI and cosine',['RMI','cosine'],'figure3'),('Alternative gradient metrics',['cosine','KL','L1','L2'],'figure8')]:
 fig,axes=plt.subplots(2,len(metrics),figsize=(4*len(metrics),7),squeeze=False)
 for row,projection in enumerate(['q','v']):
  for col,metric in enumerate(metrics):
   ax=axes[row,col];a=matrix(metric,projection)
   im=ax.imshow(a,cmap='viridis',aspect='equal')
   ax.set_title(projection.upper()+' '+metric);ax.set_xlabel('Layer');ax.set_ylabel('Layer');fig.colorbar(im,ax=ax,shrink=.75)
 fig.tight_layout();fig.savefig(folder/(slug+'.png'),dpi=200,bbox_inches='tight');plt.close(fig)
progress=json.loads((base/'training-entropy/progress.json').read_text())
assert progress['step']==1024,progress
rows=list(csv.DictReader((base/'training-entropy/entropy.csv').open()))
fig,axes=plt.subplots(1,2,figsize=(12,4))
for ax,layer in zip(axes,[0,1]):
 for projection in ['q','k','v','o']:
  values=[r for r in rows if int(r['layer'])==layer and r['projection']==projection]
  ax.plot([int(r['optimizer_step']) for r in values],[float(r['proxy_entropy']) for r in values],label=projection.upper(),linewidth=1)
 ax.set_title(f'Layer {layer}');ax.set_xlabel('Optimizer step');ax.set_ylabel('Gradient proxy entropy');ax.legend();ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(folder/'figure4.png',dpi=220,bbox_inches='tight');plt.close(fig)
p=root/'E²LoRA 论文复现报告.docx';d=Document(p)
def following_tables(prefix):
 h=next(p for p in d.paragraphs if p.text.startswith(prefix));node=h._p.getnext();tables=[]
 while node is not None:
  if node.tag.endswith('}p') and node.xpath('./w:pPr/w:pStyle'):break
  if node.tag.endswith('}tbl'):tables.append(Table(node,d._body))
  node=node.getnext()
 return tables
for t in following_tables('图 3 与图 8'):
 if len(t.rows)==1:
  c=t.rows[0].cells[0].merge(t.rows[0].cells[-1]);c.text=''
  for i,slug in enumerate(['figure3','figure8']):
   para=c.paragraphs[0] if i==0 else c.add_paragraph()
   para.add_run().add_picture(str(folder/(slug+'.png')),width=Inches(6))
 else:t._tbl.getparent().remove(t._tbl)
for t in following_tables('图 4  训练期间'):
 if len(t.rows)==1:
  c=t.cell(0,0);c.text='';c.paragraphs[0].add_run().add_picture(str(folder/'figure4.png'),width=Inches(6))
 else:t._tbl.getparent().remove(t._tbl)
for i,p2 in enumerate(d.paragraphs):
 if p2.text.startswith('用 Llama 3.1 8B Base 与 MetaMathQA'):
  p2.text='图 3 与图 8  固定种子 42，使用相同 64 条 MetaMathQA 训练样本、最大长度 512，计算 32 层 Q、V 的平均梯度。图 3 对照 RMI 与余弦，图 8 依次展示余弦、KL、L1、L2。各矩阵使用独立色阶，保留原始数值。L1、L2 为原始距离，KL 为共同取值范围内 128 格直方图概率的有向 KL，平滑量为 1e-12；距离越小表示越相近。原始明细见 experiments/dylan/results/math-similarity/。'
 if p2.text.startswith('用 Llama 3.1 8B Base 在 MetaMathQA 全量微调'):
  p2.text='图 4  使用 Llama 3.1 8B Base 和 MetaMathQA，AdaLomo 学习率 1e-4，全模型微调 1,024 步。种子 42，每步 1 条样本、最大长度 512，query 与 response 拼接并共同计算损失，使用 BF16。每 5 步记录更新前 Q、K、V、O 权重梯度的代理熵，从第 5 步记录至第 1,020 步。此图取第 0、1 层，完整逐层明细见 experiments/dylan/results/training-entropy/。'
d.save(p)

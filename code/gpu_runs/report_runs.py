"""Build the four agreed plots from complete matching inference analyses."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from compare_runs import compare


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as f:return list(csv.DictReader(f))


def short_name(repo):
    return repo.split('/')[-1].replace('-Video-Instruct','').replace('-Instruct','')


def heatmap(details, names, field, columns, path, title):
    data=np.full((len(details),len(columns)),np.nan)
    counts=np.zeros_like(data,dtype=int)
    for i,rows in enumerate(details):
        for j,col in enumerate(columns):
            subset=[r for r in rows if r[field]==col]
            counts[i,j]=len(subset)
            if subset:data[i,j]=sum(r['correct'].lower()=='true' for r in subset)/len(subset)
    fig,ax=plt.subplots(figsize=(10,max(3.3,len(names)*0.65+1.8)),layout='constrained')
    im=ax.imshow(np.ma.masked_invalid(data),vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
    ax.set_xticks(range(len(columns)),columns)
    ax.set_yticks(range(len(names)),names)
    for i in range(len(names)):
        for j in range(len(columns)):
            label='N/A\nn=0' if np.isnan(data[i,j]) else f'{data[i,j]:.0%}\nn={counts[i,j]}'
            ax.text(j,i,label,ha='center',va='center',fontsize=9,
                    color='white' if data[i,j]>0.65 else 'black')
    ax.set_title(title+'\nInvalid answers count as incorrect')
    fig.colorbar(im,ax=ax,label='Image accuracy',shrink=0.8)
    fig.savefig(path,dpi=180);plt.close(fig)
    with path.with_suffix('.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['model','category','accuracy','n'])
        writer.writeheader()
        for i,name in enumerate(names):
            for j,col in enumerate(columns):
                writer.writerow(dict(model=name,category=col,accuracy='' if np.isnan(data[i,j]) else data[i,j],n=counts[i,j]))


def confusion(details,names,path):
    ncols=min(3,len(names));nrows=(len(names)+ncols-1)//ncols
    fig,axes=plt.subplots(nrows,ncols,figsize=(5*ncols,3.7*nrows),squeeze=False,layout='constrained')
    for ax,rows,name in zip(axes.flat,details,names):
        counts=np.zeros((2,3),dtype=int)
        for r in rows:
            i=('yes','no').index(r['ground_truth'])
            pred=r['prediction'];j=('yes','no').index(pred) if pred in ('yes','no') else 2
            counts[i,j]+=1
        totals=counts.sum(axis=1,keepdims=True)
        proportions=np.divide(counts,totals,out=np.zeros_like(counts,dtype=float),where=totals!=0)
        ax.imshow(proportions,vmin=0,vmax=1,cmap='Blues',aspect='auto')
        for i in range(2):
            for j in range(3):
                pct=f'{proportions[i,j]:.0%}' if totals[i,0] else 'N/A'
                ax.text(j,i,f'{counts[i,j]}\n{pct}',ha='center',va='center',
                        color='white' if proportions[i,j]>0.6 else 'black')
        ax.set_xticks(range(3),['Yes','No','Invalid'])
        ax.set_yticks(range(2),['Yes','No']);ax.set_xlabel('Predicted');ax.set_ylabel('True')
        ax.set_title(name)
    for ax in list(axes.flat)[len(names):]:ax.set_visible(False)
    fig.suptitle('Confusion matrices: counts and percentages within each true class')
    fig.savefig(path,dpi=180);plt.close(fig)


def runtime_plot(rows,path):
    groups=defaultdict(list)
    for r in rows:groups[r['timing_group']].append(r)
    fig,axes=plt.subplots(len(groups),1,figsize=(9,max(4,3.8*len(groups))),squeeze=False,layout='constrained')
    for ax,subset in zip(axes.flat,groups.values()):
        for r in subset:
            seconds=r['mean_seconds_excluding_first']
            if seconds=='':continue
            accuracy=float(r['image_accuracy'])
            low,high=r.get('image_accuracy_ci_low',''),r.get('image_accuracy_ci_high','')
            ax.scatter(float(seconds),100*accuracy,s=75)
            if low!='' and high!='':
                ax.vlines(float(seconds),100*float(low),100*float(high),color='gray',alpha=0.6)
            ax.annotate(short_name(r['model']),(float(seconds),100*accuracy),xytext=(6,7),textcoords='offset points',fontsize=9)
        r=subset[0]
        machine=r['hardware_gpu'] or r['hardware_cpu'] or 'CPU name unavailable: verify machine manually'
        ax.set_title(f"{machine} | {r['device']} | {r['dtype']} | threads={r['threads']}")
        ax.set_xlabel('Mean seconds/image, excluding first image and model loading')
        ax.set_ylabel('Image accuracy (%)');ax.set_ylim(-5,112);ax.margins(x=0.25);ax.grid(alpha=0.2)
    fig.suptitle('Accuracy versus runtime — hardware/precision/thread groups shown separately')
    fig.savefig(path,dpi=180);plt.close(fig)


def report(analyses,out):
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise ValueError('Report folder must be new or empty.')
    out.mkdir(parents=True,exist_ok=True)
    rows=compare(analyses,out/'comparison.csv')
    details=[read_csv(Path(d)/'image_details.csv') for d in analyses]
    names=[short_name(r['model']) for r in rows]
    heatmap(details,names,'relation',['left','right','above','below','inside','outside'],
            out/'spatial_relation_heatmap.png','Image accuracy by spatial question relation')
    heatmap(details,names,'variant',['base','spatial','brightness','clutter','occlusion'],
            out/'image_condition_heatmap.png','Image accuracy by image condition')
    confusion(details,names,out/'confusion_matrices.png')
    runtime_plot(rows,out/'accuracy_runtime.png')
    (out/'README.txt').write_text(
        'Generated from actual supplied prediction analyses; no inference rerun.\n'
        'Each heatmap cell includes its number of images. N/A means no observations.\n'
        'Confusion matrices include invalid answers and normalize within true class.\n'
        'Runtime excludes loading/download and first image; bars show scene-group bootstrap accuracy intervals when available.\n'
        'Timing facets separate recorded hardware, precision and threads; verify hardware metadata before speed claims.\n'
        'Five-image single-scene checks are not balanced benchmarks.\n',encoding='utf-8')
    print(f'Four plots and comparison table saved: {out}')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analyses',nargs='+',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();report(a.analyses,a.out)

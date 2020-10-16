from batch_center_eval import Scorer
import jsonlines
from tqdm import tqdm
import numpy as np
import argparse

## Read in arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_name",default='roberta-base', type=str)
parser.add_argument("--use_metric",default='TWMD',type=str)
parser.add_argument("--use_batch_center",default='True',type=str)
parser.add_argument("--T",default=None,type=float)
parser.add_argument("--use_correlation",default='pearsonr',type=str)
parser.add_argument("--year",default=17,type=int)
args = parser.parse_args()
model_name = args.model_name
use_metric = args.use_metric
use_batch_center = True if args.use_batch_center=='True' else False
T = args.T
use_correlation = args.use_correlation
year = args.year

## Loading data
scores = {}
s1 = {}
s2 = {}
langs = []
with jsonlines.open('./dataset/WMT/wmt'+str(year)+'.jsonl') as f:
    for line in tqdm(f.iter()):
        lang = line['lang']
        if lang not in scores.keys():
            scores[lang] = []
            s1[lang] = []
            s2[lang] = []
            langs.append(lang)
        if len(line['candidate'])>=2 and len(line['reference'])>=2:
            if year==18:
            	scores[lang].append(line['rating'])
            else:
            	scores[lang].append(line['score'])
            s1[lang].append(line['candidate'])
            s2[lang].append(line['reference'])

## Obtaining correlation
scorer = Scorer(model_name = model_name)
corrs = []
for lang in langs:
    corrs.append(scorer.score(s1[lang],s2[lang],scores[lang],metric=use_metric, T=T, batch_center=use_batch_center,
    	return_correlation=use_correlation)[-1])

## Calculating mean correlation with human score
mean_corr = np.mean(np.array(corrs),axis=0)
print("Model: ",model_name,"\nDataset: WMT",str(year),"\nMetric: ",use_metric,"\nBatch centering: ",use_batch_center)
if T is not None:
	print("\nspecifying T = ",T)
if len(mean_corr) == 1:
	R = mean_corr[0]
	print("Recall :",R)
else: 
	R, P, F1 = mean_corr
	print("Recall :",R,"\nPrecision: ",P,"\nF1: ",F1)


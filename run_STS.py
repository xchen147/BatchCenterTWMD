from batch_center_eval import Scorer
import jsonlines
from tqdm import tqdm
import io
import numpy as np
import argparse

## Functions and parameters for loading data

def loadFile(fpath,year):
    subsets={}
    subsets['12']= ['MSRpar', 'MSRvid', 'SMTeuroparl',
                            'surprise.OnWN', 'surprise.SMTnews']
    subsets['13'] = ['FNWN', 'headlines', 'OnWN']
    subsets['14'] = ['deft-forum', 'deft-news', 'headlines',
                            'images', 'OnWN', 'tweet-news']
    subsets['15'] = ['answers-forums', 'answers-students',
                            'belief', 'headlines', 'images']
    subsets['16'] = ['answer-answer', 'headlines', 'plagiarism',
                            'postediting', 'question-question']

    s1={}
    s2={}
    scores={}

    for entry in subsets[year]:
        sent1, sent2 = zip(*[l.split("\t") for l in
                           io.open(fpath + '/STS.input.%s.txt' % entry,
                                   encoding='utf8').read().splitlines()])
        raw_scores = np.array([x for x in
                               io.open(fpath + '/STS.gs.%s.txt' % entry,
                                       encoding='utf8').read().splitlines()])
        not_empty_idx = raw_scores != ''

        gs_scores = [float(x) for x in raw_scores[not_empty_idx]]
        sent1 = np.array(sent1)[not_empty_idx]
        sent2 = np.array(sent2)[not_empty_idx]
        # sort data by length to minimize padding in batcher
        sorted_data = sorted(zip(sent1, sent2, gs_scores),key=lambda z: (len(z[0]), len(z[1]), z[2]))
        sent1, sent2, gs_scores = map(list, zip(*sorted_data))

        s1[entry]=sent1
        s2[entry]=sent2
        scores[key]=gs_scores
    return s1, s2, scores

## obtaining correlation
if __name__=="__main__":
    ## Read in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",default='bert-base-uncased', type=str)
    parser.add_argument("--use_metric",default='SBERT',type=str)
    parser.add_argument("--use_batch_center",default='True',type=str)
    parser.add_argument("--T",default=None,type=float)
    parser.add_argument("--use_correlation",default='pearsonr',type=str)
    args = parser.parse_args()
    model_name = args.model_name
    use_metric = args.use_metric
    use_batch_center = True if args.use_batch_center=='True' else False
    T = args.T
    use_correlation = args.use_correlation

    scorer = Scorer(model_name = model_name,use_CLS=False)
    corrs = []
    for year_of_sts in ['12','13','14','15','16']:
        corr = []
        s1, s2, scores = loadFile('./dataset/STS/STS'+year_of_sts+'-en-test/',year_of_sts = year_of_sts])
        for name in dset[year_of_sts]:
            corr.append(scorer.score(s1[name],s2[name],scores[name],metric=use_metric, batch_center=use_batch_center,
                return_correlation=use_correlation)[-1])
        len_weight = [len(s1[name]) for name in dset[year_of_sts]]
        corrs.append(np.average(corr, weights = len_weight,axis=0))

    R = np.mean(np.array(corrs),axis=0)[0]

    print("Model: ",model_name,"\nDataset: STS12-16","\nMetric: ",use_metric,"\nBatch centering: ",use_batch_center)
    print("Recall :",R)
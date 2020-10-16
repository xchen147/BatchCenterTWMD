import torch
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr, kendalltau
from transformers import AutoConfig,AutoModelWithLMHead,AutoTokenizer, AutoModel
import numpy as np
from metrics import *

class Scorer:
    def __init__(
        self,
        model_name='bert-base-uncased',
        layer_use=None,
        batch_size=32,
        device=None,
        use_CLS=True
    ):
        """
        Args:
            - :param: `model_name` (str): the name of the contextualized embedding models available through
                      https://huggingface.co/models
            - :param: `layer_use` (int): the layer where the output will be used.
            - :param: `batch_size` (int): batch size in evaluation. 
            - :param: `device` (str): device on which the model will be run. If None, cuda:0 will be used 
                      if cuda is available.
            - :param: `use_CLS` (bool): whether to include the CLS token.
        """
        
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model_name = model_name
        self.batch_size = batch_size
        
        if layer_use is None:
            if self.model_name in ['bert-base-uncased','bert-large-uncased','roberta-base','roberta-large']:
                if 'base' in self.model_name:
                    self.layer_use = 10
                else:
                    self.layer_use = 19
            else:
                # use the last layer
                self.layer_use = -1
        
        self.model, self.tokenizer = self.load_model()
        
        self.metric_dict = {
            'SBERT':SBERT,
            'CKA':cka,
            'BERTSCORE':bertscore,
            'TWMD':TWMD,
            'TRWMD':TRWMD
        }
        
        self.correlation_dict = {
            'pearsonr':pearsonr,
            'spearmanr':spearmanr,
            'kendalltau':kendalltau
        }

        self.use_CLS = use_CLS
        
    
    def score(self, cand, ref, gs=None, metric='TWMD', T=None, batch_center=True, return_correlation=None):
        """
        Args:
            - :param: `cand` (list of str): candidate sentences
            - :param: `refs` (list of str): reference sentences
            - :param: `gs` (list of float): ground truth similarity scores (e.g., human evaluation)
                      Required if return_correlation is not None
            - :param: `metric` (str): evaluation metric. Can choose from 
                      ['SBERT','CKA','bertsocre','TWMD','TRWMD']. Upper/lower case do not matter.
                      Default is TWMD
            - :param: `T` (float): temperature for TWMD and TRWMD. Default value here is None. The default
                      value used will be 0.02 for TWMD and TRWMD, 0.10 for TWMD-batch and 0.15 for TRWMD-batch
            - :param: `batch_center` (bool): whether or not to apply batch centering to the word vectors.
                      Default is True. 
            - :param: `return_correlation` (str): one of the following correlations 
                      ['pearsonr','spearmanr','kendalltau'] will be computed.  

        Return:
            - :param: `(R, P, F)` each is a list of N numbers (N = len(cand) = len(ref))
                      If returning correlations, the output will be (R, P, F, corr), where
                      `corr` is one of the three correlations [pearsonr, spearmanr, kendalltau]
                      from scipy.stats
        """
        if return_correlation is not None:
            assert gs is not None, "Providing ground truth in order to compute correlation"

        eval_result = []
        # eval_p = []
        # eval_f1 = []
        for i in tqdm(range(int(np.ceil(len(ref)/self.batch_size)))):
            input_ref = ref[self.batch_size*i:self.batch_size*(i+1)]
            input_cand = cand[self.batch_size*i:self.batch_size*(i+1)]
            embed_ref = self.get_embedding(input_ref)
            embed_cand = self.get_embedding(input_cand)
            if batch_center:
                batch_mean_ref = torch.mean(torch.cat(embed_ref),dim=0)
                batch_mean_cand = torch.mean(torch.cat(embed_cand),dim=0)
            
            for kk in range(len(embed_ref)):
                entry_ref, entry_cand = embed_ref[kk], embed_cand[kk]
                if batch_center:
                    entry_ref = entry_ref - batch_mean_ref.unsqueeze(0)
                    entry_cand = entry_cand - batch_mean_cand.unsqueeze(0)
                entry_ref = entry_ref/torch.norm(entry_ref,dim = -1).unsqueeze(-1)
                entry_cand = entry_cand/torch.norm(entry_cand,dim = -1).unsqueeze(-1)
                
                metric_used = self.metric_dict[metric.upper()]
                
                if metric.upper()[0]=='T':
                    if T is None:   
                        if metric.upper() == 'TWMD':
                            T = 0.1 if batch_center else 0.02
                        else:
                            T = 0.15 if batch_center else 0.02
                    sym = metric_used(entry_cand, entry_ref, T, device=self.device)
                else:
                    sym = metric_used(entry_cand, entry_ref)

                # if len(sym)==3:
                #     r, p,f1 = sym

                # eval_r.append(r)
                # eval_p.append(p)
                # eval_f1.append(f1)

                eval_result.append(sym)

        
        eval_result = torch.stack(eval_result).detach().cpu().numpy()

        # eval_r = torch.stack(eval_r).detach().cpu().numpy()
        # eval_p = torch.stack(eval_p).detach().cpu().numpy()
        # eval_f1 = torch.stack(eval_f1).detach().cpu().numpy()
        
        if return_correlation:
            corr_used = self.correlation_dict[return_correlation]
            corr = []
            for result in eval_result.T:
                corr.append(corr_used(result, gs)[0])
            return (eval_result.T, corr)
        
        else:
            return eval_result.T


    def load_model(self):
        config = AutoConfig.from_pretrained(self.model_name, cache_dir='./cache')
        config.output_hidden_states = True
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir='./cache')
        model = AutoModelWithLMHead.from_pretrained(self.model_name,config = config, cache_dir='./cache')
        model.to(self.device)
        return model, tokenizer
    
    def get_embedding(self, sentences):
        sentences_index = [self.tokenizer.encode(s, add_special_tokens=True) for s in sentences]
        features_input_ids = []
        features_mask = []
        max_len = max([len(sent_ids) for sent_ids in sentences_index])
        for sent_ids in sentences_index:
            sent_mask = [1] * len(sent_ids)
            # Padding
            padding_length = max_len - len(sent_ids)
            sent_ids += ([1] * padding_length)
            sent_mask += ([0] * padding_length)
            # Length Check 
            assert len(sent_ids) == max_len
            assert len(sent_mask) == max_len

            features_input_ids.append(sent_ids)
            features_mask.append(sent_mask)

        batch_input_ids = torch.tensor(features_input_ids, dtype=torch.long)
        batch_input_mask = torch.tensor(features_mask, dtype=torch.long)

        batch = [batch_input_ids.to(self.device), batch_input_mask.to(self.device)]

        inputs = {"input_ids": batch[0], "attention_mask": batch[1]}

        with torch.no_grad():
            features = self.model(**inputs)[-1]

        all_layer_embedding = torch.stack(features).permute(1, 0, 2, 3)
        embedding = self.select_layer(features_mask, all_layer_embedding)
        return embedding


    def select_layer(self, masks, all_layer_embedding):
        """
            Selecting the layer embedding of `layer_use`
        """
        unmask_num = np.sum(masks, axis=1)
        embedding = []
        for i in range(len(unmask_num)):
            sent_len = unmask_num[i]
            hidden_state_sen = all_layer_embedding[i]
            if self.use_CLS:
                embedding.append(hidden_state_sen[self.layer_use,:sent_len-1,:])
            else:
                embedding.append(hidden_state_sen[self.layer_use,1:sent_len-1,:])
        return embedding
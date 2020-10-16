import torch

def SBERT(ref, cand):
    c12 = torch.sum(ref.mm(cand.T))
    c11 = torch.sum(ref.mm(ref.T))
    c22 = torch.sum(cand.mm(cand.T))
    return (c12/torch.sqrt(c11*c22)).unsqueeze(0)

def cka(ref, cand):
    c12 = torch.sum((ref.mm(cand.T))**2)
    c11 = torch.sum((ref.mm(ref.T))**2)
    c22 = torch.sum((cand.mm(cand.T))**2)
    return (c12/torch.sqrt(c11*c22)).unsqueeze(0)

def bertscore(ref, cand):
    wv_prod = ref.mm(cand.T)
    r = torch.mean((torch.max(wv_prod,dim = 1)[0]))
    p = torch.mean((torch.max(wv_prod,dim = 0)[0]))
    f1 = 2*r*p/(r+p)
    return torch.stack([r, p, f1])

def TWMD(ref, cand, T, device, SK_iter=1):
    def compute(sent1, sent2):
        wv_prod = sent1.mm(sent2.T)
        L1, L2 = wv_prod.shape[-2:]
        logl1 = torch.log(torch.tensor(L1+0.0))
        logl2 = torch.log(torch.tensor(L2+0.0))

        log_pi = wv_prod/T
        for i in range(SK_iter-1):
            log_pi = torch.nn.LogSoftmax(dim=0)(log_pi)+logl1
            log_pi = torch.nn.LogSoftmax(dim=1)(log_pi)+logl2

        log_pi = torch.nn.LogSoftmax(dim=0)(log_pi)+logl1
        r = L2*torch.nn.Softmax(dim=1)(log_pi)

        log_pi = wv_prod/T
        for i in range(SK_iter-1):
           log_pi = torch.nn.LogSoftmax(dim=1)(log_pi)+logl2
           log_pi = torch.nn.LogSoftmax(dim=0)(log_pi)+logl1

        log_pi = torch.nn.LogSoftmax(dim=1)(log_pi)+logl2
        p = L1*torch.nn.Softmax(dim=0)(log_pi)

        R = torch.mean(r*wv_prod,dim=(0,1))

        P = torch.mean(p*wv_prod,dim=(0,1))
        F1 = 2*R*P/(R+P)
        return R, P, F1
    
    r12, p12, f1_12 = compute(ref, cand)
    r11, p11, f1_11 = compute(ref, ref)
    r22, p22, f1_22 = compute(cand, cand)
    
    r_norm, p_norm, f1_norm = r12/(torch.sqrt(r11*r22)), p12/(torch.sqrt(p11*p22)), f1_12/(torch.sqrt(f1_11*f1_22))
    return torch.stack([r_norm, p_norm, f1_norm])

def TRWMD(ref, cand, T):
    def compute(sent1, sent2):
        wv_prod = sent1.mm(sent2.T)
        softmax_r = torch.exp(torch.nn.LogSoftmax(dim=1)(wv_prod/beta_eval))
        r = torch.mean(torch.sum(soft_max_r*wv_prod,dim=1),dim=0)

        softmax_p= torch.exp(torch.nn.LogSoftmax(dim=0)(wv_prod/beta_eval))
        p = torch.mean(torch.sum(soft_max_p*wv_prod,dim=0),dim=0)
    
        f1 = 2*r*p/(r+p)
        return r, p, f1
    
    r12, p12, f1_12 = compute(ref, cand)
    r11, p11, f1_11 = compute(ref, ref)
    r22, p22, f1_22 = compute(cand, cand)
    
    r_norm, p_norm, f1_norm = r12/(torch.sqrt(r11*r22)), p12/(torch.sqrt(p11*p22)), f1_12/(torch.sqrt(f1_11*f1_22))
    return torch.stack([r_norm, p_norm, f1_norm])

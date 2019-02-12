'Model training for NLP'
from fastai.text import *

def text_classifier_learner(data:DataBunch, bptt:int=70, emb_sz:int=400, nh:int=1150, nl:int=3, pad_token:int=1,
               drop_mult:float=1., qrnn:bool=False,max_len:int=70*20, lin_ftrs:Collection[int]=None, 
               ps:Collection[float]=None, pretrained_model:str=None, **kwargs) -> 'TextClassifierLearner':
    "Create a RNN classifier from `data`."
    dps = default_dropout['classifier'] * drop_mult
    if lin_ftrs is None: lin_ftrs = [50]
    if ps is None:  ps = [0.1]
    vocab_size, n_class = len(data.vocab.itos), data.c
    layers = [emb_sz*3] + lin_ftrs + [n_class]
    ps = [dps[4]] + ps
    model = get_rnn_classifier(bptt, max_len, vocab_size, emb_sz, nh, nl, pad_token,
                layers, ps, input_p=dps[0], weight_p=dps[1], embed_p=dps[2], hidden_p=dps[3], qrnn=qrnn)
    learn = CustomRNNLearner(data, model, bptt, split_func=rnn_classifier_split, **kwargs)
    if pretrained_model is not None:
        model_path = untar_data(pretrained_model, data=False)
        fnames = [list(model_path.glob(f'*.{ext}'))[0] for ext in ['pth', 'pkl']]
        learn.load_pretrained(*fnames, strict=False)
        learn.freeze()
    return learn

class CustomRNNLearner(RNNLearner):
    "Basic class for a `Learner` in NLP."
    def __init__(self, data:DataBunch, model:nn.Module, bptt:int=70, split_func:OptSplitFunc=None, clip:float=None,
                 alpha:float=2., beta:float=1., metrics=None, **kwargs):
        super().__init__(data, model, **kwargs)

    def load_encoder(self, name:str):
        "Load the encoder `name` from the model directory."
        get_model(self.model)[0].load_state_dict(torch.load(self.path/self.model_dir/f'{name}.pth', map_location=torch.device('cpu')))
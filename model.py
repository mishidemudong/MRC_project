#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 15:08:54 2020

@author: liang
"""

import json, os, re
import numpy as np
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.snippets import sequence_padding
from bert4keras.snippets import open
from keras.layers import Lambda
from keras.models import Model
from tqdm import tqdm

def get_ngram_set(x, n):
    result = {}
    for i in range(len(x) - n + 1):
        k = tuple(x[i:i + n])
        if k[:-1] not in result:
            result[k[:-1]] = set()
        result[k[:-1]].add(k[-1])
    return result

def max_in_dict(d):
    if d:
        return sorted(d.items(), key=lambda s: -s[1])[0][0]

class ModelPredictHandler():
    
    def __init__(self, model_save_path):
        
        with open(os.path.join(model_save_path,'config.json')) as f:
            config = json.load(f)
        
        self.max_p_len = config['max_p_len']
        self.max_q_len = config['max_q_len']
        self.max_a_len = config['max_a_len']
        
        self.config_path = config['config_path'] 
        self.checkpoint_path = config['checkpoint_path']  
        self.dict_path = config['dict_path']
        
        self.best_weights_name = config['best_weights_name']
        
        self.buildmodel()
    
    def buildmodel(self):
        
        self.token_dict, self.keep_tokens = load_vocab(
            dict_path=self.dict_path,
            simplified=True,
            startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
        )
        self.tokenizer = Tokenizer(self.token_dict, do_lower_case=True)
        
        model = build_transformer_model(
            self.config_path,
            self.checkpoint_path,
            model='albert',
            with_mlm=True,
            keep_tokens=self.keep_tokens, 
        )
        output = Lambda(lambda x: x[:, 1:self.max_a_len + 1])(model.output)
        self.model = Model(model.input, output)
        
        self.model.load_weights(self.best_weights_name)
        
    def gen_answer(self, question, passages):
        '''
        feature:[CLS][MASK][MASK][SEP]question[SEP]passage[SEP]
        output:{answer:proba}
        '''
        all_p_token_ids, token_ids, segment_ids = [], [], []
        for passage in passages:
            passage = re.sub(u' |、|；|，', ',', passage)
            p_token_ids, _ = self.tokenizer.encode(passage, max_length=self.max_p_len + 1)
            q_token_ids, _ = self.tokenizer.encode(question, max_length=self.max_q_len + 1)
            all_p_token_ids.append(p_token_ids[1:])
            token_ids.append([self.tokenizer._token_start_id])
            token_ids[-1] += ([self.tokenizer._token_mask_id] * self.max_a_len)
            token_ids[-1] += [self.tokenizer._token_end_id]
            token_ids[-1] += (q_token_ids[1:] + p_token_ids[1:])
            segment_ids.append([0] * len(token_ids[-1]))
        token_ids = sequence_padding(token_ids)
        segment_ids = sequence_padding(segment_ids)
        probas = self.model.predict([token_ids, segment_ids])
        results = {}
        for t, p in zip(all_p_token_ids, probas):
            a, score = tuple(), 0.
            for i in range(self.max_a_len):
                idxs = list(get_ngram_set(t, i + 1)[a])
                if self.tokenizer._token_end_id not in idxs:
                    idxs.append(self.tokenizer._token_end_id)
                pi = np.zeros_like(p[i])
                pi[idxs] = p[i, idxs]
                a = a + (pi.argmax(),)
                score += pi.max()
                if a[-1] == self.tokenizer._token_end_id:
                    break
            score = score / (i + 1)
            a = self.tokenizer.decode(a)
            if a:
                results[a] = results.get(a, []) + [score]
        results = {
                    k: (np.array(v)**2).sum() / (sum(v) + 1)
                    for k, v in results.items()
                    }
        return results
        
    def predict(self, pred_data):
        for d in tqdm(pred_data):
            q_text = d['question']
            p_texts = d['passages']
            a = self.gen_answer(q_text, p_texts)
            a = max_in_dict(a)

def modelpredict(data, model_save_path):
    result = []
    model = ModelPredictHandler(model_save_path)
    
    for pred in data:
        tmp = model.predict(pred)
        result.append(tmp)
        
        
    return result
    
    
        
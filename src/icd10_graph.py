""" Generates a Direct Acyclic Graph (DAG) based on the ICD ontology
"""

import pandas as pd
import numpy as np
import re
import os
import pickle
import argparse
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import simple_icd_10 as icd
from collections import OrderedDict
from gensim.models import Word2Vec
import time


def train_icd_desc_w2v(icdcodes, args):
    """ Given an ICD code, get the embedded descriptions
    """
    icd_descs = []
    for icdcode in icdcodes:
        if icdcode == 'root':
            icd_descs.append('root')
        else:
            icd_descs.append(icd.get_description(icdcode))

    # Convert sentence to lowercase word list
    icd_descs = [x.lower().split(' ') for x in icd_descs]

    model = Word2Vec(min_count=args.min_count,
                    window=args.window,
                    size=args.size,
                    sample=args.sample, 
                    alpha=args.alpha, 
                    min_alpha=args.min_alpha, 
                    negative=args.negative,
                    workers=args.workers)

    t = time.time()
    model.build_vocab(icd_descs, progress_per=10000)
    print(f"Build Vocab: {time.time()-t} sec")
    
    t = time.time()
    model.train(icd_descs, total_examples=model.corpus_count, epochs=30, report_delay=1)
    print(f"Train Model: {time.time()-t} sec")
    # Freeze model, init_sims() will make model more memory-efficient
    model.init_sims(replace=True)
    
    # Save results
    # If dir does not exists, create new directory
    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)
    model_outpath = os.path.join(args.outdir, 'icd_desc_w2v.model')
    model.save(model_outpath)
    print(f"Saved model: {model}")

    return model

def generate_icd_desc_from_edgelist(edgelist, args):
    icd_edgelist = np.array(edgelist.copy())
    unique_icds = list(set(icd_edgelist.flatten()))
    unique_icds.sort()

    model = train_icd_desc_w2v(unique_icds, args)
    word_vecs = []
    for icdcode in unique_icds:
        if icdcode == 'root':
            icd_desc = ['root']
        else:
            word_vec = np.ones(args.size)
            icd_desc = icd.get_description(icdcode).lower().split(' ')
        for word in icd_desc:
            word_vec += model.wv[word]
        word_vec /= len(icd_desc)
        word_vecs.append(word_vec)
    
    df_results = pd.DataFrame(word_vecs)
    df_results.columns = [f'emb_{x}' for x in df_results.columns]
    df_results['icdcode'] = unique_icds

    return df_results


def encode_icd_edges(edgelist):
    """ Returns:
        1. DataFrane with encoded nodes
        2. Encoding mapper
    """
    icd_edgelist = np.array(edgelist.copy())
    unique_icds = list(set(icd_edgelist.flatten()))
    unique_icds.sort()
    le = preprocessing.LabelEncoder()
    le.fit(unique_icds)
    le_mapper = dict(zip(le.classes_, le.transform(le.classes_)))

    df = pd.DataFrame(edgelist)
    df.columns = ['child', 'parent']
    df['child'] = le.transform(df['child'].tolist())
    df['parent'] = le.transform(df['parent'].tolist())

    return df, le_mapper

def get_roman_nums(num):
    roman_char_dict = OrderedDict()
    roman_char_dict[1000] = "M"
    roman_char_dict[900] = "CM"
    roman_char_dict[500] = "D"
    roman_char_dict[400] = "CD"
    roman_char_dict[100] = "C"
    roman_char_dict[90] = "XC"
    roman_char_dict[50] = "L"
    roman_char_dict[40] = "XL"
    roman_char_dict[10] = "X"
    roman_char_dict[9] = "IX"
    roman_char_dict[5] = "V"
    roman_char_dict[4] = "IV"
    roman_char_dict[1] = "I"

    def get_roman_num(num):
        for r in roman_char_dict.keys():
            x, y = divmod(num, r)
            yield roman_char_dict[r] * x
            num -= (r*x)
            if num < 0:
                break
    return "".join([x for x in get_roman_num(num)])

def generate_edgelist():
    icd_edges = []
    leaf_nodes = []
    def get_child(keyword):
        code = icd.get_children(keyword)
        if len(code)==0:
            leaf_nodes.append(keyword)
            return
        else:
            # len(code) > 0
            for c in code:
                icd_edges.append([keyword, c])
                get_child(c)

    # There are 22 chapters in ICD10
    for i in range(1, 23):
        chapter = get_roman_nums(i)
        icd_edges.append(['root', chapter])
        get_child(chapter)
    
    return icd_edges, leaf_nodes


def main(args):
    
    icd_edges, leaf_nodes = generate_edgelist()
    # Generate DataFrame with encoded ICD10 edgelist
    df_encd_edgelist, encd_mapper = encode_icd_edges(icd_edges.copy())
    df_icd_emb = generate_icd_desc_from_edgelist(icd_edges.copy(), args)

    # Save Results
    # If dir does not exists, create new directory
    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    edgelist_outpath = os.path.join(args.outdir, 'edgelist.csv')
    mapper_outpath = os.path.join(args.outdir, 'encdmapper.pickle')
    emb_outpath = os.path.join(args.outdir, 'embs.csv')
    
    df_encd_edgelist.to_csv(edgelist_outpath, index=False)
    print(f"Edgelist saved: {edgelist_outpath}")

    df_icd_emb.to_csv(emb_outpath, index=False)
    print(f"Edgelist saved: {emb_outpath}")

    with open(mapper_outpath, 'wb') as fp:
        pickle.dump(encd_mapper, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Encoding Mapper saved: {mapper_outpath}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pytorch version of GraphSAGE')
    # parser.add_argument('--seed', type=int, default=824)
    parser.add_argument('--outdir', type=str, default='icd10-data')
    parser.add_argument('--min-count', type=int, default=1)
    parser.add_argument('--window', type=int, default=5)
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--sample', type=float, default=6e-5)
    parser.add_argument('--alpha', type=float, default=0.03)
    parser.add_argument('--min_alpha', type=float, default=0.0007)
    parser.add_argument('--negative', type=int, default=20)
    parser.add_argument('--workers', type=int, default=16)

    args = parser.parse_args()
    main(args)

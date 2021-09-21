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

def main(args):
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

    # Generate DataFrame with encoded ICD10 edgelist
    df_encd_edgelist, encd_mapper = encode_icd_edges(icd_edges.copy())

    # Save Results

    # If dir does not exists, create new directory
    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    edgelist_outpath = os.path.join(args.outdir, 'edgelist.csv')
    mapper_outpath = os.path.join(args.outdir, 'encdmapper.pickle')
    
    df_encd_edgelist.to_csv(edgelist_outpath, index=False)
    print(f"Edgelist saved: {edgelist_outpath}")

    with open(mapper_outpath, 'wb') as fp:
        pickle.dump(encd_mapper, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Encoding Mapper saved: {mapper_outpath}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pytorch version of GraphSAGE')
    # parser.add_argument('--seed', type=int, default=824)
    parser.add_argument('--outdir', type=str, default='icd10-data')
    args = parser.parse_args()
    main(args)

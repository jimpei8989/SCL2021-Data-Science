import sys
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from mapping import get_args, split_POI_street

def is_match(address, target):
    t_list = target.replace(',', ' ').split()
    a_list = address.replace(',', ' ').split()
    if len(a_list) < len(t_list) or len(t_list) == 0:
        return False
    for i in range(len(a_list) - len(t_list) + 1):
        if t_list[0].startswith(a_list[i]):
            same = True
            for j in range(1, len(t_list)):
                if not t_list[j].startswith(a_list[i+j]):
                    same = False
                    break
            if same:
                return True
    return False

def predict(address, targets):
    for target in targets:
        if is_match(address, target):
            return target
    return ''

def predict_all(df, targets):
    return Parallel(n_jobs=-1)(delayed(predict)(address, targets) for address in tqdm(df['raw_address'], total=len(df['raw_address'])))

def reduce_dup_and_sort(df, column):
    result = df[column].unique()
    result = sorted(result, reverse=True)
    result = sorted(result, key=lambda x: len(x.replace(',', ' ').split()), reverse=True)
    return result

if __name__ == '__main__':
    args = get_args()
    mapping_df = pd.read_csv(args.mapping)
    mapping_df = split_POI_street(mapping_df)
    poi = reduce_dup_and_sort(mapping_df, 'POI')
    input_df = pd.read_csv(args.input)
    street = reduce_dup_and_sort(mapping_df, 'street')
    input_df['POI'] = predict_all(input_df, poi)
    input_df['street'] = predict_all(input_df, street)
    input_df['POI/street'] = input_df['POI'] + '/' + input_df['street']
    input_df.to_csv(args.output, columns=['id', 'POI/street'], index=False)

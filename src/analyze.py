import argparse
import pandas as pd

from mapping import mapping_data, split_POI_street

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help='input csv')
    parser.add_argument('--mapping', '-m', help='csv of mapping source')

    return parser.parse_args()

if __name__ == '__main__':
    
    args = get_args()
    input_df = pd.read_csv(args.input)
    split_input_df = split_POI_street(input_df)
    for poi, street in zip(split_input_df['POI'], split_input_df['street']):
        
    exit()
    mapping_df = pd.read_csv(args.mapping)
    mapped_df = mapping_data(input_df, mapping_df)
    input_df = pd.read_csv(args.input)

    acc = {True: {True: 0, False: 0}, False: {True: 0, False: 0}}
    print((input_df['raw_address'] == mapped_df['raw_address']).mean())
    tf_list = []
    ff_list = []
    for address, target, mapped_address in zip(input_df['raw_address'], input_df['POI/street'], mapped_df['raw_address']):
        target_poi, target_street = target.split('/')
        origin_true = target_poi in address and target_street in address
        mapped_true = target_poi in mapped_address and target_street in mapped_address
        acc[origin_true][mapped_true] += 1
        if origin_true and not mapped_true:
            tf_list.append([address, target, mapped_address])
        if not origin_true and not mapped_true:
            ff_list.append([address, target, mapped_address])

    print(acc)
    print("TF")
    print(tf_list[:20])
    print("FF")
    print(ff_list[:20])
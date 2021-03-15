import argparse
import pandas as pd

from mapping import mapping_data, split_POI_street

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help='input csv')
    parser.add_argument('--mapping', '-m', help='csv of mapping source')
    parser.add_argument('--output', '-o', help='output csv')

    return parser.parse_args()

if __name__ == '__main__':
    
    args = get_args()
    input_df = pd.read_csv(args.input)
    output_df = pd.read_csv(args.output)

    input_df = input_df.loc[output_df['id']].reset_index()

    if 'POI/street' in input_df:
        score = (input_df['POI/street'] == output_df['POI/street']).mean()
        print(f'accuracy = {score * 100}%')

    total = len(input_df['raw_address'])
    count = 0
    for data, predict in zip(input_df['raw_address'], output_df['POI/street']):
        poi, street = predict.split('/')
        if poi in data and street in data:
            count += 1
    print(f'in address: {count} / {total} = {count / total * 100}%')

    mapping_df = pd.read_csv(args.mapping)
    mapping_df = split_POI_street(mapping_df)

    count = 0
    for data, predict in zip(mapping_data(input_df, mapping_df)['raw_address'], output_df['POI/street']):
        poi, street = predict.split('/')
        if poi in data and street in data:
            count += 1
    print(f'in mapped address: {count} / {total} = {count / total * 100}%')    
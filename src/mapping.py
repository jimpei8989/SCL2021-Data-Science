import pandas as pd
import sys
from tqdm import tqdm
from collections import defaultdict

def apply_mapping(s, mapping, mapping_2={}):
    tmp_s = []
    for words in s.split(','):
        tmp_words = []
        word_list = words.split(' ')
        i = 0
        while i < len(word_list):
            if i > 0 and f'prev_{word_list[i-1]}' in mapping_2 and word_list[i] in mapping_2[f'prev_{word_list[i-1]}']:
                tmp_words.append(mapping_2[f'prev_{word_list[i-1]}'][word_list[i]])
            elif i+1 < len(word_list) and word_list[i] in mapping_2 and word_list[i+1] in mapping_2[word_list[i]]:
                tmp_words.append(mapping_2[word_list[i]][word_list[i+1]])
            else:
                tmp_words.append(word_list[i] if word_list[i] not in mapping else mapping[word_list[i]])
            i += 1
        tmp_s.append(' '.join(tmp_words))
    new_s = ','.join(tmp_s)

    return new_s

class Mapping:
    def __init__(self, address, target):
        self.address = address
        self.target = target
        self.a_list = address.replace(',', ' ').split()
        self.t_list = target.replace(',', ' ').split()
        self.size = len(self.t_list)
        self.result = {}
        if len(self.t_list) <= len(self.a_list):
            self.brutal_find_mapping()
        self.mapping()

    def messaging(self):  
        self.message = ''
        self.message += f'target = {self.target}\n'
        self.message += f'origin address = {self.address}\n'
        self.message += f'mapping = {self.result}\n'
        self.message += f'new address = {self.new_address}\n'
        self.message += f'match ? {self.target in self.new_address}\n'

    def matching(self):
        self.match = self.target in self.new_address
        self.match_without_comma = self.target.replace(',', '') in self.new_address.replace(',', '')
        self.match_without_comma_space = self.target.replace(',', '').replace(' ','') in self.new_address.replace(',', '').replace(' ','')

    def mapping(self):
        self.new_address = apply_mapping(self.address, self.result)
        self.messaging()
        self.matching()

    def mapping_index_to_result(self, mapping_index):
        for i in range(self.size):
            address_part = self.a_list[mapping_index[i]]
            target_part = self.t_list[i]
            self.result[address_part] = target_part
    
    def brutal_find_mapping(self):
        best_score = -1
        best_mapping_index = None
        for i in range(len(self.a_list) - self.size + 1):
            mapping_index = list(range(i, i + self.size))
            score = self.calculate_score(mapping_index)
            if score > best_score:
                best_score = score
                best_mapping_index = mapping_index
                if score == self.size:
                    break

        self.mapping_index_to_result(best_mapping_index)
        
    def calculate_score(self, mapping_index):
        score = 0
        for i in range(self.size):
            address_part = self.a_list[mapping_index[i]]
            target_part = self.t_list[i]
            if address_part.startswith(target_part) or target_part.startswith(address_part):
                score += 1
        return score

class Mapping_2(Mapping):
    def __init__(self, address, target, dup_mapping):
        self.dup_mapping = dup_mapping
        self.result_2 = defaultdict(dict)
        super().__init__(address, target)

    def messaging(self):
        super().messaging()
        self.message += f'mapping_2 = {self.result_2}\n'

    def mapping(self):
        self.new_address = apply_mapping(self.address, self.result, self.result_2)
        self.messaging()
        self.matching()

    def mapping_index_to_result(self, mapping_index):
        for i in range(self.size):
            address_part = self.a_list[mapping_index[i]]
            target_part = self.t_list[i]
            self.result[address_part] = target_part
            if address_part in self.dup_mapping:
                if mapping_index[i] > 0:
                    prev_address_part = self.a_list[mapping_index[i] - 1]
                    self.result_2[f'prev_{prev_address_part}'][address_part] = target_part
                if mapping_index[i] < len(self.a_list) - 1:
                    next_address_part = self.a_list[mapping_index[i] + 1]
                    self.result_2[address_part][next_address_part] = target_part
        
def calculate_mapping_score(address_s, target_s, mapping, mapping_2={}):
    total = len(address_s)
    count = 0
    count_without_comma = 0
    count_without_comma_space = 0
    for address, target in zip(address_s, target_s):
        new_address = apply_mapping(address, mapping, mapping_2)

        count += target in new_address
        count_without_comma += target.replace(',', '') in new_address.replace(',', '')
        count_without_comma_space += target.replace(',', '').replace(' ', '') in new_address.replace(',', '').replace(' ', '')

    print(f'score = {count} / {total} = {count / total}')
    print(f'without comma = {count_without_comma} / {total} = {count_without_comma / total}')
    print(f'without comma and space = {count_without_comma_space} / {total} = {count_without_comma_space / total}')

def find_mapping(address_s, target_s_list, return_dup=False, return_unique=False):
    count = 0
    mapping = defaultdict(lambda: defaultdict(int))
    if not isinstance(target_s_list, list):
        target_s_list = [target_s_list]

    for target_s in target_s_list:
        for addr, target in zip(address_s, target_s):
            tmp_mapping = Mapping(addr, target)
            for key in tmp_mapping.result:
                mapping[key][tmp_mapping.result[key]] += 1
            if not tmp_mapping.match_without_comma_space:
                count += 1
    
    print(f'invalid count / total count = {count} / {len(address_s) * len(target_s_list)} = {count / len(address_s) / len(target_s_list)}')

    if not return_dup and not return_unique:
        return mapping

    result = [mapping]
    if return_dup:
        result.append(find_dup_mapping(mapping))
    if return_unique:
        result.append(find_unique_mapping(mapping))
    return result

def find_dup_mapping(mapping):
    dup_key = 0
    dup_value = 0
    dup_mapping = {}
    total_value = 0
    for key, value in mapping.items():
        if len(value) > 1:
            dup_key += 1
            dup_value += len(value)
            dup_mapping[key] = value
        total_value += len(value)
    
    print(f'duplicate key / total key = {dup_key} / {len(mapping)} = {dup_key / len(mapping)}')
    print(f'duplicate value / total value = {dup_value} / {total_value} = {dup_value / total_value}')

    return dup_mapping

def find_unique_mapping(mapping):
    return {key:max(value, key=lambda k: value[k]) for key, value in mapping.items()}

def find_mapping_2(address_s, target_s_list, dup_mapping=None, return_unique=False):
    count = 0
    # mapping = defaultdict(dict)
    mapping = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    if not isinstance(target_s_list, list):
        target_s_list = [target_s_list]

    for target_s in target_s_list:
        for addr, target in zip(address_s, target_s):
            tmp_mapping = Mapping_2(addr, target, dup_mapping)
            for key in tmp_mapping.result_2:
                for k in tmp_mapping.result_2[key]:
                    mapping[key][k][tmp_mapping.result_2[key][k]] += 1
            if not tmp_mapping.match_without_comma_space:
                count += 1
    print(f'invalid count / total count = {count} / {len(address_s) * len(target_s_list)} = {count / len(address_s) / len(target_s_list)}')

    if not return_unique:
        return mapping

    return mapping, {key:{k:max(v, key=lambda x: v[x]) for k, v in value.items()} for key, value in mapping.items()} 

if __name__ == '__main__':
    df = pd.read_csv(sys.argv[1])
    df[['POI','street']] = df['POI/street'].str.split('/',expand=True,)
    df.drop(columns='POI/street', inplace=True)

    mapping_poi, dup_mapping_poi, unique_mapping_poi = find_mapping(df['raw_address'], df['POI'], return_dup=True, return_unique=True)
    mapping_street, dup_mapping_street, unique_mapping_street = find_mapping(df['raw_address'], df['street'], return_dup=True, return_unique=True)
    # mapping, dup_mapping, unique_mapping = find_mapping(df['raw_address'], [df['POI'], df['street']], return_dup=True, return_unique=True)

    calculate_mapping_score(df['raw_address'], df['POI'], unique_mapping_poi)
    # calculate_mapping_score(df['raw_address'], df['POI'], unique_mapping)
    calculate_mapping_score(df['raw_address'], df['street'], unique_mapping_street)
    # calculate_mapping_score(df['raw_address'], df['street'], unique_mapping)

    mapping_2_poi, unique_mapping_2_poi = find_mapping_2(df['raw_address'], df['POI'], dup_mapping_poi, return_unique=True)
    mapping_2_street, unique_mapping_2_street = find_mapping_2(df['raw_address'], df['street'], dup_mapping_street, return_unique=True)
    calculate_mapping_score(df['raw_address'], df['POI'], unique_mapping_poi, unique_mapping_2_poi)
    calculate_mapping_score(df['raw_address'], df['street'], unique_mapping_street, unique_mapping_2_street)
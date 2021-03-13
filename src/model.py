from transformers import BertTokenizer, BertModel

def get_default_model(model_name='cahya/bert-base-indonesian-522M'):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    return tokenizer, model

def tokenize_addr(tokenizer, text):
    encoded_input = tokenizer(text, return_tensors='pt')
    tokened_res = [tokenizer.convert_ids_to_tokens(c) for c in encoded_input['input_ids']]

    return encoded_input, tokened_res

if __name__ == '__main__':
    tokenizer, model = get_default_model()

    text = "graha tirta,tirta dahlia no.5,waru,sidoarjo"
    encoded_input, tokened_res = tokenize_addr(tokenizer, text)
    print(encoded_input, '\n', tokened_res)

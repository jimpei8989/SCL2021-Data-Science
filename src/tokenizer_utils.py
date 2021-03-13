def tokenize_addr(tokenizer, text: str):
    encoded_input = tokenizer(text, return_tensors="pt")
    tokened_res = [tokenizer.convert_ids_to_tokens(c) for c in encoded_input["input_ids"]]
    return encoded_input, tokened_res

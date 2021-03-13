import pandas as pd

def check_dataset(args):
    train_df = pd.read_csv(args.dataset_dir / "train.csv")

    total_length = len(train_df)

    assert_split_length = 0
    for extract in train_df["POI/street"]:
        if len(extract.split("/")) != 2:
            assert_split_length += 1

    print(f"! Extract length after split is not 2 - {assert_split_length} / {total_length}")

    assert_continuous = 0
    for ID, address, extract in zip(
        train_df["id"], train_df["raw_address"], train_df["POI/street"]
    ):
        address = address.replace("sd neg ", "sd negeri ")
        address = address.replace("yaya ", "yayasan ")
        poi, street = extract.split("/")
        if not (poi in address and street in address):
            assert_continuous += 1
            # print(f"{ID} - {address} - {extract}")

    print(f"! Not continuous - {assert_continuous} / {total_length}")

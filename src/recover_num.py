import pandas as pd
import re
from argparse import ArgumentParser


def parse():
    parser = ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--source", type=str, default="dataset/test.csv")
    parser.add_argument("--mapping_first", action="store_true")

    return parser.parse_args()


def compute_score(source, target):
    score = 0
    for s, t in zip(source, target):
        score += t.startswith(s) or (
            ("0" in t)
            and len(re.findall("\d+", s)) == len(re.findall("\d+", t))
            and re.sub("\d+", "", t).startswith(re.sub("\d+", "", s).strip(","))
        )
        # print(re.sub("\d+", "", t), re.sub("\d+", "", s))
    # print(source, target, score)

    return score


def substitute(raw_addr_list, poi_street_list, mapping):
    score = [
        (
            compute_score(raw_addr_list[i : i + len(poi_street_list)], poi_street_list),
            i,
        )
        for i in range(len(raw_addr_list) - len(poi_street_list) + 1)
    ]

    max_index = max(score)[1]
    target = raw_addr_list[max_index : max_index + len(poi_street_list)]

    for i, t in enumerate(target):
        if re.match(".*\d+.*", t):
            if mapping:
                tmp_list = poi_street_list[i].split("0")
                res = tmp_list[0]
                num = re.findall("\d+", t)
                for j in range(1, len(tmp_list)):
                    res += num[j - 1] + tmp_list[j]

                poi_street_list[i] = res
            else:
                poi_street_list[i] = t

    return " ".join(poi_street_list)


def recover_num(input_df, source_df, mapping=False):
    result = []
    for idx, (raw_addr, poi_street) in enumerate(
        zip(source_df["raw_address"], input_df)
    ):
        poi_street_list = poi_street.split(" ")
        raw_addr_list = raw_addr.split(" ")
        # print(poi_street_list, raw_addr_list, idx)
        if "0" in poi_street and len(raw_addr_list) >= len(poi_street_list):
            result.append(substitute(raw_addr_list, poi_street_list, mapping))
        else:
            result.append(poi_street)

    return result


def recover(args):
    input_df = pd.read_csv(args.input)
    source_df = pd.read_csv(args.source)

    input_df[["POI", "street"]] = input_df["POI/street"].str.split(
        "/",
        expand=True,
    )

    input_df["POI"] = recover_num(input_df["POI"], source_df, args.mapping_first)
    input_df["street"] = recover_num(input_df["street"], source_df, args.mapping_first)
    input_df["POI/street"] = input_df["POI"] + "/" + input_df["street"]

    return input_df


if __name__ == "__main__":
    args = parse()
    input_df = recover(args)
    input_df.to_csv(args.output, columns=["id", "POI/street"], index=False)
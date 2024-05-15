import numpy as np
import pandas as pd

from collections import defaultdict


def list_duplicates(seq):
    tally = defaultdict(list)
    for i, item in enumerate(seq):
        tally[item].append(i)
    return ((key, locs) for key, locs in tally.items()
            if len(locs) > 1)


def check_duplicates():
    df_passages = pd.read_json("data/mdd_dpr/dpr.psg.multidoc2dial_all.structure.json")
    duplicate_indexes = df_passages["text"].duplicated()

    print(f"Duplicated passages percent: {duplicate_indexes.mean()}")

    # prints list of {text [duplicate id 1, duplicate id 2]
    for dup in sorted(list_duplicates(df_passages["text"])):
        print(dup)


def is_pos_only(DPR_result_list):
    return list(map(lambda x: x["is_positive"], DPR_result_list))


def count_rank_positives():
    df_dpr_res = pd.read_json("DPR_multidoc2dial_50q.json")
    df_dpr_res_rnd = pd.read_json("DPR_multidoc2dial_50q_rnd.json")

    def get_positives(df):
        list_results = df["DPR_result"].apply(is_pos_only)
        positives = []
        for list_r in list_results:
            if None in list_r:
                continue
            positives.append(list(map(lambda x: int(x), list_r)))
        return positives

    pos = get_positives(df_dpr_res)
    pos_rnd = get_positives(df_dpr_res_rnd)
    pos_res_final = pos + pos_rnd
    pos_res_final = np.array(pos_res_final)

    rank_positives = np.sum(pos_res_final, axis=0)
    for i, r in enumerate(rank_positives):
        print(f"{i}. rank - {r}/{len(pos) + len(pos_rnd)} pozitivnich ")


if __name__ == "__main__":
    # check_duplicates()

    count_rank_positives()

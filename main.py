import json
import os

from transformers import AutoModel, AutoTokenizer, DPRContextEncoderTokenizer, DPRContextEncoder
import pandas as pd
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Computes average number of tokens in list of tokenized input passages
def get_token_avg(tokenized_input):
    input_ids = tokenized_input.data['input_ids'].tolist()
    input_ids_no_zero = input_ids.apply(lambda x: np.trim_zeros(np.array(x)).tolist())
    input_ids_no_zero_len = input_ids_no_zero.apply(lambda x: len(x) - 2)
    return input_ids_no_zero_len.mean()


# Transforms passages to embeddings
def create_doc_embeddings():
    dtokenizer = AutoTokenizer.from_pretrained("sivasankalpp/dpr-multidoc2dial-structure-ctx-encoder")
    dmodel = DPRContextEncoder.from_pretrained("sivasankalpp/dpr-multidoc2dial-structure-ctx-encoder")
    dmodel = dmodel.to(device)

    batch_size = 256

    outputs = []
    num_batches = len(df_passages["text"]) // batch_size
    for i in range(num_batches + 1):
        start_idx = i * batch_size
        end_idx = np.minimum((i + 1) * batch_size, len(df_passages["text"]))
        print(f"Processing from {i * batch_size} to {end_idx}.")
        batch = df_passages["text"].tolist()[start_idx:end_idx]
        inputs_d = dtokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        inputs_d.to(device)

        with torch.no_grad():
            outputs_d = dmodel(**inputs_d)

        outputs.append(outputs_d.pooler_output)

        del inputs_d, outputs_d

        # Clear GPU cache
        torch.cuda.empty_cache()

    doc_embeddings = torch.cat(outputs, 0)
    print(doc_embeddings)
    print(doc_embeddings.shape)


# Input data are separated with [SEP], this function breaks it back to pair (X, Y) by [SEP]
def break_to_pair(sentence):
    res = sentence.split("[SEP]")
    return res[0], res[1]


def get_top_passages(questions, k):
    questions = [break_to_pair(q) for q in questions]
    inputs_q = qtokenizer(questions, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs_q.to(device)
    outputs_q = qmodel(**inputs_q)

    with torch.no_grad():
        similarity = outputs_q.pooler_output @ doc_embeddings.transpose(0, 1)

    similarity = similarity.cpu()

    passage_ids = []
    scores = []
    texts = []
    for i in range(len(similarity)):
        top_k = np.argpartition(similarity[i], -k)[-k:]
        top_k_sorted = top_k[np.argsort(-similarity[i][top_k])]

        passage_ids.append(top_k_sorted)
        scores.append(similarity[i][top_k_sorted])
        texts.append(df_passages["text"].loc[top_k_sorted])

    return passage_ids, scores, texts


if __name__ == "__main__":
    df_passages = pd.read_json("data/mdd_dpr/dpr.psg.multidoc2dial_all.structure.json")
    df_queries = pd.read_json("data/mdd_dpr/dpr.multidoc2dial_all.structure.train.json")

    df_queries["positive_ctx_len"] = df_queries["positive_ctxs"].apply(len)
    df_queries["positive_passage_id"] = df_queries["positive_ctxs"].apply(lambda x: x[0]["psg_id"])
    df_queries["positive_passage_text"] = df_queries["positive_ctxs"].apply(lambda x: x[0]["text"])

    doc_embeddings = torch.load("doc_embeddings.pt")
    doc_embeddings.to(device)

    qtokenizer = AutoTokenizer.from_pretrained("sivasankalpp/dpr-multidoc2dial-structure-question-encoder")
    qmodel = AutoModel.from_pretrained("sivasankalpp/dpr-multidoc2dial-structure-question-encoder")
    qmodel.to(device)

    question_chunk = 15
    rnd_index = np.random.randint(low=50, high=df_queries.shape[0], size=15)
    rnd_index1 = rnd_index + 1
    rnd_index2 = rnd_index + 2
    rnd_index = np.array([rnd_index, rnd_index1, rnd_index2]).flatten()

    rnd_index.sort()

    questions = list(df_queries["question"][rnd_index])
    positive_passage_ids = list(df_queries["positive_passage_id"][rnd_index])
    positive_passage_texts = list(df_queries["positive_passage_text"][rnd_index])

    passage_ids, scores, texts = get_top_passages(questions, 10)
    zipped_results = list(map(list, zip(passage_ids, scores, texts)))

    zip_questions = []
    future_json = zip(questions, positive_passage_texts, positive_passage_ids, zipped_results)
    for question, positive_passage, passage_id, result in future_json:

        DPR_results = []
        found_at_rank = None
        # result has shape (3, 10) -> zip transforms it to (10, 3)
        # for each of 10 results get passage id, score and text
        for rank, (res_passage_id, score, text) in enumerate(zip(*result)):
            res_passage_id = res_passage_id.item()
            DPR_results.append(
                {
                    "rank": rank,
                    "passage_id": res_passage_id,
                    "is_target": res_passage_id == passage_id,
                    "score": score.item(),
                    "text": text
                }
            )
            if res_passage_id == passage_id:
                found_at_rank = rank

        zip_questions.append(
            {
                "question": question,
                "positive_passage": positive_passage,
                "positive_passage_id": passage_id,
                "found_at_rank": found_at_rank,
                "DPR_result": DPR_results
            }
        )

    dpr_result_path = "DPR_results.json"
    with open(dpr_result_path, "w") as out_f:
        json.dump(zip_questions, out_f, indent=4)

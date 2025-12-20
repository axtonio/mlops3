import os

import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "model_repository/faiss_search/1/faiss.index"


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def generate_embeddings(n_samples=100000, batch_size=256):
    print(f"Loading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID)
    model.eval()

    print(f"Generating {n_samples} dummy texts...")
    texts = [
        f"This is a dummy object number {i} for testing embedding generation."
        for i in range(n_samples)
    ]

    all_embeddings = []

    print("Computing embeddings...")
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i : i + batch_size]

        encoded_input = tokenizer(
            batch_texts, padding=True, truncation=True, return_tensors="pt"
        )

        with torch.no_grad():
            model_output = model(**encoded_input)

        sentence_embeddings = mean_pooling(
            model_output, encoded_input["attention_mask"]
        )

        sentence_embeddings = torch.nn.functional.normalize(
            sentence_embeddings, p=2, dim=1
        )
        all_embeddings.append(sentence_embeddings.numpy())

    all_embeddings = np.vstack(all_embeddings)
    return all_embeddings


def create_index(embeddings):
    d = embeddings.shape[1]
    print(f"Creating FAISS index for dimension {d}...")

    index = faiss.IndexFlatL2(d)
    index.add(embeddings)

    print(f"Index contains {index.ntotal} vectors.")
    return index


def main():
    if not os.path.exists(os.path.dirname(INDEX_PATH)):
        os.makedirs(os.path.dirname(INDEX_PATH))

    embeddings = generate_embeddings()
    index = create_index(embeddings)

    print(f"Saving index to {INDEX_PATH}")
    faiss.write_index(index, INDEX_PATH)
    print("Done.")


if __name__ == "__main__":
    main()

import unittest

import torch
import tritonclient.http as httpclient
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
TRITON_URL = "localhost:8000"
MODEL_NAME = "faiss_search"


class TestTritonFaiss(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Setting up local embedder...")
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        cls.model = AutoModel.from_pretrained(MODEL_ID)
        cls.model.eval()

    def get_embedding(self, text):
        inputs = self.tokenizer(
            [text], padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            output = self.model(**inputs)

        token_embeddings = output[0]
        attention_mask = inputs["attention_mask"]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding.numpy()

    def test_nearest_neighbors(self):
        try:
            client = httpclient.InferenceServerClient(url=TRITON_URL)
            if not client.is_server_live():
                self.skipTest("Triton server is not live")
        except Exception as e:
            print(f"Could not connect to Triton: {e}")
            self.skipTest("Triton server not reachable")

        test_indices = [0, 123, 9999, 54321]

        for idx in test_indices:
            text = (
                f"This is a dummy object number {idx} for testing embedding generation."
            )
            vector = self.get_embedding(text)

            inputs = [httpclient.InferInput("INPUT", vector.shape, "FP32")]
            inputs[0].set_data_from_numpy(vector)

            outputs = [httpclient.InferRequestedOutput("OUTPUT")]

            response = client.infer(
                model_name=MODEL_NAME, inputs=inputs, outputs=outputs
            )
            result_ids = response.as_numpy("OUTPUT")

            print(f"Query Index: {idx}, Returned Neighbors: {result_ids[0]}")
            self.assertEqual(
                result_ids[0][0],
                idx,
                f"Expected nearest neighbor to be {idx}, but got {result_ids[0][0]}",
            )


if __name__ == "__main__":
    unittest.main()

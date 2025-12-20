import json
import os

import faiss
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):

        self.model_config = json.loads(args["model_config"])

        base_path = os.path.dirname(__file__)
        index_path = os.path.join(base_path, "faiss.index")

        if not os.path.exists(index_path):
            raise pb_utils.TritonModelException(f"Index file not found at {index_path}")

        print(f"Loading FAISS index from {index_path}...")
        self.index = faiss.read_index(index_path)
        self.k = 5
        print("FAISS index loaded successfully.")

    def execute(self, requests):

        responses = []
        for request in requests:
            in_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")
            vectors = in_tensor.as_numpy()

            distances, indices = self.index.search(vectors, self.k)

            out_tensor = pb_utils.Tensor("OUTPUT", indices.astype(np.int64))

            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

        return responses

    def finalize(self):
        print("Cleaning up FAISS model...")

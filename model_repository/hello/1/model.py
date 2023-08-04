import triton_python_backend_utils as pb_utils
import numpy as np
from transformers import pipeline

class TritonPythonModel:
    def initialize(self, args):
        model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.generator = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)

    def execute(self, requests):
        responses = []
        for request in requests:
            # Decode the Byte Tensor into Text
            input = pb_utils.get_input_tensor_by_name(request, "text")
            input_text = input.as_numpy()[0].decode()
            # Call the Model pipeline
            pipeline_output = self.generator(input_text)
            sentiment = pipeline_output[0]["label"]
            # Encode the text to byte tensor to send back
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor("sentiment", np.array([sentiment.encode()]))]
            )
        responses.append(inference_response)
        return responses

    def finalize(self, args):
        self.generator = None

import numpy as np
import torch
import onnxruntime
from typing import Tuple, List, Union


class PyOnnx:
    def __init__(self, onnx_path):
        # model = onnx.load(onnx_path)
        # onnx.checker.check_model(model)
        # print(onnx.helper.printable_graph(model.graph))

        self.ort_session = onnxruntime.InferenceSession(onnx_path,
                                                        providers=['CPUExecutionProvider', 'CUDAExecutionProvider'])
        self.onnx_input_names = [o_input.name for o_input in self.ort_session.get_inputs()]
        self.onnx_output_names = [o_output.name for o_output in self.ort_session.get_outputs()]
        return

    def __call__(self, input_tensor: Union[Tuple[np.ndarray], np.ndarray]) -> List[np.ndarray]:
        input_param = dict()
        if isinstance(input_tensor, np.ndarray):
            assert len(self.onnx_input_names) == 1
            input_param[self.onnx_input_names[0]] = input_tensor
        else:
            assert len(input_tensor) == len(self.onnx_input_names)
            for i in range(len(self.onnx_input_names)):
                input_param[self.onnx_input_names[i]] = input_tensor[i]

        outputs = self.ort_session.run(None, input_param)
        return outputs


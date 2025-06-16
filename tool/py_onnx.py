import numpy as np
import torch
import onnxruntime
from typing import Tuple, Sequence, Union


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

    def __call__(self, input_tensor: Union[Tuple[np.ndarray], np.ndarray]) -> Union[None, Sequence[np.ndarray]]:

        if isinstance(input_tensor, np.ndarray):
            input_tensor = [input_tensor]

        if len(self.onnx_input_names) != len(input_tensor):
            print(f'传入数据和模型输入个数不匹配 {len(self.onnx_input_names)} != {len(input_tensor)}')
            return None

        input_param = dict()
        for i in range(len(self.onnx_input_names)):
            input_param[self.onnx_input_names[i]] = input_tensor[i]

        try:
            outputs = self.ort_session.run(None, input_param)
        except Exception as err:
            print(f'推理出错: {err}')
            outputs = None
        return outputs


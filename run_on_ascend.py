import acl
import cv2
import numpy as np
import sys
from fastapi import FastAPI, Body
import argparse
import uvicorn
import base64
import os
import re



# 封装acl模型推理过程为OMNet类
class OMNet:
    ACL_MEM_MALLOC_HUGE_FIRST = 0
    ACL_MEMCPY_HOST_TO_DEVICE = 1
    ACL_MEMCPY_DEVICE_TO_HOST = 2

    def __init__(self, model_path, device_id, out_type):
        # 初始化函数
        self.device_id = device_id

        # step1: 初始化
        ret = acl.init()
        assert ret == 0, f"初始化失败: {ret}"
        # 指定运算的Device
        ret = acl.rt.set_device(self.device_id)
        assert ret == 0, f"设置device{self.device_id}失败: {ret}"

        # step2: 加载模型
        # 加载离线模型文件，返回标识模型的ID        
        self.model_id, ret = acl.mdl.load_from_file(model_path)
        assert ret == 0, f"加载模型失败 {model_path}: {ret}"

        # 创建空白模型描述信息，获取模型描述信息的指针地址
        self.model_desc = acl.mdl.create_desc()
        # 通过模型的ID，将模型的描述信息填充到model_desc
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        # step3：创建输入输出数据集
        # 创建输入数据集
        self.input_dataset, self.input_data = self.prepare_dataset('input')
        # 创建输出数据集
        self.output_dataset, self.output_data = self.prepare_dataset('output')

        self.out_type = out_type
        return

    def prepare_dataset(self, io_type):
        # 准备数据集
        if io_type == "input":
            # 获得模型输入的个数
            io_num = acl.mdl.get_num_inputs(self.model_desc)
            acl_mdl_get_size_by_index = acl.mdl.get_input_size_by_index
        else:
            # 获得模型输出的个数
            io_num = acl.mdl.get_num_outputs(self.model_desc)
            acl_mdl_get_size_by_index = acl.mdl.get_output_size_by_index
        assert io_num > 0, f"获取模型{io_type}输入个数失败: {io_num}"
        # 创建aclmdlDataset类型的数据，描述模型推理的输入。
        dataset = acl.mdl.create_dataset()
        datas = []
        for i in range(io_num):
            # 获取所需的buffer内存大小
            buffer_size = acl_mdl_get_size_by_index(self.model_desc, i)
            # 申请buffer内存
            buffer, ret = acl.rt.malloc(buffer_size, self.ACL_MEM_MALLOC_HUGE_FIRST)
            assert ret == 0, f"申请{io_type}buffer内存出错: {ret}"
            # 从内存创建buffer数据
            data_buffer = acl.create_data_buffer(buffer, buffer_size)
            # 将buffer数据添加到数据集
            _, ret = acl.mdl.add_dataset_buffer(dataset, data_buffer)
            assert ret == 0, f"将{io_type}buffer添加到数据集出错: {ret}"

            datas.append({"buffer": buffer, "data": data_buffer, "size": buffer_size})
        return dataset, datas

    def forward(self, inputs):
        # 执行推理任务
        # 遍历所有输入，拷贝到对应的buffer内存中
        copy_success = True
        input_num = len(inputs)
        for i in range(input_num):
            bytes_data = inputs[i].tobytes()
            bytes_ptr = acl.util.bytes_to_ptr(bytes_data)
            if len(bytes_data) != self.input_data[i]["size"]:
                print(f"输入数据的大小: {len(bytes_data)} 和模型的输入大小 {self.input_data[i]['size']} 不匹配")
                copy_success = False
                break

            # 先清空buffer
            ret = acl.rt.memset(self.input_data[i]["buffer"], self.input_data[i]["size"], 0, 2)
            if ret != 0:
                print(f"清空输入buffer失败: {ret}")
                copy_success = False
                break

            # 将数据从Host传输到Device。
            ret = acl.rt.memcpy(self.input_data[i]["buffer"],  # 目标地址 device
                                self.input_data[i]["size"],  # 目标地址大小
                                bytes_ptr,  # 源地址 host
                                len(bytes_data),  # 源地址大小
                                self.ACL_MEMCPY_HOST_TO_DEVICE)  # 模式:从host到device
            if ret != 0:
                print(f"拷贝输入数据到device失败: {ret}")
                copy_success = False
                break

        inference_result = []

        if not copy_success:
            return inference_result

        # 执行模型推理。
        ret = acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset)
        if ret != 0:
            print(f"模型推理失败: {ret}")
            return inference_result

        # 处理模型推理的输出数据。
        inference_result = []
        for i, item in enumerate(self.output_data):
            buffer_host, ret = acl.rt.malloc_host(self.output_data[i]["size"])
            if ret != 0:
                print(f"分配host memory 失败: {ret}")
                continue

            # 将推理输出数据从Device传输到Host。
            ret = acl.rt.memcpy(buffer_host,  # 目标地址 host
                                self.output_data[i]["size"],  # 目标地址大小
                                self.output_data[i]["buffer"],  # 源地址 device
                                self.output_data[i]["size"],  # 源地址大小
                                self.ACL_MEMCPY_DEVICE_TO_HOST)  # 模式：从device到host
            if ret != 0:
                print(f"拷贝数据到host失败: {ret}")
                continue

            # 从内存地址获取bytes对象
            bytes_out = acl.util.ptr_to_bytes(buffer_host, self.output_data[i]["size"])
            if len(bytes_out) != self.output_data[i]["size"]:
                print(f"输入数据的大小({len(bytes_out)})和模型定义输出数据的大小({self.output_data[i]['size']})不匹配")
                continue
            # 按照设定的dtype格式将数据转为numpy数组
            data = np.frombuffer(bytes_out, dtype=self.out_type)
            inference_result.append(data)
            # 释放内存
            ret = acl.rt.free_host(buffer_host)
            if ret != 0:
                print(f"释放host内存失败: {ret}")
        # vals = np.array(inference_result).flatten()
        return inference_result

    def __del__(self):
        # 析构函数 按照初始化资源的相反顺序释放资源。
        # 销毁输入输出数据集
        for dataset in [self.input_data, self.output_data]:
            while dataset:
                item = dataset.pop()
                ret = acl.destroy_data_buffer(item["data"])  # 销毁buffer数据
                ret = acl.rt.free(item["buffer"])  # 释放buffer内存
        ret = acl.mdl.destroy_dataset(self.input_dataset)  # 销毁输入数据集
        ret = acl.mdl.destroy_dataset(self.output_dataset)  # 销毁输出数据集
        # 销毁模型描述
        ret = acl.mdl.destroy_desc(self.model_desc)
        # 卸载模型
        ret = acl.mdl.unload(self.model_id)
        # 释放device
        ret = acl.rt.reset_device(self.device_id)
        # acl去初始化
        ret = acl.finalize()
        return
    


class PDModel:
    def __init__(self, mode_path, output):
        self.row_anchor = [18, 36, 54, 72, 90, 108, 126, 144, 162, 180, 198, 216, 234, 252, 270, 288, 306, 324, 342, 360, 378,
                396, 414, 432, 450, 468, 486, 504, 522, 540, 558, 576, 594, 612, 630, 648, 666, 684, 702, 720, 738,
                756, 774, 792, 810, 828, 846, 864, 882, 900, 918, 936, 954, 972, 990, 1008, 1026, 1044, 1062]
        self.row_h = 1080   

        if mode_path is None:
            mode_path = '/home/ckp/pidai_bs1.om'
        self.net = OMNet(mode_path, device_id=0, out_type=np.float32)   
        self.output = output  
        return
    
    @staticmethod
    def softmax(x, axis=None):
        # 为了数值稳定性，减去最大值
        x_max = np.max(x, axis=axis, keepdims=True)
        x = x - x_max

        # 计算指数
        exp_x = np.exp(x)

        # 计算分母
        sum_exp = np.sum(exp_x, axis=axis, keepdims=True)

        # 返回结果
        return exp_x / sum_exp

    def to_pts(self, cls_group, img_shape, col_sample_step, in_width=800):
        h, w = img_shape
        anchors, nums = cls_group.shape
        pts = list()
        for i in range(nums):
            if np.sum(cls_group[:, i] != 0) < 2:
                continue
            for k in range(anchors):
                if cls_group[k, i] < 0:
                    continue

                p = (int(cls_group[k, i] * col_sample_step * w / in_width) - 1,
                    int(h * (self.row_anchor[k] / self.row_h)) - 1)
                pts.append(p)
        return pts


    def __call__(self, img_bgr):
        w = 640
        h = 384
        grid = 100
        # img_bgr = cv2.imread(file_name, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img_rgb, dsize=(w, h), interpolation=cv2.INTER_LINEAR)

        in_np = np.stack([img.astype(np.float32) / 255])
        in_np = in_np.transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        in_np = np.ascontiguousarray(in_np)

        logits = self.net.forward(in_np)
        if len(logits) == 0:
            print('输出为空')
            raise RuntimeError('输出为空')

        logits = np.array(logits, dtype=np.float32)
        logits = logits.reshape(1, 101, 59, 2)
        
        bs, cls_nums, anchors, nums = logits.shape

        # logits = torch.flip(logits, dims=[2])
        prob = self.softmax(logits[:, :-1, ...], axis=1)
        idx = np.arange(1, grid + 1)[np.newaxis, ...]
        idx = idx.repeat(bs, 1)
        idx = idx.reshape(bs, -1, 1, 1)
        loc = np.sum(prob * idx, axis=1)
        logits = np.argmax(logits, axis=1)
        loc[logits == grid] = 0

        col_sample = np.linspace(0, w - 1, grid)
        step = col_sample[1] - col_sample[0]

        pts = self.to_pts(loc[0], (h, w), step, w)
        for pt in pts:
            cv2.circle(img, pt, 5, (255, 0, 0), -1)

        if self.output is not None:
            cv2.imwrite(f'{self.output}/lane.jpg', np.ascontiguousarray(img[:, :, ::-1]))
        return img[:, :, ::-1]



app = FastAPI(title="推理服务")
pd_model = PDModel(None, None)


def base2numpy(base):
	"""
	@param base:浏览器可识别的base64编码格式
	"""
	img_bin= base64.b64decode(base.split(';base64,')[-1])  
	img_buff = np.frombuffer(img_bin, dtype='uint8')
	image = cv2.imdecode(img_buff,1)
	return image


def numpy2base(image):
	"""
	@param image:numpy数组格式图片
	"""
	success, encoded_image = cv2.imencode(".jpg", image)
	byte_data = encoded_image.tobytes()
	base = "data:image/jpg;base64," + base64.b64encode(byte_data).decode('utf-8')
	return base



@app.post("/infer", summary="返回推理结果")
async def infer(buffer: str = Body(..., embed=True)):
    arr = base2numpy(buffer)
    arr = pd_model(arr)
    return {'result': numpy2base(arr)}



if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
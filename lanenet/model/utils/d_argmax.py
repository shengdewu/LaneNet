import torch


def differentiable_argmax(x, dim=0, temperature=1e-2):
    """
    可导的“软索引”计算，近似argmax的行为

    参数:
        x: 输入张量
        dim: 计算最大值的维度
        temperature: 温度参数，越小越接近argmax
    返回:
        软索引（连续值，可导）
    """
    # 生成概率分布（温度越低，概率越集中在最大值位置）
    softmax_weights = torch.softmax(x / temperature, dim=dim)

    # 生成索引序列（如0,1,2,...,C-1）
    indices = torch.arange(x.size(dim), device=x.device, dtype=x.dtype)

    # 将索引广播到与x匹配的形状
    indices = indices.view([1, -1] + [1] * (x.ndim - 2))  # 调整维度以便广播
    indices = indices.expand_as(x)

    # 概率加权求和得到软索引（可导）
    soft_index = torch.sum(softmax_weights * indices, dim=dim)
    return soft_index


def argmax_approx(x, temperature=0.01, dim=-1):
    """
    代替argmax, 输出类似one-hot
    :param x:
    :param temperature:
    :param dim
    :return:
    """
    # 温度越低，越接近argmax的one-hot分布
    prob = torch.softmax(x / temperature, dim=dim)
    return prob  # 返回概率分布，而非索引


def gumbel_softmax(logits, temperature=0.01, hard=False, dim=-1):
    """Gumbel-Softmax实现"""
    n, c, h, w = logits.shape
    # 展平空间维度以便处理（保持类别维度dim=1）
    logits_flat = logits.flatten(2)  # [N, C, H*W]

    # 生成Gumbel噪声（形状与logits_flat匹配）
    gumbel = -torch.log(-torch.log(torch.rand_like(logits_flat) + 1e-10) + 1e-10)

    # 加噪并计算softmax
    y = torch.softmax((logits_flat + gumbel) / temperature, dim=dim)

    # 若需要硬选择（模拟argmax的one-hot）
    if hard:
        # 找到概率最大的类别索引
        max_idx = y.argmax(dim=dim, keepdim=True)  # [N, 1, H*W]
        # 生成one-hot编码（硬选择）
        y_hard = torch.zeros_like(y).scatter_(1, max_idx, 1.0)
        # 关键：用detach()保留梯度传播路径
        y = (y_hard - y).detach() + y

    # 恢复空间维度
    return y.view(n, c, h, w)


def ce_loss(logits, labels):
    """
    边缘检测的类别必定是大于2
    :param logits: n, c, h, w
    :param labels: n, 1, h, w
    :return:
    """
    prob = torch.softmax(logits, dim=1)
    prob = torch.log(prob)
    index = - torch.gather(prob, dim=1, index=labels)
    return index.mean()

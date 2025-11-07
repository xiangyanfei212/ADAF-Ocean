import torch
import torch.nn as nn
from icecream import ic

class MaskedMSELoss(nn.Module):
    """
    自定义 MSE 损失函数，支持忽略 label 中的 NaN 值，并根据额外的 mask 将特定位置设为 0。
    """
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, prediction, label, mask):
        """
        计算损失。

        Args:
            prediction (torch.Tensor): 模型的预测值，形状为 (batch_size, num_context, channel_num)。
            label (torch.Tensor): 实际值（ground truth），形状与 prediction 一致，可能包含 NaN。
            mask (torch.Tensor): 布尔掩码，形状为 (batch_size, num_context, 1)，用于将 prediction 和 label 的相应位置设置为 0。

        Returns:
            torch.Tensor: 损失值（标量）。
        """
        # ic(mask.shape, prediction.shape, label.shape)
        # 将 mask 的维度扩展到与 prediction 和 label 一致
        mask = mask.expand_as(prediction)  # 将 (batch_size, 1) 扩展为 (batch_size, channel_num)
        mask = mask.bool()  # 将 mask 转换为布尔类型

        # 使用 mask 将 prediction 和 label 的相应位置设置为 0
        prediction = prediction * (~mask)  # mask 为 True 的位置设为 0
        label = label * (~mask)

        # 创建布尔掩码，忽略 label 中为 NaN 的位置
        valid_mask = ~torch.isnan(label)  # valid_mask 为 True 表示有效值的位置

        # 仅对有效值计算损失
        diff = prediction[valid_mask] - label[valid_mask]  # 计算有效值的差异
        loss = torch.mean(diff**2)  # 均方误差

        return loss

        
if __name__ == '__main__':
    # 创建随机测试数据
    batch_size, seq_len, feature_dim = 100, 2400, 5

    prediction = torch.rand(batch_size, seq_len, feature_dim, requires_grad=True)  # 预测值
    label = torch.rand(batch_size, seq_len, feature_dim)  # 标签值
    label[0, 0, 0] = float("nan")  # 模拟 NaN 值
    mask = torch.randint(0, 2, (batch_size, seq_len, 1))  # 随机生成掩码
    ic(mask)

    # 初始化损失函数
    criterion = MaskedMSELoss()

    # 计算损失
    loss = criterion(prediction, label, mask)

    print("Loss:", loss.item())
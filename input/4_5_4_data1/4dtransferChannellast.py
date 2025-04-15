import numpy as np
import os

def convert_to_channel_last(data, n, c, h, w, align_to=16):
    """
    将 NCHW 格式的数据转换为 NHWC 格式，并按 `align_to` 通道对齐存储。
    :param data: 输入数据，形状为 (n, c, h, w)
    :param n: 批量大小 (batch size)
    :param c: 通道数
    :param h: 高度
    :param w: 宽度
    :param align_to: 按多少通道对齐存储
    :return: 分组文件列表
    """
    if data.shape != (n, c, h, w):
        raise ValueError(f"输入数据的形状 {data.shape} 与指定的 (n, c, h, w) = ({n}, {c}, {h}, {w}) 不匹配")
    
    # 计算需要补零的通道数
    padding_channels = (align_to - c % align_to) % align_to
    
    # 在通道维度 (axis=1) 上补零
    padded_data = np.pad(data, ((0, 0), (0, padding_channels), (0, 0), (0, 0)), mode='constant')
    
    # 将通道维度移动到最后 (NCHW -> NHWC)
    channel_last_data = np.transpose(padded_data, (0, 2, 3, 1))
    
    # 计算组数
    total_channels = c + padding_channels
    num_groups = total_channels // align_to
    
    group_files = []
    for group in range(num_groups):
        start = group * align_to
        end = start + align_to
        group_data = channel_last_data[:, :, :, start:end]
        
        # 存储数据
        output_path = f"output_group_{group + 1}.txt"
        np.savetxt(output_path, group_data.flatten(), fmt="%.6f")
        group_files.append(output_path)
        print(f"第 {group + 1} 组数据已保存到 {output_path}")
    
    return group_files

def merge_group_files(group_files, output_path):
    """
    合并所有分组文件为一个文件。
    """
    merged_data = []
    for file in group_files:
        data = np.loadtxt(file)
        merged_data.extend(data)
    
    np.savetxt(output_path, merged_data, fmt="%.6f")
    print(f"所有分组文件已合并到 {output_path}")

def main(data_path, n, c, h, w, output_path, align_to):
    """
    主函数：读取数据并处理。
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"输入文件 {data_path} 不存在")
    
    data = np.loadtxt(data_path)  # 读取 1D 数据
    
    if data.size == n * c * h * w:
        data = data.reshape(n, c, h, w)
    else:
        raise ValueError(f"数据总量 {data.size} 与 (n * c * h * w) = {n * c * h * w} 不匹配")
    
    group_files = convert_to_channel_last(data, n, c, h, w, align_to)
    merge_group_files(group_files, output_path)

if __name__ == "__main__":
    data_path = 'output_data.txt'
    n = 1  # 批量大小
    c = 48  # 通道数
    h = 48  # 高度
    w = 1   # 宽度
    align_to = 16
    output_path = "output_data-channellast16.txt"
    
    main(data_path, n, c, h, w, output_path, align_to)

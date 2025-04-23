import numpy as np

def chw_to_hwc(file_path, channels, height, width):
    # 读取txt文件中的所有数据
    with open(file_path, 'r') as file:
        data = file.readlines()
    
    # 过滤掉非数值行
    filtered_data = []
    for line in data:
        try:
            # 尝试将每行转换为浮点数，如果失败则跳过
            filtered_data.append(float(line.strip()))
        except ValueError:
            continue
    
    # 检查数据大小是否匹配
    if len(filtered_data) != channels * height * width:
        raise ValueError("数据大小不匹配指定的通道、高度和宽度。")
    
    # 转换为NumPy数组并重塑为CHW格式
    chw_array = np.array(filtered_data).reshape((channels, height, width))
    
    # 转换为HWC格式
    hwc_array = np.transpose(chw_array, (1, 2, 0))
    
    return hwc_array

def save_hwc_to_txt(hwc_array, output_file_path):
    # 以HWC顺序展平并保存到txt文件
    hwc_flat = hwc_array.flatten(order='C')  # 默认按C顺序展平
    
    with open(output_file_path, 'w') as file:
        for value in hwc_flat:
            file.write(f"{value}\n")

# 示例用法
input_file_path = 'input_data.txt'  # 输入txt文件路径
output_file_path = 'input_data_hwc.txt'  # 输出txt文件路径
channels = 48  # 通道数
height = 4  # 图像高度
width = 4   # 图像宽度

# CHW转HWC
hwc_image = chw_to_hwc(input_file_path, channels, height, width)

# 保存HWC格式数据到新的txt文件（以HWC顺序展平保存）
save_hwc_to_txt(hwc_image, output_file_path)

print(f"转换后的HWC数据已保存到 {output_file_path}")

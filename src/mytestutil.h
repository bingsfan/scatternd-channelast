#ifndef MYTESTUTIL_H
#define MYTESTUTIL_H

#include "mat.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>
using namespace std;

extern std::map<std::string, double> convert_time_accumulator;


extern int test_scatterND3x2x1(vector<int> shape1, vector<int> shape2, vector<int> shape3,
							   const char *file_path, const char *indices_path,
							   const char *updates_path, const char *output_path, int op);

int test_scatterND3x2x1channellast(vector<int> shape1, vector<int> shape2, vector<int> shape3,
								   const char *file_path, const char *indices_path,
								   const char *updates_path, const char *output_path, int op,
								   int align_to);

extern int test_scatterND3x4x3channellast(vector<int> shape1, vector<int> shape2,
										  vector<int> shape3, const char *file_path,
										  const char *indices_path, const char *updates_path,
										  const char *output_path, int op, int align_to);
extern int test_scatterND4x5x4channellast(vector<int> shape1, vector<int> shape2,
										  vector<int> shape3, const char *file_path,
										  const char *indices_path, const char *updates_path,
										  const char *output_path, int op, int align_to);
extern int test_scatterND3x3x3(vector<int> shape1, vector<int> shape2, vector<int> shape3,
							   const char *file_path, const char *indices_path,
							   const char *updates_path, const char *output_path, int op);
extern int test_scatterND3x3x3channellast(vector<int> shape1, vector<int> shape2,
										  vector<int> shape3, const char *file_path,
										  const char *indices_path, const char *updates_path,
										  const char *output_path, int op, int align_to);
extern int test_scatterND_transfer(vector<int> shape1, vector<int> shape2, vector<int> shape3,
								   const char *file_path, const char *indices_path,
								   const char *updates_path, const char *output_path, int op,
								   int align_to);
static void matToFloatArray4d(float *output, const ncnn::Mat &m) {
	if(output == nullptr) {
		printf("output pointer is null\n");
		return;
	}

	int index = 0;
	for(int q = 0; q < m.c; q++) {
		const float *ptr = m.channel(q);
		for(int z = 0; z < m.d; z++) {
			for(int y = 0; y < m.h; y++) {
				for(int x = 0; x < m.w; x++) {
					output[index++] = ptr[x];
				}
				ptr += m.w;
			}
		}
	}

	// printf("Data copied to float array\n");
}
/**
 * @description: 给scatternd三维索引用的，将第一列的数据转移到最后一列去，和input匹配
 * @param {int} *indices
 * @param {int} rows
 * @param {int} cols
 * @return {*}
 */
static void rotateColumns(int *indices, int rows, int cols) {
	for(int i = 0; i < rows; ++i) {
		std::rotate(indices + i * cols, indices + i * cols + 1, indices + (i + 1) * cols);
	}
}
/**
 * @description: 给scatternd四维索引用的，将第二列的数据转移到最后一列去，和input匹配
 * @param {int} *array
 * @param {int} rows
 * @param {int} cols
 * @return {*}
 */
static void moveSecondColumnToLast(int *array, int rows, int cols) {
	for(int i = 0; i < rows; ++i) {
		std::rotate(array + i * cols + 1, array + i * cols + 2, array + (i + 1) * cols);
	}
}

/**
 * @description: 带索引的mat打印函数,scatternd调试用的
 * @param {char} *prefix
 * @param {int} index
 * @param {Mat} &m
 * @return {*}
 */
static void printMatWithIndex(const char *prefix, int index, const ncnn::Mat &m) {
	char filename[256];
	// 生成带索引的文件名，如 input_0.txt, index_0.txt
	snprintf(filename, sizeof(filename), "%s_%d.txt", prefix, index);

	FILE *fp = fopen(filename, "w");
	if(!fp) {
		fprintf(stderr, "无法打开文件 %s\n", filename);
		return;
	}

	// 写入数据
	for(int q = 0; q < m.c; ++q) {
		const float *ptr = m.channel(q);
		for(int y = 0; y < m.h; ++y) {
			for(int x = 0; x < m.w; ++x) {
				fprintf(fp, "%.6f\n", ptr[x]);
			}
			ptr += m.w;
		}
	}

	fclose(fp);
	printf("数据已写入 %s\n", filename);
}
static void writeAllMats4dToFile(const char *path, const std::vector<ncnn::Mat> &result) {
	FILE *fp = fopen(path, "w");
	if(fp == nullptr) {
		printf("无法打开文件 %s\n", path);
		return;
	}

	for(size_t i = 0; i < result.size(); i++) {
		const ncnn::Mat &m = result[i];
		for(int q = 0; q < m.c; q++) {
			const float *ptr = m.channel(q);
			for(int z = 0; z < m.d; z++)	// 新增d维度循环
			{
				for(int y = 0; y < m.h; y++) {
					for(int x = 0; x < m.w; x++) {
						fprintf(fp, "%.6f\n", ptr[x]);
					}
					ptr += m.w;	   // 按行步进指针
				}
			}
		}
	}

	fclose(fp);
	// printf("所有数据已写入 %s\n", path);
}
static void writeAllMatsToFile(const char *path,
							   const std::vector<ncnn::Mat> &result) {	  // smh
	FILE *fp = fopen(path, "w");	// 只打开一次文件
	if(fp == nullptr) {
		printf("无法打开文件 %s\n", path);
		return;
	}

	// 遍历所有 result 元素
	for(size_t i = 0; i < result.size(); i++) {
		const ncnn::Mat &m = result[i];

		// 添加区块分隔标记

		// 写入数据
		for(int q = 0; q < m.c; q++) {
			const float *ptr = m.channel(q);
			for(int y = 0; y < m.h; y++) {
				for(int x = 0; x < m.w; x++) {
					fprintf(fp, "%.6f\n", ptr[x]);
				}
				// fprintf(fp, "\n"); // 行尾换行
				ptr += m.w;
			}
			// fprintf(fp, "\n"); // 通道间空行
		}
	}

	fclose(fp);
	// printf("所有数据已写入 %s\n", path);
}
// 修改后的文件写入函数（添加索引后缀）
static void matToFileWithIndex(const char *base_path, int index, const ncnn::Mat &m) {
	char path[256];
	// 生成带索引的唯一文件名，例如 output_0.txt, output_1.txt
	snprintf(path, sizeof(path), "%s_%d.txt", base_path, index);

	FILE *fp = fopen(path, "w");
	if(fp == nullptr) {
		printf("无法打开文件 %s\n", path);
		return;
	}

	// 写入维度信息（可选）
	// fprintf(fp, "Channels: %d\nHeight: %d\nWidth: %d\n", m.c, m.h, m.w);

	// 写入数据
	for(int q = 0; q < m.c; q++) {
		const float *ptr = m.channel(q);
		for(int y = 0; y < m.h; y++) {
			for(int x = 0; x < m.w; x++) {
				fprintf(fp, "%.6f\n", ptr[x]);	  // 用空格分隔数据点
			}
			ptr += m.w;
		}
	}

	fclose(fp);
	printf("数据已写入 %s\n", path);
}

static float *convert_3d_block_to_hwc(float *input_data, size_t H, size_t W, size_t C,
									  size_t align) {
	// 计算 block_num
	if(C % align != 0) {
		cerr << "通道数 " << C << " 不是 " << align << " 的整数倍，数据格式错误！" << endl;
		return nullptr;
	}
	size_t block_num = C / align;

	// 申请输出数据的空间
	float *output_data = (float *)malloc(H * W * C * sizeof(float));
	if(!output_data) {
		cerr << "内存分配失败！" << endl;
		return nullptr;
	}

	// 遍历输入数据 (block_num, H, W, align) 并重新排列到 (H, W, C)
	for(size_t b = 0; b < block_num; ++b) {
		for(size_t h = 0; h < H; ++h) {
			for(size_t w = 0; w < W; ++w) {
				for(size_t c = 0; c < align; ++c) {
					size_t input_index = b * (H * W * align) + h * (W * align) + w * align + c;
					size_t output_index = h * (W * C) + w * C + (b * align + c);

					output_data[output_index] = input_data[input_index];
				}
			}
		}
	}
	return output_data;
}
/**
 * 将四维块格式数据转换为NHWC格式
 * @param input_data 输入数据指针 (block_num, N, H, W, align)
 * @param N batch大小
 * @param H 高度
 * @param W 宽度
 * @param C 通道数
 * @param align 对齐大小
 * @return 转换后的NHWC格式数据指针
 */
static float *convert_4d_block_to_nhwc(float *input_data, size_t N, size_t H, size_t W,
									   size_t C, size_t align) {
	if(C % align != 0) {
		cerr << "通道数 " << C << " 不是 " << align << " 的整数倍，数据格式错误！" << endl;
		return nullptr;
	}
	size_t block_num = C / align;

	float *output_data = (float *)malloc(N * H * W * C * sizeof(float));
	if(!output_data) {
		cerr << "内存分配失败！" << endl;
		return nullptr;
	}

	// 输入数据布局: (block_num, N, H, W, align)
	// 输出数据布局: (N, H, W, C)
	for(size_t b = 0; b < block_num; ++b) {
		for(size_t n = 0; n < N; ++n) {
			for(size_t h = 0; h < H; ++h) {
				for(size_t w = 0; w < W; ++w) {
					for(size_t c = 0; c < align; ++c) {
						size_t input_index = b * (N * H * W * align) + n * (H * W * align)
						 + h * (W * align) + w * align + c;
						size_t output_index =
						 n * (H * W * C) + h * (W * C) + w * C + (b * align + c);

						output_data[output_index] = input_data[input_index];
					}
				}
			}
		}
	}
	return output_data;
}
/**
 * 将三维HWC格式数据转换为块格式并写入文件
 * @param input_data 输入数据指针 (H, W, C)
 * @param output_file_path 输出文件路径
 * @param H 高度
 * @param W 宽度
 * @param C 通道数
 * @param align 对齐大小
 */
static void convert_hwc_to_block_3d(const float *input_data, const char *output_file_path,
									size_t H, size_t W, size_t C, size_t align) {
	// 打开输出文件
	std::ofstream output_file(output_file_path);
	if(!output_file.is_open()) {
		std::cerr << "Error: Failed to open output file: " << output_file_path << std::endl;
		return;
	}

	// 设置输出浮点数格式：至少六位小数
	output_file << std::fixed << std::setprecision(6);

	// 计算块的数量
	size_t num_blocks	  = (C + align - 1) / align;	// 向上取整
	size_t total_elements = H * W * C;

	// 按块处理数据
	for(size_t block = 0; block < num_blocks; ++block) {
		size_t start_c = block * align;					  // 当前块的起始通道
		size_t end_c   = std::min(start_c + align, C);	  // 当前块的结束通道

		// 先处理当前块中的所有空间位置(H×W)的有效数据
		for(size_t h = 0; h < H; ++h) {
			for(size_t w = 0; w < W; ++w) {
				for(size_t c = start_c; c < end_c; ++c) {
					size_t hwc_index = h * W * C + w * C + c;
					if(hwc_index >= total_elements) {
						std::cerr << "Error: Index out of bounds! hwc_index=" << hwc_index
								  << " >= total_elements=" << total_elements << std::endl;
						output_file.close();
						return;
					}
					output_file << input_data[hwc_index] << "\n";
				}

				// 如果当前块不足align个通道，补充零
				for(size_t c = end_c; c < start_c + align; ++c) {
					output_file << "0.000000\n";
				}
			}
		}
	}

	output_file.close();
}
/**
 * 将四维NHWC格式数据转换为块格式并写入文件
 * @param input_data 输入数据指针 (N, H, W, C)
 * @param output_file_path 输出文件路径
 * @param N batch大小
 * @param H 高度
 * @param W 宽度
 * @param C 通道数
 * @param align 对齐大小
 */
static void convert_nhwc_to_block_4d(const float *input_data, const char *output_file_path,
									 size_t N, size_t H, size_t W, size_t C, size_t align) {
	// 打开输出文件
	std::ofstream output_file(output_file_path);
	if(!output_file.is_open()) {
		std::cerr << "Error: Failed to open output file: " << output_file_path << std::endl;
		return;
	}

	// 设置输出浮点数格式
	output_file << std::fixed << std::setprecision(6);

	// 计算块的数量和总元素数
	size_t num_blocks	  = (C + align - 1) / align;
	size_t total_elements = N * H * W * C;

	// 按块处理数据
	for(size_t block = 0; block < num_blocks; ++block) {
		size_t start_c = block * align;
		size_t end_c   = std::min(start_c + align, C);

		// 处理当前块的有效数据
		for(size_t n = 0; n < N; ++n) {
			for(size_t h = 0; h < H; ++h) {
				for(size_t w = 0; w < W; ++w) {
					for(size_t c = start_c; c < end_c; ++c) {
						size_t nhwc_index = n * H * W * C + h * W * C + w * C + c;
						if(nhwc_index >= total_elements) {
							std::cerr
							 << "Error: Index out of bounds! nhwc_index=" << nhwc_index
							 << " >= total_elements=" << total_elements << std::endl;
							output_file.close();
							return;
						}
						output_file << input_data[nhwc_index] << "\n";
					}

					// 补充零
					for(size_t c = end_c; c < start_c + align; ++c) {
						output_file << "0.000000\n";
					}
				}
			}
		}
	}

	output_file.close();
}
#endif
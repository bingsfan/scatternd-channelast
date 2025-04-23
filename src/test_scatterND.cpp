// Tencent is pleased to support the open source community by making ncnn
// available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use
// this file except in compliance with the License. You may obtain a copy of
// the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.

#include "mytestutil.h"
#include "op.h"
#include "testutil.h"
#include <chrono>

int test_scatterND3x2x1(vector<int> shape1, vector<int> shape2, vector<int> shape3,
						const char *file_path, const char *indices_path,
						const char *updates_path, const char *output_path, int op) {
	printf("=====   test_scatterND3x2x1 start   ======\n");
	int ret;
	int w1 = shape1[2], h1 = shape1[1], c1 = shape1[0];
	int w2 = shape2[1], h2 = shape2[0];
	int w3			 = shape3[0];
	int input_size	 = w1 * h1 * c1;
	int indices_size = w2 * h2;
	int update_size	 = w3;
	float *data		 = readFile(file_path, input_size);
	int *indices	 = readFileINT(indices_path, indices_size);
	float *updates	 = readFile(updates_path, update_size);
	ncnn::Mat input	 = InitMat3D_float(w1, h1, c1, data);
	ncnn::Mat index	 = InitMat2D_int(w2, h2, indices);
	ncnn::Mat update = InitMat1D_float(w3, updates);
	ncnn::Mat result;
	scatterND(result, input, index, update, op);
	matToFile(output_path, result);
	printf("====    test_scatterND3x2x1 end    =======\n\n");
	return 0;
}
int test_scatterND_transfer(vector<int> shape1, vector<int> shape2, vector<int> shape3,
							const char *file_path, const char *indices_path,
							const char *updates_path, const char *output_path, int op,
							int align_to) {
	// printf("=====   test_scatterND3x2x1 start   ======\n");
	int ret;
	int input_size = 1;
	for(auto i : shape1) {
		input_size *= i;
	}
	int indices_size = 1;
	for(auto i : shape2) {
		indices_size *= i;
	}
	int update_size = 1;
	for(auto i : shape3) {
		update_size *= i;
	}
	// 先得到三个数组的总长度

	int *indices   = readFileINT(indices_path, indices_size);
	float *updates = readFile(updates_path, update_size);
	float *hwc_data;
	size_t n, t, d, h, w, c;
	int input_dims	 = shape1.size();
	int indices_dims = shape2.size();
	int updates_dims = shape3.size();
	// 测读取数据和转换的时间，不然转换时间太快了看不到
	auto start1 = chrono::high_resolution_clock::now();
	float *data = readFile(file_path, input_size);
	if(input_dims == 3) {
		h = shape1[0], w = shape1[1], c = shape1[2];
		hwc_data = convert_3d_block_to_hwc(data, h, w, c, align_to);
	} else if(input_dims == 4) {
		d = shape1[0], h = shape1[1], w = shape1[2], c = shape1[3];
		hwc_data = convert_4d_block_to_nhwc(data, d, h, w, c, align_to);
	}
	auto end1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration = end1 - start1;
	convert_time_accumulator["read_and_convert_block_to_hwc"] += duration.count();
	int w1, h1, c1, b1;
	int w2, h2, c2, b2, d2;
	int w3, h3, c3, b3;
	ncnn::Mat input;
	ncnn::Mat index;
	ncnn::Mat update;
	ncnn::Mat result;
	if(input_dims == 3 && indices_dims == 2 && updates_dims == 1) {
		c1 = shape1[0], h1 = shape1[1], w1 = shape1[2];
		h2 = shape2[0], w2 = shape2[1];
		w3 = shape3[0];
		rotateColumns(indices, h2, w2);	   // 添加的更换索引的轴，将0,1,2换成1,2,0
		vector<float> vec_index(indices, indices + indices_size);
		input  = InitMat3D_float(w1, h1, c1, hwc_data);
		index  = InitMat2D_int(w2, h2, indices);
		update = InitMat1D_float(w3, updates);
	} else if(input_dims == 3 && indices_dims == 4 && updates_dims == 3) {
		c1 = shape1[0], h1 = shape1[1], w1 = shape1[2];
		b2 = shape2[0], c2 = shape2[1], h2 = shape2[2], w2 = shape2[3];
		c3 = shape3[0], h3 = shape3[1], w3 = shape3[2];
		rotateColumns(indices, h2 * c2 * b2, w2);
		input  = InitMat3D_float(w1, h1, c1, hwc_data);
		index  = InitMat2D_int(3, update_size, indices);
		update = InitMat1D_float(update_size, updates);
	} else if(input_dims == 4 && indices_dims == 5 && updates_dims == 4) {
		b1 = shape1[0], c1 = shape1[1], h1 = shape1[2], w1 = shape1[3];
		w2 = shape2[4], h2 = shape2[3], c2 = shape2[2], b2 = shape2[1], d2 = shape2[0];
		c3 = shape3[0], h3 = shape3[1], w3 = shape3[2];
		moveSecondColumnToLast(indices, h2 * c2 * b2 * d2, w2);
		input  = InitMat4D_float(b1, c1, h1, w1, hwc_data);
		index  = InitMat2D_int(4, update_size, indices);
		update = InitMat1D_float(update_size, updates);
	} else if(input_dims == 3 && indices_dims == 3 && updates_dims == 3) {
		c1 = shape1[0], h1 = shape1[1], w1 = shape1[2];
		c2 = shape2[0], h2 = shape2[1], w2 = shape2[2];
		c3 = shape3[0], h3 = shape3[1], w3 = shape3[2];
		// 转变索引，将索引变成四维的，调用3x4x3
		vector<int> new_indices(c2 * h2 * w3 * (w2 + 1));
		for(int i = 0; i < c2; ++i) {
			for(int j = 0; j < h2; ++j) {
				for(int k = 0; k < w3; ++k) {
					for(int l = 0; l < w2 + 1; ++l) {
						if(l < w2) {
							new_indices[((i * h2 + j) * w3 + k) * (w2 + 1) + l] =
							 indices[(i * h2 + j) * w2 + l];
						} else {
							new_indices[((i * h2 + j) * w3 + k) * (w2 + 1) + l] =
							 k;	   // 新加的那一列的值是0->w3-1
						}
					}
				}
			}
		}
		rotateColumns(new_indices.data(), c2 * h2 * w3, w2 + 1);

		input  = InitMat3D_float(w1, h1, c1, hwc_data);
		index  = InitMat2D_int(3, update_size, new_indices.data());
		update = InitMat1D_float(update_size, updates);
	} else {
		printf("input_dims:%d indices_dims:%d updates_dims:%d\n", input_dims, indices_dims,
			   updates_dims);
	}
	scatterND(result, input, index, update, op);
	float *result_data = new float[input_size];
	matToFloatArray4d(result_data, result);
	// 然后将result_data转换成block格式
	auto start2 = std::chrono::high_resolution_clock::now();
	if(input_dims == 3) {
		convert_hwc_to_block_3d(result_data, output_path, c1, h1, w1, align_to);
	} else if(input_dims == 4) {
		convert_nhwc_to_block_4d(result_data, output_path, b1, c1, h1, w1, align_to);
	}
	auto end2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration2 = end2 - start2;
	convert_time_accumulator["print_and_convert_4d_block_to_hwc"] += duration2.count();
	// printf("====    test_scatterND3x2x1 end    =======\n\n");
	return 0;
}
/**
 * @description:
 * @param {vector<int>} shape1  data的shape
 * @param {vector<int>} shape2	indices的shape
 * @param {vector<int>} shape3	updates的shape
 * @param {char} *file_path		data的路径
 * @param {char} *indices_path	indices的路径
 * @param {char} *updates_path	updates的路径
 * @param {char} *output_path	输出的路径
 * @param {int} op				可以取0，1，2，3，4，updates的功能不同
 * @param {int} align_to		通道对齐数
 * @return {*}
 */
int test_scatterND3x2x1channellast(vector<int> shape1, vector<int> shape2, vector<int> shape3,
								   const char *file_path, const char *indices_path,
								   const char *updates_path, const char *output_path, int op,
								   int align_to) {
	// printf("=====   test_scatterND3x2x1 start   ======\n");
	int ret;
	int w1 = shape1[2], h1 = shape1[1], c1 = shape1[0];
	int w2 = shape2[1], h2 = shape2[0];
	int w3			 = shape3[0];
	int input_size	 = w1 * h1 * c1;
	int indices_size = w2 * h2;
	int update_size	 = w3;

	int block_size = input_size / shape1[shape1.size() - 1] * align_to;
	int blockNums  = input_size / block_size;

	vector<vector<float>> newinput(blockNums);
	vector<vector<int>> newIndices(blockNums);
	vector<vector<float>> newUpdates(blockNums);

	vector<size_t> block_shape(shape1.size());
	for(int i = 0; i < shape1.size() - 1; i++) {
		block_shape[i] = shape1[i];
	}
	block_shape[shape1.size() - 1] = align_to;

	auto start1 = chrono::high_resolution_clock::now();
	float *data = readFile(file_path, input_size);
	auto end1	= std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration = end1 - start1;
	convert_time_accumulator["read_block_time"] += duration.count();

	int *indices   = readFileINT(indices_path, indices_size);
	float *updates = readFile(updates_path, update_size);
	rotateColumns(indices, h2, w2);	   // 添加的更换索引的轴，将0,1,2换成1,2,0
	for(int i = 0; i < blockNums; i++) {
		newinput[i] = vector<float>(data + i * block_size, data + (i + 1) * block_size);
	}
	vector<int> vec_index(indices, indices + indices_size);
	int k = 0;
	// 写一个方法将blocknums块的索引分开,input,index，updates都得分开
	// input在读取的时候按块分开了，index和updates需要手动计算出来
	// 将每一块的c和updates放到不同的数组中去
	for(int i = 2; i < indices_size; i += 3) {
		int j = indices[i] / align_to;
		newIndices[j].push_back(indices[i - 2]);
		newIndices[j].push_back(indices[i - 1]);
		newIndices[j].push_back(indices[i] % align_to);
		newUpdates[j].push_back(updates[k++]);
	}
	ncnn::Mat input;
	ncnn::Mat index;
	ncnn::Mat update;
	vector<ncnn::Mat> result(blockNums);
	for(int j = 0; j < blockNums; j++) {
		input  = InitMat3D_float(align_to, h1, c1, newinput[j].data());
		index  = InitMat2D_int(3, newUpdates[j].size(), newIndices[j].data());
		update = InitMat1D_float(newUpdates[j].size(), newUpdates[j].data());
		/* 	printMatWithIndex("input", j, input);	   // 生成 input_0.txt, input_1.txt...
			printMatWithIndex("index", j, index);	   // 生成 index_0.txt, index_1.txt...
			printMatWithIndex("update", j, update);	   // 生成 update_0.txt, update_1.txt... */
		scatterND(result[j], input, index, update, op);
		// matToFileWithIndex(output_path, j, result[j]);
	}
	auto start2 = chrono::high_resolution_clock::now();
	writeAllMatsToFile(output_path, result);
	auto end2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration2 = end2 - start2;
	convert_time_accumulator["print_block_time"] += duration2.count();
	// printf("====    test_scatterND3x2x1 end    =======\n\n");
	return 0;
}
int test_scatterND3x3x3(vector<int> shape1, vector<int> shape2, vector<int> shape3,
						const char *file_path, const char *indices_path,
						const char *updates_path, const char *output_path, int op) {
	printf("=====   test_scatterND3x3x3 start   ======\n");
	int ret;
	int w1 = shape1[2], h1 = shape1[1], c1 = shape1[0];
	int w2 = shape2[2], h2 = shape2[1], c2 = shape2[0];
	int w3 = shape3[2], h3 = shape3[1], c3 = shape3[0];
	int input_size	 = w1 * h1 * c1;
	int indices_size = w2 * h2 * c2;
	int update_size	 = w3 * h3 * c3;
	float *data		 = readFile(file_path, input_size);
	int *indices	 = readFileINT(indices_path, indices_size);
	float *updates	 = readFile(updates_path, update_size);

	ncnn::Mat input	 = InitMat3D_float(w1, h1, c1, data);
	ncnn::Mat index	 = InitMat3D_int(w2, h2, c2, indices);
	ncnn::Mat update = InitMat3D_float(w3, h3, c3, updates);
	ncnn::Mat result;
	scatterND(result, input, index, update, op);
	matToFile(output_path, result);

	printf("====    test_scatterND3x3x3 end    =======\n\n");
	return 0;
}
int test_scatterND3x3x3channellast(vector<int> shape1, vector<int> shape2, vector<int> shape3,
								   const char *file_path, const char *indices_path,
								   const char *updates_path, const char *output_path, int op,
								   int align_to) {
	// printf("=====   test_scatterND3x3x3 start   ======\n");
	int ret;
	int w1 = shape1[2], h1 = shape1[1], c1 = shape1[0];
	int w2 = shape2[2], h2 = shape2[1], c2 = shape2[0];
	int w3 = shape3[2], h3 = shape3[1], c3 = shape3[0];
	int input_size	 = w1 * h1 * c1;
	int indices_size = w2 * h2 * c2;
	int update_size	 = w3 * h3 * c3;

	auto start1		 = chrono::high_resolution_clock::now();
	float *data		 = readFile(file_path, input_size);
	auto end1		 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration = end1 - start1;
	convert_time_accumulator["read_block_time"] += duration.count();

	int *indices	 = readFileINT(indices_path, indices_size);
	float *updates	 = readFile(updates_path, update_size);
	// 转变索引，将索引变成四维的，调用3x4x3
	vector<int> new_indices(c2 * h2 * w3 * (w2 + 1));
	for(int i = 0; i < c2; ++i) {
		for(int j = 0; j < h2; ++j) {
			for(int k = 0; k < w3; ++k) {
				for(int l = 0; l < w2 + 1; ++l) {
					if(l < w2) {
						new_indices[((i * h2 + j) * w3 + k) * (w2 + 1) + l] =
						 indices[(i * h2 + j) * w2 + l];
					} else {
						new_indices[((i * h2 + j) * w3 + k) * (w2 + 1) + l] =
						 k;	   // 新加的那一列的值是0->w3-1
					}
				}
			}
		}
	}
	rotateColumns(new_indices.data(), c2 * h2 * w3, w2 + 1);

	// 使用 new_indices 进行后续操作
	int block_size = input_size / shape1[shape1.size() - 1] * align_to;
	int blockNums  = input_size / block_size;

	vector<vector<float>> newinput(blockNums);
	vector<vector<int>> newIndices(blockNums);
	vector<vector<float>> newUpdates(blockNums);
	vector<size_t> block_shape(shape1.size());
	for(int i = 0; i < shape1.size() - 1; i++) {
		block_shape[i] = shape1[i];
	}
	block_shape[shape1.size() - 1] = align_to;
	for(int i = 0; i < blockNums; i++) {
		newinput[i] = vector<float>(data + i * block_size, data + (i + 1) * block_size);
	}

	int k = 0;
	// 将每一块的c和updates放到不同的数组中去
	for(int i = 2; i < new_indices.size(); i += 3) {
		int j = new_indices[i] / align_to;
		newIndices[j].push_back(new_indices[i - 2]);
		newIndices[j].push_back(new_indices[i - 1]);
		newIndices[j].push_back(new_indices[i] % align_to);
		newUpdates[j].push_back(updates[k++]);
	}

	ncnn::Mat input;
	ncnn::Mat index;
	ncnn::Mat update;
	vector<ncnn::Mat> result(blockNums);
	// 得写一个方法将blocknums块的索引分开,input,index，updates都得分开
	// input在读取的时候按块分开了，index和updates需要手动计算出来

	for(int j = 0; j < blockNums; j++) {
		input  = InitMat3D_float(align_to, h1, c1, newinput[j].data());
		index  = InitMat2D_int(3, newUpdates[j].size(), newIndices[j].data());
		update = InitMat1D_float(newUpdates[j].size(), newUpdates[j].data());
		// printMatWithIndex("input", j, input);   // 生成 input_0.txt, input_1.txt...
		// printMatWithIndex("index", j, index);   // 生成 index_0.txt, index_1.txt...
		// printMatWithIndex("update", j, update); // 生成 update_0.txt, update_1.txt...
		scatterND(result[j], input, index, update, op);
		// matToFileWithIndex(output_path, j, result[j]);
	}
	auto start2 = chrono::high_resolution_clock::now();
	writeAllMatsToFile(output_path, result);
	auto end2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration2 = end2 - start2;
	convert_time_accumulator["print_block_time"] += duration2.count();

	// printf("====    test_scatterND3x3x3 end    =======\n\n");
	return 0;
}
int test_scatterND3x4x3(vector<int> shape1, vector<int> shape2, vector<int> shape3,
						const char *file_path, const char *indices_path,
						const char *updates_path, const char *output_path, int op) {
	printf("=====   test_scatterND3x4x3 start   ======\n");
	int ret;
	int w1 = shape1[2], h1 = shape1[1], c1 = shape1[0];
	int w2 = shape2[3], h2 = shape2[2], c2 = shape2[1], b2 = shape2[0];
	int w3 = shape3[2], h3 = shape3[1], c3 = shape3[0];
	int input_size	 = w1 * h1 * c1;
	int indices_size = w2 * h2 * c2 * b2;
	int update_size	 = w3 * h3 * c3;
	float *data		 = readFile(file_path, input_size);
	int *indices	 = readFileINT(indices_path, indices_size);
	float *updates	 = readFile(updates_path, update_size);

	ncnn::Mat input	 = InitMat3D_float(w1, h1, c1, data);
	ncnn::Mat index	 = InitMat4D_int(w2, h2, c2, b2, indices);
	ncnn::Mat update = InitMat3D_float(w3, h3, c3, updates);
	ncnn::Mat result;
	scatterND(result, input, index, update, op);
	matToFile(output_path, result);

	printf("====    test_scatterND3x4x3 end    =======\n\n");
	return 0;
}
int test_scatterND3x4x3channellast(vector<int> shape1, vector<int> shape2, vector<int> shape3,
								   const char *file_path, const char *indices_path,
								   const char *updates_path, const char *output_path, int op,
								   int align_to) {
	// printf("=====   test_scatterND3x4x3 start   ======\n");
	int ret;
	int w1 = shape1[2], h1 = shape1[1], c1 = shape1[0];
	int w2 = shape2[3], h2 = shape2[2], c2 = shape2[1], b2 = shape2[0];
	int w3 = shape3[2], h3 = shape3[1], c3 = shape3[0];
	int input_size	 = w1 * h1 * c1;
	int indices_size = w2 * h2 * c2 * b2;
	int update_size	 = w3 * h3 * c3;

	auto start1 = chrono::high_resolution_clock::now();
	float *data = readFile(file_path, input_size);
	auto end1	= std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration = end1 - start1;
	convert_time_accumulator["read_block_time"] += duration.count();

	int *indices   = readFileINT(indices_path, indices_size);
	float *updates = readFile(updates_path, update_size);
	rotateColumns(indices, h2 * c2 * b2, w2);
	int block_size = input_size / shape1[shape1.size() - 1] * align_to;
	int blockNums  = input_size / block_size;
	vector<vector<float>> newinput(blockNums);
	vector<vector<int>> newIndices(blockNums);
	vector<vector<float>> newUpdates(blockNums);
	vector<size_t> block_shape(shape1.size());
	for(int i = 0; i < shape1.size() - 1; i++) {
		block_shape[i] = shape1[i];
	}
	block_shape[shape1.size() - 1] = align_to;
	for(int i = 0; i < blockNums; i++) {
		newinput[i] = vector<float>(data + i * block_size, data + (i + 1) * block_size);
	}
	// 处理indices和updates
	vector<int> vec_index(indices, indices + indices_size);
	int c_index = 0;
	int k		= 0;
	// 将每一块的c和updates放到不同的数组中去
	for(int i = 2; i < indices_size; i += 3) {
		int j = indices[i] / align_to;
		newIndices[j].push_back(indices[i - 2]);
		newIndices[j].push_back(indices[i - 1]);
		newIndices[j].push_back(indices[i] % align_to);
		newUpdates[j].push_back(updates[k++]);
	}

	ncnn::Mat input;
	ncnn::Mat index;
	ncnn::Mat update;
	vector<ncnn::Mat> result(blockNums);
	for(int j = 0; j < blockNums; j++) {
		input  = InitMat3D_float(align_to, h1, c1, newinput[j].data());
		index  = InitMat2D_int(3, newUpdates[j].size(), newIndices[j].data());
		update = InitMat1D_float(newUpdates[j].size(), newUpdates[j].data());
		scatterND(result[j], input, index, update, op);
	}
	auto start2 = chrono::high_resolution_clock::now();
	writeAllMatsToFile(output_path, result);
	auto end2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration2 = end2 - start2;
	convert_time_accumulator["print_block_time"] += duration2.count();
	// printf("====    test_scatterND3x4x3 end    =======\n\n");
	return 0;
}
int test_scatterND4x5x4(vector<int> shape1, vector<int> shape2, vector<int> shape3,
						const char *file_path, const char *indices_path,
						const char *updates_path, const char *output_path,
						int op) {	 // 目前只能处理indices是五维，但是第一维是1的情况
	printf("=====   test_scatterND4x5x4 start   ======\n");
	int ret;
	int w1 = shape1[3], h1 = shape1[2], c1 = shape1[1], b1 = shape1[0];
	int w2 = shape2[4], h2 = shape2[3], c2 = shape2[2], b2 = shape2[1], d2 = shape2[0];
	int w3 = shape3[3], h3 = shape3[2], c3 = shape3[1], b3 = shape3[0];
	int input_size	 = w1 * h1 * c1 * b1;
	int indices_size = w2 * h2 * c2 * b2 * d2;
	int update_size	 = w3 * h3 * c3 * b3;
	float *data		 = readFile(file_path, input_size);
	int *indices	 = readFileINT(indices_path, indices_size);
	float *updates	 = readFile(updates_path, update_size);

	ncnn::Mat input	 = InitMat4D_float(w1, h1, c1, b1, data);
	ncnn::Mat index	 = InitMat4D_int(w2, h2, c2, b2, indices);
	ncnn::Mat update = InitMat3D_float(w3, h3, c3, updates);
	ncnn::Mat result;
	scatterND(result, input, index, update, op);
	matToFIle4d(output_path, result);

	printf("====    test_scatterND4x5x4 end    =======\n\n");
	return 0;
}
int test_scatterND4x5x4channellast(vector<int> shape1, vector<int> shape2, vector<int> shape3,
								   const char *file_path, const char *indices_path,
								   const char *updates_path, const char *output_path, int op,
								   int align_to) {
	// printf("=====   test_scatterND4x5x4 start   ======\n");
	int ret;
	int w1 = shape1[3], h1 = shape1[2], c1 = shape1[1], b1 = shape1[0];
	int w2 = shape2[4], h2 = shape2[3], c2 = shape2[2], b2 = shape2[1], d2 = shape2[0];
	int w3 = shape3[3], h3 = shape3[2], c3 = shape3[1], b3 = shape3[0];
	int input_size	 = w1 * h1 * c1 * b1;
	int indices_size = w2 * h2 * c2 * b2 * d2;
	int update_size	 = w3 * h3 * c3 * b3;

	auto start1		 = chrono::high_resolution_clock::now();
	float *data		 = readFile(file_path, input_size);
	auto end1		 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration = end1 - start1;
	convert_time_accumulator["read_block_time"] += duration.count();

	int *indices	 = readFileINT(indices_path, indices_size);
	float *updates	 = readFile(updates_path, update_size);
	moveSecondColumnToLast(indices, h2 * c2 * b2 * d2, w2);

	int block_size = input_size / shape1[shape1.size() - 1] * align_to;
	int blockNums  = input_size / block_size;

	vector<vector<float>> newinput(blockNums);
	vector<vector<int>> newIndices(blockNums);
	vector<vector<float>> newUpdates(blockNums);
	vector<size_t> block_shape(shape1.size());
	for(int i = 0; i < shape1.size() - 1; i++) {
		block_shape[i] = shape1[i];
	}
	block_shape[shape1.size() - 1] = align_to;
	for(int i = 0; i < blockNums; i++) {
		newinput[i] = vector<float>(data + i * block_size, data + (i + 1) * block_size);
	}
	int k = 0;
	// 将每一块的c和updates放到不同的数组中去
	for(int i = 3; i < indices_size; i += 4) {
		int j = indices[i] / align_to;
		newIndices[j].push_back(indices[i - 3]);
		newIndices[j].push_back(indices[i - 2]);
		newIndices[j].push_back(indices[i - 1]);
		newIndices[j].push_back(indices[i] % align_to);
		newUpdates[j].push_back(updates[k++]);
	}

	ncnn::Mat input;
	ncnn::Mat index;
	ncnn::Mat update;
	vector<ncnn::Mat> result(blockNums);
	for(int j = 0; j < blockNums; j++) {
		input  = InitMat4D_float(align_to, h1, c1, b1, newinput[j].data());
		index  = InitMat2D_int(4, newUpdates[j].size(), newIndices[j].data());
		update = InitMat1D_float(newUpdates[j].size(), newUpdates[j].data());
		scatterND(result[j], input, index, update, op);
	}
	
	auto start2 = chrono::high_resolution_clock::now();
	writeAllMats4dToFile(output_path, result);
	auto end2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration2 = end2 - start2;
	convert_time_accumulator["print_block_time"] += duration2.count();
	// printf("====    test_scatterND4x5x4 end    =======\n\n");
	return 0;
}

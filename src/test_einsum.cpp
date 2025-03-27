// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

// #include "../src/layer/einsum.h"
#include "op.h"
#include "testutil.h"
#include <iostream>
#include <string>
using namespace std;

int test_einsum1x1(vector<int> shape1, vector<int> shape2, const char *file_path1, 
				const char *file_path2,string equation, const char *output_path)
{
	printf("=====   test_einsum1x1 start   ======\n");
	int ret;
	int size1=shape1[0];
	int size2=shape2[0];
	std::vector<ncnn::Mat> data(2);
	std::vector<ncnn::Mat> res;
	float *data1 = readFile(file_path1, size1);
	float *data2 = readFile(file_path2, size2);
	data[0] = InitMat1D_float(size1,data1);
	data[1] = InitMat1D_float(size2,data2);
	ret = einsum(res, data, equation);
	matToFile(output_path, res[0]);

	printf("====    test_einsum1x1 end    =======\n\n");
	return 0;
}
int test_einsum4x3(vector<int> shape1, vector<int> shape2, const char *file_path1,
				   const char *file_path2, string equation, const char *output_path)
{
	printf("=====   test_einsum4x3 start   ======\n");
	int ret;
	std::vector<ncnn::Mat> data(2);
	std::vector<ncnn::Mat> res;
	int w1=shape1[3],h1=shape1[2],c1=shape1[1],b1=shape1[0];
	int w2=shape2[2],h2=shape2[1],c2=shape2[0];
	int input_size1 = w1*h1*c1*b1;
	int input_size2 = w2*h2*c2;

	float *data1 = readFile(file_path1, input_size1);
	float *data2 = readFile(file_path2, input_size2);
	
	data[0] = InitMat4D_float(w1, h1, c1, b1, data1);
	data[1] = InitMat3D_float(w2, h2, c2, data2);
	ret = einsum(res, data, equation);
	matToFIle4d(output_path, res[0]);

	printf("====    test_einsum4x3 end    =======\n\n");
	return 0;
}
int test_einsum4x3channellast(vector<int> shape1, vector<int> shape2, const char *file_path1,
				   const char *file_path2, string equation, const char *output_path,int align_to)
{//统一先找到真实形状，再补零
	printf("=====   test_einsum4x3 start   ======\n");
	int ret;
	std::vector<ncnn::Mat> data(2);
	std::vector<ncnn::Mat> res;
	int w1 = shape1[3], h1 = shape1[2], c1 = shape1[1], b1 = shape1[0];
	int w2 = shape2[2], h2 = shape2[1], c2 = shape2[0];
	int input_size1 = w1 * h1 * c1 * b1;
	int input_size2 = w2 * h2 * c2;
// 用一个新的readFIleChannelast替代，返回float*，参数中多加一个realc,align_to,c,h,w通过block_size,input_size去计算
// 只有最后一块需要通过realc来操作，其他的block整个块都是要的
	int realw1 = 0;
	int realw2=c1;
	float *data1 = readFile(file_path1, input_size1);
	int out_len;
	float *data2 = readFileChannellast(file_path2, c2,h2,w2,align_to,realw2,out_len);

	data[0] = InitMat4D_float(w1, h1, c1, b1, data1);
	data[1] = InitMat3D_float(realw2, h2, c2, data2);
	ret = einsum(res, data, equation);
	matToFIle4d(output_path, res[0]);
	// 要给output补零,主要是res[0].w
	int resw=res[0].w;
	int resh=res[0].h;
	int resd=res[0].d;
	int resc=res[0].c;

	align_channels(output_path,1,1,resc,resd,resh,resw,align_to);
	printf("====    test_einsum4x3 end    =======\n\n");
	return 0;
}
int test_einsum3x4(vector<int> shape1, vector<int> shape2, const char *file_path1,
				   const char *file_path2, string equation, const char *output_path)
{
	printf("=====   test_einsum3x4 start   ======\n");
	int ret;
	std::vector<ncnn::Mat> data(2);
	std::vector<ncnn::Mat> res;
	int w1=shape1[2],h1=shape1[1],c1=shape1[0];
	int w2=shape2[3],h2=shape2[2],c2=shape2[1],b2=shape2[0];
	int input_size1 = w1*h1*c1;
	int input_size2 = w2*h2*c2*b2;

	float *data1 = readFile(file_path1, input_size1);
	float *data2 = readFile(file_path2, input_size2);

	data[0] = InitMat3D_float(w1, h1, c1, data1);
	data[1] = InitMat4D_float(w2, h2, c2, b2, data2);
	ret = einsum(res, data, equation);
	int resw = res[0].w;
	int resh = res[0].h;
	int resd = res[0].d;
	int resc = res[0].c;
	matToFIle4d(output_path, res[0]);

	printf("====    test_einsum3x4 end    =======\n\n");
	return 0;
}
int test_einsum3x4channnellast(vector<int> shape1, vector<int> shape2, const char *file_path1,
				   const char *file_path2, string equation, const char *output_path,int align_to)

{
	printf("=====   test_einsum3x4 start   ======\n");
	int ret;
	std::vector<ncnn::Mat> data(2);
	std::vector<ncnn::Mat> res;
	int w1 = shape1[2], h1 = shape1[1], c1 = shape1[0];
	int w2 = shape2[3], h2 = shape2[2], c2 = shape2[1], b2 = shape2[0];
	int input_size1 = w1 * h1 * c1;
	int input_size2 = w2 * h2 * c2 * b2;

	float *data1 = readFile(file_path1, input_size1);
	float *data2 = readFile(file_path2, input_size2);

	data[0] = InitMat3D_float(w1, h1, c1, data1);
	data[1] = InitMat4D_float(w2, h2, c2, b2, data2);
	ret = einsum(res, data, equation);
	int resw = res[0].w;
	int resh = res[0].h;
	int resd = res[0].d;
	int resc = res[0].c;
	matToFIle4d(output_path, res[0]);
	align_channels(output_path,1,1,resc,resd,resh,resw,align_to);
	printf("====    test_einsum3x4 end    =======\n\n");
	return 0;
}

int test_einsum4x4(vector<int> shape1, vector<int> shape2, const char *file_path1,
				   const char *file_path2, string equation, const char *output_path)
{
	printf("=====   test_einsum4x4 start   ======\n");
	int ret;
	std::vector<ncnn::Mat> data(2);
	std::vector<ncnn::Mat> res;
	int w1=shape1[3],h1=shape1[2],c1=shape1[1],b1=shape1[0];
	int w2=shape2[3],h2=shape2[2],c2=shape2[1],b2=shape2[0];
	int input_size1 = w1*h1*c1*b1;
	int input_size2 = w2*h2*c2*b2;

	float *data1 = readFile(file_path1, input_size1);
	float *data2 = readFile(file_path2, input_size2);
	
	data[0] = InitMat4D_float(w1, h1, c1, b1, data1);
	data[1] = InitMat4D_float(w2, h2, c2, b2, data2);
	ret = einsum(res, data, equation);
	matToFIle4d(output_path, res[0]);

	printf("====    test_einsum4x4 end    =======\n\n");
	return 0;
}

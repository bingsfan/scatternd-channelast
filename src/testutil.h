// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifndef TESTUTIL_H
#define TESTUTIL_H

//#include "layer.h"
#include "mat.h"
#include "prng.h"
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iomanip>

#define EPSILON 1e-8 // 设定精度误差范围
using namespace std;

extern int test_einsum4x3(vector<int> shape1, vector<int> shape2, const char *file_path1,
                   const char *file_path2, string equation, const char *output_path) ;
extern int test_einsum4x3channellast(vector<int> shape1, vector<int> shape2, const char *file_path1,
				   const char *file_path2, string equation, const char *output_path,int align_to);
extern int test_einsum3x4channnellast(vector<int> shape1, vector<int> shape2, const char *file_path1,
                                      const char *file_path2, string equation, const char *output_path, int align_to);
extern int test_einsum4x4(vector<int> shape1, vector<int> shape2, const char *file_path1,
                          const char *file_path2, string equation, const char *output_path);
extern int test_einsum3x4(vector<int> shape1, vector<int> shape2, const char *file_path1,
                          const char *file_path2, string equation, const char *output_path);
static struct prng_rand_t g_prng_rand_state;

#define SRAND(seed) prng_srand(seed, &g_prng_rand_state)
#define RAND()      prng_rand(&g_prng_rand_state)


#define TEST_LAYER_DISABLE_AUTO_INPUT_PACKING (1 << 0)
#define TEST_LAYER_DISABLE_AUTO_INPUT_CASTING (1 << 1)
#define TEST_LAYER_DISABLE_GPU_TESTING        (1 << 2)
#define TEST_LAYER_ENABLE_FORCE_INPUT_PACK8   (1 << 3)


static void align_channels(const std::string &file_path,
                    int D0, int D1, int D2, int D3, int D4, int C,
                    int align_C)
{//smh
    if (align_C != 16 && align_C != 64)
    {
        std::cerr << "通道对齐数必须是 16 或 64，当前输入: " << align_C << std::endl;
        return;
    }

    std::ifstream infile(file_path);
    if (!infile.is_open())
    {
        std::cerr << "无法打开文件: " << file_path << std::endl;
        return;
    }

    // 读取数据
    std::vector<float> data;
    float value;
    while (infile >> value)
    {
        data.push_back(value);
    }
    infile.close();

    // 计算原始张量大小
    int original_size = D0 * D1 * D2 * D3 * D4 * C;
    if (data.size() != original_size)
    {
        std::cerr << "数据大小与张量形状不匹配，期望大小: " << original_size
                  << "，实际大小: " << data.size() << std::endl;
        return;
    }

    // 计算对齐后的通道数
    int aligned_C = (C <= align_C) ? align_C : ((C - 1) / align_C + 1) * align_C;

    // 重新排列数据并进行通道对齐
    std::vector<float> aligned_data(D0 * D1 * D2 * D3 * D4 * aligned_C, 0.0f);

    for (int d0 = 0; d0 < D0; ++d0)
    {
        for (int d1 = 0; d1 < D1; ++d1)
        {
            for (int d2 = 0; d2 < D2; ++d2)
            {
                for (int d3 = 0; d3 < D3; ++d3)
                {
                    for (int d4 = 0; d4 < D4; ++d4)
                    {
                        for (int c = 0; c < C; ++c)
                        {
                            int old_index = (((((d0 * D1 + d1) * D2 + d2) * D3 + d3) * D4 + d4) * C) + c;
                            int new_index = (((((d0 * D1 + d1) * D2 + d2) * D3 + d3) * D4 + d4) * aligned_C) + c;
                            aligned_data[new_index] = data[old_index];
                        }
                    }
                }
            }
        }
    }

    // 将修改后的数据写回文件
    std::ofstream outfile(file_path);
    if (!outfile.is_open())
    {
        std::cerr << "无法写入文件: " << file_path << std::endl;
        return;
    }

    outfile << std::fixed << std::setprecision(6);
    for (const float &val : aligned_data)
    {
        outfile << val << "\n";
    }
    outfile.close();

    std::cout << "通道对齐完成（" << align_C << " 通道对齐），已写回文件: " << file_path << std::endl;
}

static float *readFile(const char *path, int len)
{
    FILE *fp = fopen(path, "r");
    if (fp == NULL)
        printf("cannot open file\n");
    else
    {
        float *dataBuf = (float *)malloc(len * sizeof(float));
        for (int i = 0; i < len; i++)
        {
            fscanf(fp, "%f", dataBuf + i);
        }
        return dataBuf;
    }
}

static float *readFileChannellast(const char *path, int H, int W, int C, int C_aligned, int C_actual, int &out_len)
{//einsum用的
    FILE *fp = fopen(path, "r");
    if (!fp)
    {
        printf("cannot open file\n");
        out_len = 0; // 出错时设置 out_len 为 0
        return nullptr;
    }

    // 计算总数据量
    int total_elements = H * W * C;
    int block_size = H * W * C_aligned; // 每个块的大小

    // 计算总块数
    int num_blocks = total_elements / block_size;

    // 计算有效数据的总长度
    out_len = (num_blocks - 1) * H * W * C_aligned + H * W * C_actual;

    // 分配内存
    float *dataBuf = (float *)malloc(out_len * sizeof(float));
    if (!dataBuf)
    {
        printf("memory allocation failed\n");
        fclose(fp);
        out_len = 0; // 出错时设置 out_len 为 0
        return nullptr;
    }

    int index = 0;
    float temp;

    // 读取前 num_blocks - 1 块
    for (int b = 0; b < num_blocks - 1; b++)
    {
        for (int i = 0; i < block_size; i++)
        {
            if (fscanf(fp, "%f", &temp) != 1)
            {
                printf("Error: Unexpected end of file\n");
                free(dataBuf);
                fclose(fp);
                out_len = 0; // 出错时设置 out_len 为 0
                return nullptr;
            }
            dataBuf[index++] = temp;
        }
    }

    // 读取最后一块，只读取 C_actual 个通道的值
    for (int w = 0; w < W; w++)
    {
        for (int h = 0; h < H; h++) // 每行 h 也要遍历
        {
            for (int c = 0; c < C_actual; c++) // 只读取 C_actual 个通道的数据
            {
                if (fscanf(fp, "%f", &temp) != 1)
                {
                    printf("Error: Unexpected end of file\n");
                    free(dataBuf);
                    fclose(fp);
                    out_len = 0; // 出错时设置 out_len 为 0
                    return nullptr;
                }
                dataBuf[index++] = temp;
            }

            // 跳过 C_aligned - C_actual 个补零的通道数据
            for (int c = C_actual; c < C_aligned; c++)
            {
                if (fscanf(fp, "%f", &temp) != 1)
                {
                    printf("Error: Unexpected end of file\n");
                    free(dataBuf);
                    fclose(fp);
                    out_len = 0; // 出错时设置 out_len 为 0
                    return nullptr;
                }
            }
        }
    }

    fclose(fp);
    return dataBuf;
}
static int *readFileINT( const char *path, int len)
{
    FILE *fp = fopen(path, "r");
    if (fp == NULL)
        printf("cannot open file\n");
    else
    {
        int *dataBuf = (int *)malloc(len * sizeof(int));
        for (int i = 0; i < len; i++)
        {
            fscanf(fp, "%d", dataBuf + i);
        }
        return dataBuf;
    }
}
static void outputFile_line(const char *path, const vector<float> &output)
{
    FILE *fp = fopen(path, "w");
    if (fp == NULL)
    {
        printf("cannot open file for writing\n");
        return;
    }

    for (size_t i = 0; i < output.size(); ++i)
    {
        fprintf(fp, "%.6f\n", output[i]);
    }

    fclose(fp);
    printf("Data written to %s\n", path);
}
inline void WriteTensorToFile(const std::string &filename, const float *data, const std::vector<int> &input_dims, int depth)
{ // yanshaox
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Failed to open output file: " << filename << std::endl;
        return;
    }

    // 计算输入元素总数
    int input_elements = 1;
    for (int dim : input_dims)
    {
        input_elements *= dim;
    }

    // 按输入元素逐行写入one-hot编码
    for (int i = 0; i < input_elements; ++i)
    {
        for (int j = 0; j < depth; ++j)
        {
            // file << data[i * depth + j];
            file << std::fixed << std::setprecision(6) << data[i * depth + j];
            if (j < depth - 1)
            {
                file << "\n";
            }
        }
        file << std::endl;
    }

    file.close();
}
static void outputFile2d_line(const char *path, const vector<vector<int>> &output)
{//smh
    FILE *fp = fopen(path, "w");
    if (fp == NULL)
    {
        printf("cannot open file for writing\n");
        return;
    }

    for (size_t i = 0; i < output.size(); ++i)
    {
        for (size_t j = 0; j < output[i].size(); ++j)
        {
            fprintf(fp, "%d\n", output[i][j]);
        }
    }

    fclose(fp);
    printf("Data written to %s\n", path);
}

static void matToFile(const char *path,const ncnn::Mat &m)
{//smh
    cout<<path<<endl;
    FILE *fp = fopen(path, "w");
    if (fp == NULL)
    {
        printf("cannot open file for writing\n");
        return;
    }
    for (int q = 0; q < m.c; q++)
    {
        const float *ptr = m.channel(q);
        for (int y = 0; y < m.h; y++)
        {
            for (int x = 0; x < m.w; x++)
            {
                fprintf(fp, "%.6f\n", ptr[x]);
            }
            ptr += m.w;
        }
    }
    fclose(fp);
    printf("Data written to %s\n", path);
}
static void matToFIle4d(const char *path,const ncnn::Mat &m){
    FILE *fp = fopen(path, "w");
    if (fp == NULL)
    {
        printf("cannot open file for writing\n");
        return;
    }
    for (int q = 0; q < m.c; q++)
    {
        const float *ptr = m.channel(q);
        for (int z = 0; z < m.d; z++)
        {
            for (int y = 0; y < m.h; y++)
            {
                for (int x = 0; x < m.w; x++)
                {
                    fprintf(fp, "%.6f\n", ptr[x]);
                }
                ptr += m.w;
            }
        }
    }
    fclose(fp);
    printf("Data written to %s\n", path);
}

static void outputFile_space(const char *path, const vector<float> &output)
{//smh
    FILE *fp = fopen(path, "w");
    if (fp == NULL)
    {
        printf("cannot open file for writing\n");
        return;
    }

    for (size_t i = 0; i < output.size(); ++i)
    {
        fprintf(fp, "%.2f ", output[i]);
    }

    fclose(fp);
    printf("Data written to %s\n", path);
}
// 写一个随机生成data的函数，在filepath的txt文件中写入data
static void generate_data(const char *filepath, size_t size)
{//smh
    // srand(time(NULL));
    FILE *fp = fopen(filepath, "w");
    if (fp == NULL)
    {
        printf("cannot open file for writing\n");
        return;
    }
    for (size_t i = 0; i < size; ++i)
    {
        fprintf(fp, "%.6f\n", (float)rand() / RAND_MAX * 100.0f);
    }
    fclose(fp);
    printf("Data written to %s\n", filepath);
}
static void generate_indices(const char *filepath, size_t size, int max)
{
    srand(time(NULL)); // 使用当前时间作为随机种子
    FILE *fp = fopen(filepath, "w");
    if (fp == NULL)
    {
        printf("cannot open file for writing\n");
        return;
    }
    for (size_t i = 0; i < size; ++i)
    {
        fprintf(fp, "%d\n", rand() % max);
    }
    fclose(fp);
    printf("Data written to %s\n", filepath);
}
static void generate_indices_scatternd2(const char *filepath, size_t size, int dim0,int dim1)
{
    srand(time(NULL)); // 使用当前时间作为随机种子
    FILE *fp = fopen(filepath, "w");
    if (fp == NULL)
    {
        printf("cannot open file for writing\n");
        return;
    }
    for (size_t i = 0; i < size; ++i)
    {
        if(i%2==0)
            fprintf(fp, "%d\n", rand() % dim0);
        else
            fprintf(fp, "%d\n", rand() % dim1);
    }
    fclose(fp);
    printf("Data written to %s\n", filepath);
}
static void generate_indices_scatternd3(const char *filepath, size_t size, int dim0, int dim1,int dim2)
{
    srand(time(NULL)); // 使用当前时间作为随机种子
    FILE *fp = fopen(filepath, "w");
    if (fp == NULL)
    {
        printf("cannot open file for writing\n");
        return;
    }
    for (size_t i = 0; i < size; ++i)
    {
        if (i % 3 == 0)
            fprintf(fp, "%d\n", rand() % dim0);
        else if(i%3==1)
            fprintf(fp, "%d\n", rand() % dim1);
        else
            fprintf(fp, "%d\n", rand() % dim2);
    }
    fclose(fp);
    printf("Data written to %s\n", filepath);
}
static void matToVector(const ncnn::Mat &m,vector<float> &res)
{//smh
    for(int q=0;q<m.c;q++)
    {
        const float* ptr = m.channel(q);
        for(int y=0;y<m.h;y++)
        {
            for(int x=0;x<m.w;x++)
            {
                res.push_back(ptr[x]);
            }
            ptr += m.w;
        }
    }
}
static void matToVector4d(const ncnn::Mat &m, vector<float> &res)
{ // smh
    for (int q = 0; q < m.c; q++)
    {
        const float *ptr = m.channel(q);
        for (int z = 0; z < m.d; z++)
        {
            for (int y = 0; y < m.h; y++)
            {
                for (int x = 0; x < m.w; x++)
                {
                    res.push_back(ptr[x]);
                }
                ptr += m.w;
            }
        }
    }
}
static void print_mat(const ncnn::Mat& m)
{
	printf("mat.c  = %d\n", m.c);
	printf("mat.h  = %d\n", m.h);
	printf("mat.w  = %d\n", m.w);

	for (int q = 0; q < m.c; q++)
	{
		const float* ptr = m.channel(q);
		for (int y = 0; y < m.h; y++)
		{
			for (int x = 0; x < m.w; x++)
			{
				printf("%f ", ptr[x]);
			}
			ptr += m.w;
			printf("\n");
		}
		printf("------------------------\n");
	}
	printf("\n");
}
static void print_matshape(const ncnn::Mat &m)
{//smh
    printf("mat.c  = %d\n", m.c);
    printf("mat.h  = %d\n", m.h);
    printf("mat.w  = %d\n", m.w);
    printf("\n");
}
static void print_mat4d(const ncnn::Mat &m)
{//smh
    printf("mat.c  = %d\n", m.c);
    printf("mat.d  = %d\n", m.d);
    printf("mat.h  = %d\n", m.h);
    printf("mat.w  = %d\n", m.w);

    for (int q = 0; q < m.c; q++)
    {
        const float *ptr = m.channel(q);
        for (int z = 0; z < m.d; z++)
        {
            for (int y = 0; y < m.h; y++)
            {
                for (int x = 0; x < m.w; x++)
                {
                    printf("%f ", ptr[x]);
                }
                ptr += m.w;
                printf("\n");
            }
            printf("------------------------\n");
        }
        printf("------------------------\n");
    }
    printf("\n");
}
static void print_mat4dshape(const ncnn::Mat &m)
{//smh
    printf("mat.c  = %d\n", m.c);
    printf("mat.d  = %d\n", m.d);
    printf("mat.h  = %d\n", m.h);
    printf("mat.w  = %d\n", m.w);
    printf("\n");
}
static ncnn::Mat InitMat1D_int(int w, int* data)
{
	ncnn::Mat m(w);
	for (int i = 0; i < w; i++)
	{
		m[i] = data[i];
	}

	return m;
}
static ncnn::Mat InitMat1D_float(int w, float* data)
{
	ncnn::Mat m(w);
	for (int i = 0; i < w; i++)
	{
		m[i] = data[i];
	}

	return m;
}
static ncnn::Mat InitMat_int(int w, int h, int* data)
{
	ncnn::Mat m(w, h);
	int size = w * h;
	for (int i = 0; i < size; i++)
	{
		m[i] = data[i];
	}

	return m;
}
static ncnn::Mat InitMat_float(int w, int h, float* data)
{
	ncnn::Mat m(w, h);
	int size = w * h;
	for (int i = 0; i < size; i++)
	{
		m[i] = data[i];
	}

	return m;
}
static ncnn::Mat InitMat3D_float(int w, int h, int c, float* data)
{
	ncnn::Mat m(w, h, c);
	int size = w * h * c;
	for (int i = 0; i < size; i++)
	{
		m[i] = data[i];
	}

	return m;
}
static ncnn::Mat InitMat3D_int(int w, int h, int c,int *data)
{//smh
    ncnn::Mat m(w, h, c);
    int size = w * h * c;
    for (int i = 0; i < size; i++)
    {
        m[i] = data[i];
    }
    return m;
}
static ncnn::Mat InitMat2D_int(int w, int h)
{
	ncnn::Mat m(w, h);
	int size = w * h;
	for (int i = 0; i < size; i++)
	{
		m[i] = i;
	}

	return m;
}
static ncnn::Mat InitMat2D_int(int w, int h, int *data)
{//smh
    ncnn::Mat m(w, h);
    int size = w * h;
    for (int i = 0; i < size; i++)
    {
        m[i] = data[i];
    }

    return m;
}

static ncnn::Mat InitMat3D_int(int w, int h, int c)
{
	ncnn::Mat m(w,h,c);
	int size = w * h * c;
	for (int i = 0; i < size; i++)
	{
		m[i] = i;
	}

	return m;
}

static ncnn::Mat InitMat4D_int(int w, int h, int c,int b)
{
	ncnn::Mat m(w, h, c, b);
	int size = w * h * c * b;
	for (int i = 0; i < size; i++)
	{
		m[i] = i;
	}
	return m;
}
static ncnn::Mat InitMat4D_int(int w, int h, int c, int b,int *data)
{//smh
    ncnn::Mat m(w, h, c, b);
    int size = w * h * c * b;
    for (int i = 0; i < size; i++)
    {
        m[i] = data[i];
    }
    return m;
}
static ncnn::Mat InitMat4D_float(int w, int h, int c, int b, float *data)
{//smh
    ncnn::Mat m(w, h, c, b);
    int size = w * h * c * b;
    for (int i = 0; i < size; i++)
    {
        m[i] = data[i];
    }

    return m;
}

static float RandomFloat(float a = -1.2f, float b = 1.2f)
{
    float random = ((float)RAND()) / (float)uint64_t(-1); //RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    float v = a + r;
    // generate denormal as zero
    if (v < 0.0001 && v > -0.0001)
        v = 0.f;
    return v;
}

static int RandomInt(int a = -10000, int b = 10000)
{
    float random = ((float)RAND()) / (float)uint64_t(-1); //RAND_MAX;
    int diff = b - a;
    float r = random * diff;
    return a + (int)r;
}

static signed char RandomS8()
{
    return (signed char)RandomInt(-127, 127);
}

static void Randomize(ncnn::Mat& m, float a = -1.2f, float b = 1.2f)
{
    for (size_t i = 0; i < m.total(); i++)
    {
        m[i] = RandomFloat(a, b);
    }
}

static void RandomizeInt(ncnn::Mat& m, int a = -10000, int b = 10000)
{
    for (size_t i = 0; i < m.total(); i++)
    {
        ((int*)m)[i] = RandomInt(a, b);
    }
}

static void RandomizeS8(ncnn::Mat& m)
{
    for (size_t i = 0; i < m.total(); i++)
    {
        ((signed char*)m)[i] = RandomS8();
    }
}

static ncnn::Mat RandomMat(int w, float a = -1.2f, float b = 1.2f)
{
    ncnn::Mat m(w);
    Randomize(m, a, b);
    return m;
}

static ncnn::Mat RandomMat(int w, int h, float a = -1.2f, float b = 1.2f)
{
    ncnn::Mat m(w, h);
    Randomize(m, a, b);
    return m;
}

static ncnn::Mat RandomMat(int w, int h, int c, float a = -1.2f, float b = 1.2f)
{
    ncnn::Mat m(w, h, c);
    Randomize(m, a, b);
    return m;
}

static ncnn::Mat RandomMat(int w, int h, int d, int c, float a = -1.2f, float b = 1.2f)
{
    ncnn::Mat m(w, h, d, c);
    Randomize(m, a, b);
    return m;
}

static ncnn::Mat RandomIntMat(int w)
{
    ncnn::Mat m(w);
    RandomizeInt(m);
    return m;
}

static ncnn::Mat RandomIntMat(int w, int h)
{
    ncnn::Mat m(w, h);
    RandomizeInt(m);
    return m;
}

static ncnn::Mat RandomIntMat(int w, int h, int c)
{
    ncnn::Mat m(w, h, c);
    RandomizeInt(m);
    return m;
}

static ncnn::Mat RandomIntMat(int w, int h, int d, int c)
{
    ncnn::Mat m(w, h, d, c);
    RandomizeInt(m);
    return m;
}

static ncnn::Mat RandomS8Mat(int w)
{
    ncnn::Mat m(w, (size_t)1u);
    RandomizeS8(m);
    return m;
}

static ncnn::Mat RandomS8Mat(int w, int h)
{
    ncnn::Mat m(w, h, (size_t)1u);
    RandomizeS8(m);
    return m;
}

static ncnn::Mat RandomS8Mat(int w, int h, int c)
{
    ncnn::Mat m(w, h, c, (size_t)1u);
    RandomizeS8(m);
    return m;
}

static ncnn::Mat RandomS8Mat(int w, int h, int d, int c)
{
    ncnn::Mat m(w, h, d, c, (size_t)1u);
    RandomizeS8(m);
    return m;
}

static ncnn::Mat scales_mat(const ncnn::Mat& mat, int m, int k, int ldx)
{
    ncnn::Mat weight_scales(m);
    for (int i = 0; i < m; ++i)
    {
        float min = mat[0], _max = mat[0];
        const float* ptr = (const float*)(mat.data) + i * ldx;
        for (int j = 0; j < k; ++j)
        {
            if (min > ptr[j])
            {
                min = ptr[j];
            }
            if (_max < ptr[j])
            {
                _max = ptr[j];
            }
        }
        const float abs_min = abs(min), abs_max = abs(_max);
        weight_scales[i] = 127.f / (abs_min > abs_max ? abs_min : abs_max);
    }
    return weight_scales;
}

static bool NearlyEqual(float a, float b, float epsilon)
{
    if (a == b)
        return true;

    float diff = (float)fabs(a - b);
    if (diff <= epsilon)
        return true;

    // relative error
    return diff < epsilon * (std::max)(fabs(a), fabs(b));
}

static int Compare(const ncnn::Mat& a, const ncnn::Mat& b, float epsilon = 0.001)
{
#define CHECK_MEMBER(m)                                                                 \
    if (a.m != b.m)                                                                     \
    {                                                                                   \
        fprintf(stderr, #m " not match    expect %d but got %d\n", (int)a.m, (int)b.m); \
        return -1;                                                                      \
    }

    CHECK_MEMBER(dims)
    CHECK_MEMBER(w)
    CHECK_MEMBER(h)
    CHECK_MEMBER(d)
    CHECK_MEMBER(c)
    CHECK_MEMBER(elemsize)
    CHECK_MEMBER(elempack)

#undef CHECK_MEMBER

    for (int q = 0; q < a.c; q++)
    {
        const ncnn::Mat ma = a.channel(q);
        const ncnn::Mat mb = b.channel(q);
        for (int z = 0; z < a.d; z++)
        {
            const ncnn::Mat da = ma.depth(z);
            const ncnn::Mat db = mb.depth(z);
            for (int i = 0; i < a.h; i++)
            {
                const float* pa = da.row(i);
                const float* pb = db.row(i);
                for (int j = 0; j < a.w; j++)
                {
                    if (!NearlyEqual(pa[j], pb[j], epsilon))
                    {
                        fprintf(stderr, "value not match  at c:%d d:%d h:%d w:%d    expect %f but got %f\n", q, z, i, j, pa[j], pb[j]);
                        return -1;
                    }
                }
            }
        }
    }

    return 0;
}
#if 0
static int CompareMat(const ncnn::Mat& a, const ncnn::Mat& b, float epsilon = 0.001)
{
    ncnn::Option opt;
    opt.num_threads = 1;

    if (a.elempack != 1)
    {
        ncnn::Mat a1;
        ncnn::convert_packing(a, a1, 1, opt);
        return CompareMat(a1, b, epsilon);
    }

    if (b.elempack != 1)
    {
        ncnn::Mat b1;
        ncnn::convert_packing(b, b1, 1, opt);
        return CompareMat(a, b1, epsilon);
    }

    if (a.elemsize == 2u)
    {
        ncnn::Mat a32;
        cast_float16_to_float32(a, a32, opt);
        return CompareMat(a32, b, epsilon);
    }
    if (a.elemsize == 1u)
    {
        ncnn::Mat a32;
        cast_int8_to_float32(a, a32, opt);
        return CompareMat(a32, b, epsilon);
    }

    if (b.elemsize == 2u)
    {
        ncnn::Mat b32;
        cast_float16_to_float32(b, b32, opt);
        return CompareMat(a, b32, epsilon);
    }
    if (b.elemsize == 1u)
    {
        ncnn::Mat b32;
        cast_int8_to_float32(b, b32, opt);
        return CompareMat(a, b32, epsilon);
    }

    return Compare(a, b, epsilon);
}

static int CompareMat(const std::vector<ncnn::Mat>& a, const std::vector<ncnn::Mat>& b, float epsilon = 0.001)
{
    if (a.size() != b.size())
    {
        fprintf(stderr, "output blob count not match %zu %zu\n", a.size(), b.size());
        return -1;
    }

    for (size_t i = 0; i < a.size(); i++)
    {
        if (CompareMat(a[i], b[i], epsilon))
        {
            fprintf(stderr, "output blob %zu not match\n", i);
            return -1;
        }
    }

    return 0;
}

template<typename T>
int test_layer_naive(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const std::vector<ncnn::Mat>& a, int top_blob_count, std::vector<ncnn::Mat>& b, void (*func)(T*), int flag)
{
    ncnn::Layer* op = ncnn::create_layer(typeindex);

    if (func)
    {
        (*func)((T*)op);
    }

    op->load_param(pd);

    if (op->one_blob_only && a.size() != 1)
    {
        fprintf(stderr, "layer with one_blob_only but consume multiple inputs\n");
        delete op;
        return -1;
    }

    ncnn::ModelBinFromMatArray mb(weights.data());

    op->load_model(mb);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.lightmode = false;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_shader_pack8 = false;
    opt.use_image_storage = false;
    opt.use_bf16_storage = false;
    opt.use_vulkan_compute = false;

    op->create_pipeline(opt);

    b.resize(top_blob_count);

    if (op->support_inplace)
    {
        for (size_t i = 0; i < a.size(); i++)
        {
            b[i] = a[i].clone();
        }

        ((T*)op)->T::forward_inplace(b, opt);
    }
    else
    {
        ((T*)op)->T::forward(a, b, opt);
    }

    op->destroy_pipeline(opt);

    delete op;

    return 0;
}

template<typename T>
int test_layer_cpu(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& _opt, const std::vector<ncnn::Mat>& a, int top_blob_count, std::vector<ncnn::Mat>& c, const std::vector<ncnn::Mat>& top_shapes, void (*func)(T*), int flag)
{
    ncnn::Layer* op = ncnn::create_layer(typeindex);

    if (!op->support_packing && _opt.use_packing_layout)
    {
        delete op;
        return 233;
    }
    if (!op->support_bf16_storage && !op->support_fp16_storage && (_opt.use_bf16_storage || _opt.use_fp16_arithmetic))
    {
        delete op;
        return 233;
    }

    if (func)
    {
        (*func)((T*)op);
    }

    if (!top_shapes.empty())
    {
        op->bottom_shapes = a;
        op->top_shapes = top_shapes;
    }

    op->load_param(pd);

    if (op->one_blob_only && a.size() != 1)
    {
        fprintf(stderr, "layer with one_blob_only but consume multiple inputs\n");
        delete op;
        return -1;
    }

    ncnn::ModelBinFromMatArray mb(weights.data());

    op->load_model(mb);

    ncnn::Option opt = _opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = false;

    op->create_pipeline(opt);

    if (!op->support_packing && _opt.use_packing_layout)
    {
        op->destroy_pipeline(opt);
        delete op;
        return 233;
    }
    if (!op->support_bf16_storage && !op->support_fp16_storage && (_opt.use_bf16_storage || _opt.use_fp16_arithmetic))
    {
        op->destroy_pipeline(opt);
        delete op;
        return 233;
    }

    std::vector<ncnn::Mat> a4(a.size());

    for (size_t i = 0; i < a4.size(); i++)
    {
        // clang-format off
        // *INDENT-OFF*
#if NCNN_ARM82
        if (opt.use_fp16_storage && ncnn::cpu_support_arm_asimdhp() && op->support_fp16_storage && !(flag & TEST_LAYER_DISABLE_AUTO_INPUT_CASTING))
        {
            ncnn::cast_float32_to_float16(a[i], a4[i], opt);
        }
        else
#endif // NCNN_ARM82
#if NCNN_RVV
        if (opt.use_fp16_storage && ncnn::cpu_support_riscv_v() && ncnn::cpu_support_riscv_zfh() && op->support_fp16_storage && !(flag & TEST_LAYER_DISABLE_AUTO_INPUT_CASTING))
        {
            ncnn::cast_float32_to_float16(a[i], a4[i], opt);
        }
        else
#endif // NCNN_RVV
#if NCNN_BF16
        if (opt.use_bf16_storage && op->support_bf16_storage && !(flag & TEST_LAYER_DISABLE_AUTO_INPUT_CASTING))
        {
            ncnn::cast_float32_to_bfloat16(a[i], a4[i], opt);
        }
        else
#endif // NCNN_BF16
        if (opt.use_fp16_storage && op->support_fp16_storage && !(flag & TEST_LAYER_DISABLE_AUTO_INPUT_CASTING))
        {
            ncnn::cast_float32_to_float16(a[i], a4[i], opt);
        }
        else
        {
            a4[i] = a[i];
        }
        // *INDENT-ON*
        // clang-format on

        if (opt.use_packing_layout && op->support_packing && !(flag & TEST_LAYER_DISABLE_AUTO_INPUT_PACKING))
        {
            // resolve dst_elempack
            int dims = a4[i].dims;
            int elemcount = 0;
            if (dims == 1) elemcount = a4[i].elempack * a4[i].w;
            if (dims == 2) elemcount = a4[i].elempack * a4[i].h;
            if (dims == 3 || dims == 4) elemcount = a4[i].elempack * a4[i].c;

            int elembits = a4[i].elembits();

            int dst_elempack = 1;

            if (elembits == 32)
            {
#if NCNN_AVX512
                if (elemcount % 16 == 0 && ncnn::cpu_support_x86_avx512())
                    dst_elempack = 16;
                else if (elemcount % 8 == 0 && ncnn::cpu_support_x86_avx())
                    dst_elempack = 8;
                else if (elemcount % 4 == 0)
                    dst_elempack = 4;
#elif NCNN_AVX
                if (elemcount % 8 == 0 && ncnn::cpu_support_x86_avx())
                    dst_elempack = 8;
                else if (elemcount % 4 == 0)
                    dst_elempack = 4;
#elif NCNN_RVV
                const int packn = ncnn::cpu_riscv_vlenb() / (elembits / 8);
                if (elemcount % packn == 0)
                    dst_elempack = packn;
#else
                if (elemcount % 4 == 0)
                    dst_elempack = 4;
#endif
            }
            if (elembits == 16)
            {
#if NCNN_ARM82
                if (elemcount % 8 == 0 && ncnn::cpu_support_arm_asimdhp() && opt.use_fp16_arithmetic)
                    dst_elempack = 8;
                else if (elemcount % 4 == 0)
                    dst_elempack = 4;
#elif NCNN_RVV
                const int packn = ncnn::cpu_riscv_vlenb() / 2;
                if (elemcount % packn == 0)
                    dst_elempack = packn;
#else
                if (elemcount % 4 == 0)
                    dst_elempack = 4;
#endif
            }
            if (elembits == 8)
            {
#if NCNN_RVV
                const int packn = ncnn::cpu_riscv_vlenb() / 1;
                if (elemcount % packn == 0)
                    dst_elempack = packn;
#else
                if (elemcount % 8 == 0)
                    dst_elempack = 8;
#endif
            }

            if (flag & TEST_LAYER_ENABLE_FORCE_INPUT_PACK8)
                dst_elempack = 8;

            ncnn::Mat a4_packed;
            ncnn::convert_packing(a4[i], a4_packed, dst_elempack, opt);
            a4[i] = a4_packed;
        }
    }

    c.resize(top_blob_count);

    if (op->support_inplace)
    {
        for (size_t i = 0; i < a4.size(); i++)
        {
            c[i] = a4[i].clone();
        }

        op->forward_inplace(c, opt);
    }
    else
    {
        op->forward(a4, c, opt);
    }

    for (size_t i = 0; i < c.size(); i++)
    {
        // clang-format off
        // *INDENT-OFF*
#if NCNN_ARM82
        if (opt.use_fp16_storage && ncnn::cpu_support_arm_asimdhp() && op->support_fp16_storage && c[i].elembits() == 16)
        {
            ncnn::Mat c_fp32;
            ncnn::cast_float16_to_float32(c[i], c_fp32, opt);
            c[i] = c_fp32;
        }
        else
#endif // NCNN_ARM82
#if NCNN_RVV
        if (opt.use_fp16_storage && ncnn::cpu_support_riscv_v() && ncnn::cpu_support_riscv_zfh() && op->support_fp16_storage && c[i].elembits() == 16)
        {
            ncnn::Mat c_fp32;
            ncnn::cast_float16_to_float32(c[i], c_fp32, opt);
            c[i] = c_fp32;
        }
        else
#endif // NCNN_RVV
#if NCNN_BF16
        if (opt.use_bf16_storage && op->support_bf16_storage && c[i].elembits() == 16)
        {
            ncnn::Mat c_fp32;
            ncnn::cast_bfloat16_to_float32(c[i], c_fp32, opt);
            c[i] = c_fp32;
        }
        else
#endif // NCNN_BF16
        if (opt.use_fp16_storage && op->support_fp16_storage && c[i].elembits() == 16)
        {
            ncnn::Mat c_fp32;
            ncnn::cast_float16_to_float32(c[i], c_fp32, opt);
            c[i] = c_fp32;
        }
        // *INDENT-ON*
        // clang-format on
    }

    op->destroy_pipeline(opt);

    delete op;

    return 0;
}

#if NCNN_VULKAN
template<typename T>
int test_layer_gpu(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& _opt, const std::vector<ncnn::Mat>& a, int top_blob_count, std::vector<ncnn::Mat>& d, const std::vector<ncnn::Mat>& top_shapes, void (*func)(T*), int flag)
{
    if (!_opt.use_packing_layout)
    {
        // pack1 test is useless for gpu
        return 233;
    }

    ncnn::Layer* op = ncnn::create_layer(typeindex);

    if (!op->support_vulkan)
    {
        delete op;
        return 233;
    }

    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();

    op->vkdev = vkdev;

    if (func)
    {
        (*func)((T*)op);
    }

    if (!top_shapes.empty())
    {
        op->bottom_shapes = a;
        op->top_shapes = top_shapes;
    }

    op->load_param(pd);

    if (op->one_blob_only && a.size() != 1)
    {
        fprintf(stderr, "layer with one_blob_only but consume multiple inputs\n");
        delete op;
        return -1;
    }

    ncnn::ModelBinFromMatArray mb(weights.data());

    op->load_model(mb);

    ncnn::VkWeightAllocator g_weight_vkallocator(vkdev);
    ncnn::VkWeightStagingAllocator g_weight_staging_vkallocator(vkdev);

    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();

    ncnn::Option opt = _opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;

#if __APPLE__
    opt.use_image_storage = false;
#endif

    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    if (!vkdev->info.support_fp16_packed()) opt.use_fp16_packed = false;
    if (!vkdev->info.support_fp16_storage()) opt.use_fp16_storage = false;
    if (!vkdev->info.support_fp16_arithmetic()) opt.use_fp16_arithmetic = false;
    if (!vkdev->info.support_cooperative_matrix()) opt.use_cooperative_matrix = false;

    // FIXME fp16a may produce large error
    opt.use_fp16_arithmetic = false;

    op->create_pipeline(opt);

    if (!op->support_vulkan)
    {
        op->destroy_pipeline(opt);
        delete op;
        return 233;
    }

    {
        ncnn::VkTransfer cmd(vkdev);

        ncnn::Option opt_upload = opt;
        opt_upload.blob_vkallocator = &g_weight_vkallocator;
        opt_upload.workspace_vkallocator = &g_weight_vkallocator;
        opt_upload.staging_vkallocator = &g_weight_staging_vkallocator;

        op->upload_model(cmd, opt_upload);

        cmd.submit_and_wait();
    }

    d.resize(top_blob_count);

    {
        // forward
        ncnn::VkCompute cmd(vkdev);

        if (op->support_image_storage && opt.use_image_storage)
        {
            // upload
            std::vector<ncnn::VkImageMat> a_gpu(a.size());
            for (size_t i = 0; i < a_gpu.size(); i++)
            {
                cmd.record_upload(a[i], a_gpu[i], opt);
            }

            std::vector<ncnn::VkImageMat> d_gpu(top_blob_count);
            if (op->support_inplace)
            {
                op->forward_inplace(a_gpu, cmd, opt);

                d_gpu = a_gpu;
            }
            else
            {
                op->forward(a_gpu, d_gpu, cmd, opt);
            }

            // download
            for (size_t i = 0; i < d_gpu.size(); i++)
            {
                cmd.record_download(d_gpu[i], d[i], opt);
            }
        }
        else
        {
            // upload
            std::vector<ncnn::VkMat> a_gpu(a.size());
            for (size_t i = 0; i < a_gpu.size(); i++)
            {
                cmd.record_upload(a[i], a_gpu[i], opt);
            }

            std::vector<ncnn::VkMat> d_gpu(top_blob_count);
            if (op->support_inplace)
            {
                op->forward_inplace(a_gpu, cmd, opt);

                d_gpu = a_gpu;
            }
            else
            {
                op->forward(a_gpu, d_gpu, cmd, opt);
            }

            // download
            for (size_t i = 0; i < d_gpu.size(); i++)
            {
                cmd.record_download(d_gpu[i], d[i], opt);
            }
        }

        cmd.submit_and_wait();
    }

    op->destroy_pipeline(opt);

    delete op;

    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);
    g_weight_vkallocator.clear();
    g_weight_staging_vkallocator.clear();

    return 0;
}
#endif // NCNN_VULKAN


template<typename T>
int test_layer(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& _opt, const std::vector<ncnn::Mat>& a, int top_blob_count, const std::vector<ncnn::Mat>& top_shapes = std::vector<ncnn::Mat>(), float epsilon = 0.001, void (*func)(T*) = 0, int flag = 0)
{
    // naive
    std::vector<ncnn::Mat> b;
    {
        int ret = test_layer_naive(typeindex, pd, weights, a, top_blob_count, b, func, flag);
        if (ret != 233 && ret != 0)
        {
            fprintf(stderr, "test_layer_naive failed\n");
            return -1;
        }
    }

    // cpu
    {
        std::vector<ncnn::Mat> c;
        int ret = test_layer_cpu(typeindex, pd, weights, _opt, a, top_blob_count, c, std::vector<ncnn::Mat>(), func, flag);
        if (ret != 233 && (ret != 0 || CompareMat(b, c, epsilon) != 0))
        {
            fprintf(stderr, "test_layer_cpu failed\n");
            return -1;
        }
    }

    // cpu shape hint
    {
        std::vector<ncnn::Mat> c;
        int ret = test_layer_cpu(typeindex, pd, weights, _opt, a, top_blob_count, c, b, func, flag);
        if (ret != 233 && (ret != 0 || CompareMat(b, c, epsilon) != 0))
        {
            fprintf(stderr, "test_layer_cpu failed with shape hint\n");
            return -1;
        }
    }

#if NCNN_VULKAN
    // gpu
    if (!(flag & TEST_LAYER_DISABLE_GPU_TESTING))
    {
        std::vector<ncnn::Mat> d;
        int ret = test_layer_gpu(typeindex, pd, weights, _opt, a, top_blob_count, d, std::vector<ncnn::Mat>(), func, flag);
        if (ret != 233 && (ret != 0 || CompareMat(b, d, epsilon) != 0))
        {
            fprintf(stderr, "test_layer_gpu failed\n");
            return -1;
        }
    }

    // gpu shape hint
    if (!(flag & TEST_LAYER_DISABLE_GPU_TESTING))
    {
        std::vector<ncnn::Mat> d;
        int ret = test_layer_gpu(typeindex, pd, weights, _opt, a, top_blob_count, d, b, func, flag);
        if (ret != 233 && (ret != 0 || CompareMat(b, d, epsilon) != 0))
        {
            fprintf(stderr, "test_layer_gpu failed with shape hint\n");
            return -1;
        }
    }
#endif // NCNN_VULKAN

    return 0;
}

template<typename T>
int test_layer_naive(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Mat& a, ncnn::Mat& b, void (*func)(T*), int flag)
{
    ncnn::Layer* op = ncnn::create_layer(typeindex);

    if (func)
    {
        (*func)((T*)op);
    }

    op->load_param(pd);

    ncnn::ModelBinFromMatArray mb(weights.data());

    op->load_model(mb);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.lightmode = false;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_shader_pack8 = false;
    opt.use_image_storage = false;
    opt.use_bf16_storage = false;
    opt.use_vulkan_compute = false;

    op->create_pipeline(opt);

    if (op->support_inplace)
    {
        b = a.clone();
        ((T*)op)->T::forward_inplace(b, opt);
    }
    else
    {
        ((T*)op)->T::forward(a, b, opt);
    }

    op->destroy_pipeline(opt);

    delete op;

    return 0;
}

template<typename T>
int test_layer_cpu(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& _opt, const ncnn::Mat& a, ncnn::Mat& c, const ncnn::Mat& top_shape, void (*func)(T*), int flag)
{
    ncnn::Layer* op = ncnn::create_layer(typeindex);

    if (!op->support_packing && _opt.use_packing_layout)
    {
        delete op;
        return 233;
    }
    if (!op->support_bf16_storage && !op->support_fp16_storage && (_opt.use_bf16_storage || _opt.use_fp16_arithmetic))
    {
        delete op;
        return 233;
    }

    if (func)
    {
        (*func)((T*)op);
    }

    if (top_shape.dims)
    {
        op->bottom_shapes.resize(1);
        op->top_shapes.resize(1);
        op->bottom_shapes[0] = a;
        op->top_shapes[0] = top_shape;
    }

    op->load_param(pd);

    ncnn::ModelBinFromMatArray mb(weights.data());

    op->load_model(mb);

    ncnn::Option opt = _opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = false;

    op->create_pipeline(opt);

    if (!op->support_packing && _opt.use_packing_layout)
    {
        op->destroy_pipeline(opt);
        delete op;
        return 233;
    }
    if (!op->support_bf16_storage && !op->support_fp16_storage && (_opt.use_bf16_storage || _opt.use_fp16_arithmetic))
    {
        op->destroy_pipeline(opt);
        delete op;
        return 233;
    }

    ncnn::Mat a4;

    // clang-format off
    // *INDENT-OFF*
#if NCNN_ARM82
    if (opt.use_fp16_storage && ncnn::cpu_support_arm_asimdhp() && op->support_fp16_storage && !(flag & TEST_LAYER_DISABLE_AUTO_INPUT_CASTING))
    {
        ncnn::cast_float32_to_float16(a, a4, opt);
    }
    else
#endif // NCNN_ARM82
#if NCNN_RVV
    if (opt.use_fp16_storage && ncnn::cpu_support_riscv_v() && ncnn::cpu_support_riscv_zfh() && op->support_fp16_storage && !(flag & TEST_LAYER_DISABLE_AUTO_INPUT_CASTING))
    {
        ncnn::cast_float32_to_float16(a, a4, opt);
    }
    else
#endif // NCNN_RVV
#if NCNN_BF16
    if (opt.use_bf16_storage && op->support_bf16_storage && !(flag & TEST_LAYER_DISABLE_AUTO_INPUT_CASTING))
    {
        ncnn::cast_float32_to_bfloat16(a, a4, opt);
    }
    else
#endif // NCNN_BF16
    if (opt.use_fp16_storage && op->support_fp16_storage && !(flag & TEST_LAYER_DISABLE_AUTO_INPUT_CASTING))
    {
        ncnn::cast_float32_to_float16(a, a4, opt);
    }
    else
    {
        a4 = a;
    }
    // *INDENT-ON*
    // clang-format on

    if (opt.use_packing_layout && op->support_packing && !(flag & TEST_LAYER_DISABLE_AUTO_INPUT_PACKING))
    {
        // resolve dst_elempack
        int dims = a4.dims;
        int elemcount = 0;
        if (dims == 1) elemcount = a4.elempack * a4.w;
        if (dims == 2) elemcount = a4.elempack * a4.h;
        if (dims == 3 || dims == 4) elemcount = a4.elempack * a4.c;

        int elembits = a4.elembits();

        int dst_elempack = 1;

        if (elembits == 32)
        {
#if NCNN_AVX512
            if (elemcount % 16 == 0 && ncnn::cpu_support_x86_avx512())
                dst_elempack = 16;
            else if (elemcount % 8 == 0 && ncnn::cpu_support_x86_avx())
                dst_elempack = 8;
            else if (elemcount % 4 == 0)
                dst_elempack = 4;
#elif NCNN_AVX
            if (elemcount % 8 == 0 && ncnn::cpu_support_x86_avx())
                dst_elempack = 8;
            else if (elemcount % 4 == 0)
                dst_elempack = 4;
#elif NCNN_RVV
            const int packn = ncnn::cpu_riscv_vlenb() / (elembits / 8);
            if (elemcount % packn == 0)
                dst_elempack = packn;
#else
            if (elemcount % 4 == 0)
                dst_elempack = 4;
#endif
        }
        if (elembits == 16)
        {
#if NCNN_ARM82
            if (elemcount % 8 == 0 && ncnn::cpu_support_arm_asimdhp() && opt.use_fp16_arithmetic)
                dst_elempack = 8;
            else if (elemcount % 4 == 0)
                dst_elempack = 4;
#elif NCNN_RVV
            const int packn = ncnn::cpu_riscv_vlenb() / 2;
            if (elemcount % packn == 0)
                dst_elempack = packn;
#else
            if (elemcount % 4 == 0)
                dst_elempack = 4;
#endif
        }
        if (elembits == 8)
        {
#if NCNN_RVV
            const int packn = ncnn::cpu_riscv_vlenb() / 1;
            if (elemcount % packn == 0)
                dst_elempack = packn;
#else
            if (elemcount % 8 == 0)
                dst_elempack = 8;
#endif
        }

        if (flag & TEST_LAYER_ENABLE_FORCE_INPUT_PACK8)
            dst_elempack = 8;

        ncnn::Mat a4_packed;
        ncnn::convert_packing(a4, a4_packed, dst_elempack, opt);
        a4 = a4_packed;
    }

    if (op->support_inplace)
    {
        c = a4.clone();
        op->forward_inplace(c, opt);
    }
    else
    {
        op->forward(a4, c, opt);
    }

    // clang-format off
    // *INDENT-OFF*
#if NCNN_ARM82
    if (opt.use_fp16_storage && ncnn::cpu_support_arm_asimdhp() && op->support_fp16_storage && c.elembits() == 16)
    {
        ncnn::Mat c_fp32;
        ncnn::cast_float16_to_float32(c, c_fp32, opt);
        c = c_fp32;
    }
    else
#endif // NCNN_ARM82
#if NCNN_RVV
    if (opt.use_fp16_storage && ncnn::cpu_support_riscv_v() && ncnn::cpu_support_riscv_zfh() && op->support_fp16_storage && c.elembits() == 16)
    {
        ncnn::Mat c_fp32;
        ncnn::cast_float16_to_float32(c, c_fp32, opt);
        c = c_fp32;
    }
    else
#endif // NCNN_RVV
#if NCNN_BF16
    if (opt.use_bf16_storage && op->support_bf16_storage && c.elembits() == 16)
    {
        ncnn::Mat c_fp32;
        ncnn::cast_bfloat16_to_float32(c, c_fp32, opt);
        c = c_fp32;
    }
    else
#endif // NCNN_BF16
    if (opt.use_fp16_storage && op->support_fp16_storage && c.elembits() == 16)
    {
        ncnn::Mat c_fp32;
        ncnn::cast_float16_to_float32(c, c_fp32, opt);
        c = c_fp32;
    }
    // *INDENT-ON*
    // clang-format on

    op->destroy_pipeline(opt);

    delete op;

    return 0;
}

#if NCNN_VULKAN
template<typename T>
int test_layer_gpu(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& _opt, const ncnn::Mat& a, ncnn::Mat& d, const ncnn::Mat& top_shape, void (*func)(T*), int flag)
{
    if (!_opt.use_packing_layout)
    {
        // pack1 test is useless for gpu
        return 233;
    }

    ncnn::Layer* op = ncnn::create_layer(typeindex);

    if (!op->support_vulkan)
    {
        delete op;
        return 233;
    }

    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();

    op->vkdev = vkdev;

    if (func)
    {
        (*func)((T*)op);
    }

    if (top_shape.dims)
    {
        op->bottom_shapes.resize(1);
        op->top_shapes.resize(1);
        op->bottom_shapes[0] = a;
        op->top_shapes[0] = top_shape;
    }

    op->load_param(pd);

    ncnn::ModelBinFromMatArray mb(weights.data());

    op->load_model(mb);

    ncnn::VkWeightAllocator g_weight_vkallocator(vkdev);
    ncnn::VkWeightStagingAllocator g_weight_staging_vkallocator(vkdev);

    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();

    ncnn::Option opt = _opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;

#if __APPLE__
    opt.use_image_storage = false;
#endif

    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    if (!vkdev->info.support_fp16_packed()) opt.use_fp16_packed = false;
    if (!vkdev->info.support_fp16_storage()) opt.use_fp16_storage = false;
    if (!vkdev->info.support_fp16_arithmetic()) opt.use_fp16_arithmetic = false;
    if (!vkdev->info.support_cooperative_matrix()) opt.use_cooperative_matrix = false;

    // FIXME fp16a may produce large error
    opt.use_fp16_arithmetic = false;

    op->create_pipeline(opt);

    if (!op->support_vulkan)
    {
        op->destroy_pipeline(opt);
        delete op;
        return 233;
    }

    {
        ncnn::VkTransfer cmd(vkdev);

        ncnn::Option opt_upload = opt;
        opt_upload.blob_vkallocator = &g_weight_vkallocator;
        opt_upload.workspace_vkallocator = &g_weight_vkallocator;
        opt_upload.staging_vkallocator = &g_weight_staging_vkallocator;

        op->upload_model(cmd, opt_upload);

        cmd.submit_and_wait();
    }

    {
        // forward
        ncnn::VkCompute cmd(vkdev);

        if (op->support_image_storage && opt.use_image_storage)
        {
            // upload
            ncnn::VkImageMat a_gpu;
            cmd.record_upload(a, a_gpu, opt);

            ncnn::VkImageMat d_gpu;
            if (op->support_inplace)
            {
                op->forward_inplace(a_gpu, cmd, opt);

                d_gpu = a_gpu;
            }
            else
            {
                op->forward(a_gpu, d_gpu, cmd, opt);
            }

            // download
            cmd.record_download(d_gpu, d, opt);
        }
        else
        {
            // upload
            ncnn::VkMat a_gpu;
            cmd.record_upload(a, a_gpu, opt);

            ncnn::VkMat d_gpu;
            if (op->support_inplace)
            {
                op->forward_inplace(a_gpu, cmd, opt);

                d_gpu = a_gpu;
            }
            else
            {
                op->forward(a_gpu, d_gpu, cmd, opt);
            }

            // download
            cmd.record_download(d_gpu, d, opt);
        }

        cmd.submit_and_wait();
    }

    op->destroy_pipeline(opt);

    delete op;

    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);
    g_weight_vkallocator.clear();
    g_weight_staging_vkallocator.clear();

    return 0;
}
#endif // NCNN_VULKAN

template<typename T>
int test_layer(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& _opt, const ncnn::Mat& a, const ncnn::Mat& top_shape = ncnn::Mat(), float epsilon = 0.001, void (*func)(T*) = 0, int flag = 0)
{
    // naive
    ncnn::Mat b;
    {
        int ret = test_layer_naive(typeindex, pd, weights, a, b, func, flag);
        if (ret != 233 && ret != 0)
        {
            fprintf(stderr, "test_layer_naive failed\n");
            return -1;
        }
    }

    // cpu
    {
        ncnn::Mat c;
        int ret = test_layer_cpu(typeindex, pd, weights, _opt, a, c, ncnn::Mat(), func, flag);
        if (ret != 233 && (ret != 0 || CompareMat(b, c, epsilon) != 0))
        {
            fprintf(stderr, "test_layer_cpu failed\n");
            return -1;
        }
    }

    // cpu shape hint
    {
        ncnn::Mat c;
        int ret = test_layer_cpu(typeindex, pd, weights, _opt, a, c, b, func, flag);
        if (ret != 233 && (ret != 0 || CompareMat(b, c, epsilon) != 0))
        {
            fprintf(stderr, "test_layer_cpu failed with shape hint\n");
            return -1;
        }
    }

#if NCNN_VULKAN
    // gpu
    if (!(flag & TEST_LAYER_DISABLE_GPU_TESTING))
    {
        ncnn::Mat d;
        int ret = test_layer_gpu(typeindex, pd, weights, _opt, a, d, ncnn::Mat(), func, flag);
        if (ret != 233 && (ret != 0 || CompareMat(b, d, epsilon) != 0))
        {
            fprintf(stderr, "test_layer_gpu failed\n");
            return -1;
        }
    }

    // gpu shape hint
    if (!(flag & TEST_LAYER_DISABLE_GPU_TESTING))
    {
        ncnn::Mat d;
        int ret = test_layer_gpu(typeindex, pd, weights, _opt, a, d, b, func, flag);
        if (ret != 233 && (ret != 0 || CompareMat(b, d, epsilon) != 0))
        {
            fprintf(stderr, "test_layer_gpu failed with shape hint\n");
            return -1;
        }
    }
#endif // NCNN_VULKAN

    return 0;
}

template<typename T>
int test_layer_opt(const char* layer_type, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& opt, const std::vector<ncnn::Mat>& a, int top_blob_count = 1, float epsilon = 0.001, void (*func)(T*) = 0, int flag = 0)
{
    // fp16 representation
    std::vector<ncnn::Mat> a_fp16;
    if (opt.use_bf16_storage && !(flag & TEST_LAYER_DISABLE_AUTO_INPUT_CASTING))
    {
        a_fp16.resize(a.size());
        for (size_t j = 0; j < a.size(); j++)
        {
            ncnn::Mat tmp;
            ncnn::cast_float32_to_bfloat16(a[j], tmp, opt);
            ncnn::cast_bfloat16_to_float32(tmp, a_fp16[j], opt);
        }
    }
    else if ((opt.use_fp16_packed || opt.use_fp16_storage) && !(flag & TEST_LAYER_DISABLE_AUTO_INPUT_CASTING))
    {
        a_fp16.resize(a.size());
        for (size_t j = 0; j < a.size(); j++)
        {
            ncnn::Mat tmp;
            ncnn::cast_float32_to_float16(a[j], tmp, opt);
            ncnn::cast_float16_to_float32(tmp, a_fp16[j], opt);
        }
    }
    else
    {
        a_fp16 = a;
    }

    std::vector<ncnn::Mat> weights_fp16;
    float epsilon_fp16;
    if (opt.use_bf16_storage)
    {
        weights_fp16.resize(weights.size());
        for (size_t j = 0; j < weights.size(); j++)
        {
            ncnn::Mat tmp;
            ncnn::cast_float32_to_bfloat16(weights[j], tmp, opt);
            ncnn::cast_bfloat16_to_float32(tmp, weights_fp16[j], opt);
        }
        epsilon_fp16 = epsilon * 100; // 0.1
    }
    else if (opt.use_fp16_packed || opt.use_fp16_storage)
    {
        weights_fp16.resize(weights.size());
        for (size_t j = 0; j < weights.size(); j++)
        {
            ncnn::Mat tmp;
            ncnn::cast_float32_to_float16(weights[j], tmp, opt);
            ncnn::cast_float16_to_float32(tmp, weights_fp16[j], opt);
        }
        epsilon_fp16 = epsilon * 100; // 0.1
    }
    else
    {
        weights_fp16 = weights;
        epsilon_fp16 = epsilon;
    }

    if (opt.use_fp16_arithmetic)
    {
        epsilon_fp16 = epsilon * 1000; // 1.0
    }

    std::vector<ncnn::Mat> top_shapes;
    int ret = test_layer<T>(ncnn::layer_to_index(layer_type), pd, weights_fp16, opt, a_fp16, top_blob_count, top_shapes, epsilon_fp16, func, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_layer %s failed use_packing_layout=%d use_fp16_packed=%d use_fp16_storage=%d use_fp16_arithmetic=%d use_shader_pack8=%d use_bf16_storage=%d use_image_storage=%d use_sgemm_convolution=%d use_winograd_convolution=%d\n", layer_type, opt.use_packing_layout, opt.use_fp16_packed, opt.use_fp16_storage, opt.use_fp16_arithmetic, opt.use_shader_pack8, opt.use_bf16_storage, opt.use_image_storage, opt.use_sgemm_convolution, opt.use_winograd_convolution);
        return ret;
    }

    return 0;
}

template<typename T>
int test_layer_opt(const char* layer_type, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& opt, const ncnn::Mat& a, float epsilon = 0.001, void (*func)(T*) = 0, int flag = 0)
{
    // fp16 representation
    ncnn::Mat a_fp16;
    if (opt.use_bf16_storage && !(flag & TEST_LAYER_DISABLE_AUTO_INPUT_CASTING))
    {
        ncnn::Mat tmp;
        ncnn::cast_float32_to_bfloat16(a, tmp, opt);
        ncnn::cast_bfloat16_to_float32(tmp, a_fp16, opt);
    }
    else if ((opt.use_fp16_packed || opt.use_fp16_storage) && !(flag & TEST_LAYER_DISABLE_AUTO_INPUT_CASTING))
    {
        ncnn::Mat tmp;
        ncnn::cast_float32_to_float16(a, tmp, opt);
        ncnn::cast_float16_to_float32(tmp, a_fp16, opt);
    }
    else
    {
        a_fp16 = a;
    }

    std::vector<ncnn::Mat> weights_fp16;
    float epsilon_fp16;
    if (opt.use_bf16_storage)
    {
        weights_fp16.resize(weights.size());
        for (size_t j = 0; j < weights.size(); j++)
        {
            ncnn::Mat tmp;
            ncnn::cast_float32_to_bfloat16(weights[j], tmp, opt);
            ncnn::cast_bfloat16_to_float32(tmp, weights_fp16[j], opt);
        }
        epsilon_fp16 = epsilon * 100; // 0.1
    }
    else if (opt.use_fp16_packed || opt.use_fp16_storage)
    {
        weights_fp16.resize(weights.size());
        for (size_t j = 0; j < weights.size(); j++)
        {
            ncnn::Mat tmp;
            ncnn::cast_float32_to_float16(weights[j], tmp, opt);
            ncnn::cast_float16_to_float32(tmp, weights_fp16[j], opt);
        }
        epsilon_fp16 = epsilon * 100; // 0.1
    }
    else
    {
        weights_fp16 = weights;
        epsilon_fp16 = epsilon;
    }

    if (opt.use_fp16_arithmetic)
    {
        epsilon_fp16 = epsilon * 1000; // 1.0
    }

    ncnn::Mat top_shape;
    int ret = test_layer<T>(ncnn::layer_to_index(layer_type), pd, weights_fp16, opt, a_fp16, top_shape, epsilon_fp16, func, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_layer %s failed use_packing_layout=%d use_fp16_packed=%d use_fp16_storage=%d use_fp16_arithmetic=%d use_shader_pack8=%d use_bf16_storage=%d use_image_storage=%d use_sgemm_convolution=%d use_winograd_convolution=%d\n", layer_type, opt.use_packing_layout, opt.use_fp16_packed, opt.use_fp16_storage, opt.use_fp16_arithmetic, opt.use_shader_pack8, opt.use_bf16_storage, opt.use_image_storage, opt.use_sgemm_convolution, opt.use_winograd_convolution);
        return ret;
    }

    return 0;
}

template<typename T>
int test_layer(const char* layer_type, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const std::vector<ncnn::Mat>& a, int top_blob_count = 1, float epsilon = 0.001, void (*func)(T*) = 0, int flag = 0)
{
    // pack fp16p fp16s fp16a bf16s shader8 image
    const int options[][7] = {
        {0, 0, 0, 0, 0, 0, 0},
        {0, 0, 1, 0, 0, 0, 0},
        {0, 0, 1, 1, 1, 0, 0},
        {1, 0, 0, 0, 0, 0, 0},
        {1, 1, 0, 0, 1, 0, 0},
        {1, 0, 1, 0, 0, 1, 0},
        {1, 1, 1, 1, 0, 1, 1},
    };

    const int opt_count = sizeof(options) / sizeof(options[0]);

    for (int i = 0; i < opt_count; i++)
    {
        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = options[i][0];
        opt.use_fp16_packed = options[i][1];
        opt.use_fp16_storage = options[i][2];
        opt.use_fp16_arithmetic = options[i][3];
        opt.use_bf16_storage = options[i][4];
        opt.use_shader_pack8 = options[i][5];
        opt.use_image_storage = options[i][6];

        int ret = test_layer_opt<T>(layer_type, pd, weights, opt, a, top_blob_count, epsilon, func, flag);
        if (ret != 0)
            return ret;
    }

    return 0;
}

template<typename T>
int test_layer(const char* layer_type, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Mat& a, float epsilon = 0.001, void (*func)(T*) = 0, int flag = 0)
{
    // pack fp16p fp16s fp16a bf16s shader8 image
    const int options[][7] = {
        {0, 0, 0, 0, 0, 0, 0},
        {0, 0, 1, 0, 0, 0, 0},
        {0, 0, 1, 1, 1, 0, 0},
        {1, 0, 0, 0, 0, 0, 0},
        {1, 1, 0, 0, 1, 0, 0},
        {1, 0, 1, 0, 0, 1, 0},
        {1, 1, 1, 1, 0, 1, 1},
    };

    const int opt_count = sizeof(options) / sizeof(options[0]);

    for (int i = 0; i < opt_count; i++)
    {
        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = options[i][0];
        opt.use_fp16_packed = options[i][1];
        opt.use_fp16_storage = options[i][2];
        opt.use_fp16_arithmetic = options[i][3];
        opt.use_bf16_storage = options[i][4];
        opt.use_shader_pack8 = options[i][5];
        opt.use_image_storage = options[i][6];

        int ret = test_layer_opt<T>(layer_type, pd, weights, opt, a, epsilon, func, flag);
        if (ret != 0)
            return ret;
    }

    return 0;
}
#endif 

#endif // TESTUTIL_H

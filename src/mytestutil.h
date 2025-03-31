#ifndef MYTESTUTIL_H
#define MYTESTUTIL_H

#include "mat.h"
#include <iostream>
#include <map>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>
using namespace std;
extern int test_scatterND3x2x1(vector<int> shape1, vector<int> shape2, vector<int> shape3, const char *file_path,
                               const char *indices_path, const char *updates_path, const char *output_path, int op);

int test_scatterND3x2x1channellast(vector<int> shape1, vector<int> shape2, vector<int> shape3, const char *file_path,
                                   const char *indices_path, const char *updates_path, const char *output_path, int op,
                                   int align_to);

extern int test_scatterND3x4x3channellast(vector<int> shape1, vector<int> shape2, vector<int> shape3,
                                          const char *file_path, const char *indices_path, const char *updates_path,
                                          const char *output_path, int op, int align_to);
extern int test_scatterND4x5x4channellast(vector<int> shape1, vector<int> shape2, vector<int> shape3,
                                          const char *file_path, const char *indices_path, const char *updates_path,
                                          const char *output_path, int op, int align_to);
extern int test_scatterND3x3x3(vector<int> shape1, vector<int> shape2, vector<int> shape3, const char *file_path,
                               const char *indices_path, const char *updates_path, const char *output_path, int op);
extern int test_scatterND3x3x3channellast(vector<int> shape1, vector<int> shape2, vector<int> shape3,
                                          const char *file_path, const char *indices_path, const char *updates_path,
                                          const char *output_path, int op, int align_to);
/**
 * @description: 给scatternd三维索引用的，将第一列的数据转移到最后一列去，和input匹配
 * @param {int} *indices
 * @param {int} rows
 * @param {int} cols
 * @return {*}
 */
static void rotateColumns(int *indices, int rows, int cols)
{
    for (int i = 0; i < rows; ++i)
    {
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
static void moveSecondColumnToLast(int *array, int rows, int cols)
{
    for (int i = 0; i < rows; ++i)
    {
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
static void printMatWithIndex(const char *prefix, int index, const ncnn::Mat &m)
{
    char filename[256];
    // 生成带索引的文件名，如 input_0.txt, index_0.txt
    snprintf(filename, sizeof(filename), "%s_%d.txt", prefix, index);

    FILE *fp = fopen(filename, "w");
    if (!fp)
    {
        fprintf(stderr, "无法打开文件 %s\n", filename);
        return;
    }

    // 写入数据
    for (int q = 0; q < m.c; ++q)
    {
        const float *ptr = m.channel(q);
        for (int y = 0; y < m.h; ++y)
        {
            for (int x = 0; x < m.w; ++x)
            {
                fprintf(fp, "%.6f\n", ptr[x]);
            }
            ptr += m.w;
        }
    }

    fclose(fp);
    printf("数据已写入 %s\n", filename);
}
static void writeAllMats4dToFile(const char *path, const std::vector<ncnn::Mat> &result)
{
    FILE *fp = fopen(path, "w");
    if (fp == nullptr)
    {
        printf("无法打开文件 %s\n", path);
        return;
    }

    for (size_t i = 0; i < result.size(); i++)
    {
        const ncnn::Mat &m = result[i];
        for (int q = 0; q < m.c; q++)
        {
            const float *ptr = m.channel(q);
            for (int z = 0; z < m.d; z++) // 新增d维度循环
            {
                for (int y = 0; y < m.h; y++)
                {
                    for (int x = 0; x < m.w; x++)
                    {
                        fprintf(fp, "%.6f\n", ptr[x]);
                    }
                    ptr += m.w; // 按行步进指针
                }
            }
        }
    }

    fclose(fp);
    printf("所有数据已写入 %s\n", path);
}
static void writeAllMatsToFile(const char *path, const std::vector<ncnn::Mat> &result)
{                                // smh
    FILE *fp = fopen(path, "w"); // 只打开一次文件
    if (fp == nullptr)
    {
        printf("无法打开文件 %s\n", path);
        return;
    }

    // 遍历所有 result 元素
    for (size_t i = 0; i < result.size(); i++)
    {
        const ncnn::Mat &m = result[i];

        // 添加区块分隔标记

        // 写入数据
        for (int q = 0; q < m.c; q++)
        {
            const float *ptr = m.channel(q);
            for (int y = 0; y < m.h; y++)
            {
                for (int x = 0; x < m.w; x++)
                {
                    fprintf(fp, "%.6f\n", ptr[x]);
                }
                // fprintf(fp, "\n"); // 行尾换行
                ptr += m.w;
            }
            // fprintf(fp, "\n"); // 通道间空行
        }
    }

    fclose(fp);
    printf("所有数据已写入 %s\n", path);
}
// 修改后的文件写入函数（添加索引后缀）
static void matToFileWithIndex(const char *base_path, int index, const ncnn::Mat &m)
{
    char path[256];
    // 生成带索引的唯一文件名，例如 output_0.txt, output_1.txt
    snprintf(path, sizeof(path), "%s_%d.txt", base_path, index);

    FILE *fp = fopen(path, "w");
    if (fp == nullptr)
    {
        printf("无法打开文件 %s\n", path);
        return;
    }

    // 写入维度信息（可选）
    // fprintf(fp, "Channels: %d\nHeight: %d\nWidth: %d\n", m.c, m.h, m.w);

    // 写入数据
    for (int q = 0; q < m.c; q++)
    {
        const float *ptr = m.channel(q);
        for (int y = 0; y < m.h; y++)
        {
            for (int x = 0; x < m.w; x++)
            {
                fprintf(fp, "%.6f\n", ptr[x]); // 用空格分隔数据点
            }
            ptr += m.w;
        }
    }

    fclose(fp);
    printf("数据已写入 %s\n", path);
}
#endif
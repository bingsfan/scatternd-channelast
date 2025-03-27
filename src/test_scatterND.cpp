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

#include "op.h"
#include "testutil.h"
#include "mytestutil.h"

int test_scatterND3x2x1(vector<int> shape1, vector<int> shape2, vector<int> shape3, const char *file_path,
                        const char *indices_path, const char *updates_path, const char *output_path, int op)
{
    printf("=====   test_scatterND3x2x1 start   ======\n");
    int ret;
    int w1 = shape1[2], h1 = shape1[1], c1 = shape1[0];
    int w2 = shape2[1], h2 = shape2[0];
    int w3 = shape3[0];
    int input_size = w1 * h1 * c1;
    int indices_size = w2 * h2;
    int update_size = w3;
    float *data = readFile(file_path, input_size);
    int *indices = readFileINT(indices_path, indices_size);
    float *updates = readFile(updates_path, update_size);

    ncnn::Mat input = InitMat3D_float(w1, h1, c1, data);
    ncnn::Mat index = InitMat2D_int(w2, h2, indices);
    ncnn::Mat update = InitMat1D_float(w3, updates);
    ncnn::Mat result;
    scatterND(result, input, index, update, op);
    matToFile(output_path, result);

    printf("====    test_scatterND3x2x1 end    =======\n\n");
    return 0;
}

void rotateColumns(int *indices, int rows, int cols)
{
    for (int i = 0; i < rows; ++i)
    {
        std::rotate(indices + i * cols, indices + i * cols + 1, indices + (i + 1) * cols);
    }
}
void moveSecondColumnToLast(int *array, int rows, int cols)
{
    for (int i = 0; i < rows; ++i)
    {
        std::rotate(array + i * cols + 1, array + i * cols + 2, array + (i + 1) * cols);
    }
}
vector<vector<int>> reconstructQueue(vector<vector<int>> &people)
{
    sort(people.begin(),people.end(),[](vector<int> a,vector<int> b){
        if(a[0]==b[0]) return a[1]<b[1];
        return a[0]>b[0];
    });
    vector<vector<int>> res;
    for(int i=0;i<people.size();i++){
        int pos = people[i][1];
        res.insert(res.begin()+pos,people[i]);
    }
    return res;
}

int test_scatterND3x2x1channellast(vector<int> shape1, vector<int> shape2, vector<int> shape3, const char *file_path,
                                   const char *indices_path, const char *updates_path, const char *output_path, int op,int align_to)
{
    printf("=====   test_scatterND3x2x1 start   ======\n");
    int ret;
    int w1 = shape1[2], h1 = shape1[1], c1 = shape1[0];
    int w2 = shape2[1], h2 = shape2[0];
    int w3 = shape3[0];
    int input_size = w1 * h1 * c1;
    int indices_size = w2 * h2;
    int update_size = w3;

    int block_size = input_size / shape1[shape1.size() - 1] * align_to;
    int blockNums = input_size / block_size;

    vector<vector<float>> newinput(blockNums);
    vector<vector<int>> newIndices(blockNums);
    vector<vector<float>> newUpdates(blockNums);

    vector<size_t> block_shape(shape1.size());
    for (int i = 0; i < shape1.size() - 1; i++)
    {
        block_shape[i] = shape1[i];
    }
    block_shape[shape1.size() - 1] = align_to;

    float *data = readFile(file_path, input_size);
    int *indices = readFileINT(indices_path, indices_size);
    float *updates = readFile(updates_path, update_size);
    rotateColumns(indices, h2, w2); // 添加的更换索引的轴，将0,1,2换成1,2,0

    for (int i = 0; i < blockNums; i++)
    {
        newinput[i] = vector<float>(data + i * block_size, data + (i + 1) * block_size);
        // 将newinput写入文件我挨个看
    }
    vector<int> vec_index(indices,indices+indices_size);
    int c_index=0;
    int k=0;
    // 将每一块的c和updates放到不同的数组中去
    for(int i=2;i<indices_size;i+=3){
        int j=indices[i]/align_to;
        newIndices[j].push_back(indices[i - 2]);
        newIndices[j].push_back(indices[i - 1]);
        newIndices[j].push_back(indices[i]%align_to);
        newUpdates[j].push_back(updates[k++]);
    }


    ncnn::Mat input;
    ncnn::Mat index;
    ncnn::Mat update;
    vector<ncnn::Mat> result(blockNums);
    // 得写一个方法将blocknums块的索引分开,input,index，updates都得分开
    // input在读取的时候按块分开了，index和updates需要手动计算出来
    
    for (int j = 0; j < blockNums; j++)
    {
        input = InitMat3D_float(align_to, h1, c1, newinput[j].data());
        index = InitMat2D_int(3, newUpdates[j].size(), newIndices[j].data());
        update = InitMat1D_float(newUpdates[j].size(), newUpdates[j].data());
        printMatWithIndex("input", j, input);   // 生成 input_0.txt, input_1.txt...
        printMatWithIndex("index", j, index);   // 生成 index_0.txt, index_1.txt...
        printMatWithIndex("update", j, update); // 生成 update_0.txt, update_1.txt...
        scatterND(result[j], input, index, update, op);
        // matToFileWithIndex(output_path, j, result[j]);
    }

    writeAllMatsToFile(output_path,result);
    printf("====    test_scatterND3x2x1 end    =======\n\n");
    return 0;
}
int test_scatterND3x3x3(vector<int> shape1, vector<int> shape2, vector<int> shape3, const char *file_path,
                        const char *indices_path, const char *updates_path, const char *output_path, int op)
{
    printf("=====   test_scatterND3x3x3 start   ======\n");
    int ret;
    int w1 = shape1[2], h1 = shape1[1], c1 = shape1[0];
    int w2 = shape2[2], h2 = shape2[1], c2 = shape2[0];
    int w3 = shape3[2], h3 = shape3[1], c3 = shape3[0];
    int input_size = w1 * h1 * c1;
    int indices_size = w2 * h2 * c2;
    int update_size = w3 * h3 * c3;
    float *data = readFile(file_path, input_size);
    int *indices = readFileINT(indices_path, indices_size);
    float *updates = readFile(updates_path, update_size);

    ncnn::Mat input = InitMat3D_float(w1, h1, c1, data);
    ncnn::Mat index = InitMat3D_int(w2, h2, c2, indices);
    ncnn::Mat update = InitMat3D_float(w3, h3, c3, updates);
    ncnn::Mat result;
    scatterND(result, input, index, update, op);
    matToFile(output_path, result);

    printf("====    test_scatterND3x3x3 end    =======\n\n");
    return 0;
}
int test_scatterND3x3x3channellast(vector<int> shape1, vector<int> shape2, vector<int> shape3, const char *file_path,
                        const char *indices_path, const char *updates_path, const char *output_path, int op)
{
    printf("=====   test_scatterND3x3x3 start   ======\n");
    int ret;
    int w1 = shape1[2], h1 = shape1[1], c1 = shape1[0];
    int w2 = shape2[2], h2 = shape2[1], c2 = shape2[0];
    int w3 = shape3[2], h3 = shape3[1], c3 = shape3[0];
    int input_size = w1 * h1 * c1;
    int indices_size = w2 * h2 * c2;
    int update_size = w3 * h3 * c3;
    float *data = readFile(file_path, input_size);
    int *indices = readFileINT(indices_path, indices_size);
    float *updates = readFile(updates_path, update_size);

    // 转变索引，将索引变成四维的，调用3x4x3
    vector<int> new_indices(c2 * h2 * w3 * (w2 + 1));

    for (int i = 0; i < c2; ++i)
    {
        for (int j = 0; j < h2; ++j)
        {
            for (int k = 0; k < w3; ++k)
            {
                for (int l = 0; l < w2 + 1; ++l)
                {
                    if (l < w2)
                    {
                        new_indices[((i * h2 + j) * w3 + k) * (w2 + 1) + l] = indices[(i * h2 + j) * w2 + l];
                    }
                    else
                    {
                        new_indices[((i * h2 + j) * w3 + k) * (w2 + 1) + l] = k; // 新加的那一列的值是0->w3-1
                    }
                }
            }
        }
    }
    rotateColumns(new_indices.data(), c2*h2*w3, w2+1);
    // 使用 new_indices 进行后续操作
    ncnn::Mat input = InitMat3D_float(w1, h1, c1, data);
    ncnn::Mat index = InitMat4D_int(w2 + 1, w3, h2, c2, new_indices.data());
    ncnn::Mat update = InitMat3D_float(w3, h3, c3, updates);
    ncnn::Mat result;
    scatterND(result, input, index, update, op);
    matToFile(output_path, result);

    printf("====    test_scatterND3x3x3 end    =======\n\n");
    return 0;
}
int test_scatterND3x4x3(vector<int> shape1, vector<int> shape2, vector<int> shape3, const char *file_path,
                        const char *indices_path, const char *updates_path, const char *output_path, int op)
{
    printf("=====   test_scatterND3x4x3 start   ======\n");
    int ret;
    int w1 = shape1[2], h1 = shape1[1], c1 = shape1[0];
    int w2 = shape2[3], h2 = shape2[2], c2 = shape2[1], b2 = shape2[0];
    int w3 = shape3[2], h3 = shape3[1], c3 = shape3[0];
    int input_size = w1 * h1 * c1;
    int indices_size = w2 * h2 * c2 * b2;
    int update_size = w3 * h3 * c3;
    float *data = readFile(file_path, input_size);
    int *indices = readFileINT(indices_path, indices_size);
    float *updates = readFile(updates_path, update_size);

    ncnn::Mat input = InitMat3D_float(w1, h1, c1, data);
    ncnn::Mat index = InitMat4D_int(w2, h2, c2, b2, indices);
    ncnn::Mat update = InitMat3D_float(w3, h3, c3, updates);
    ncnn::Mat result;
    scatterND(result, input, index, update, op);
    matToFile(output_path, result);

    printf("====    test_scatterND3x4x3 end    =======\n\n");
    return 0;
}
int test_scatterND3x4x3channellast(vector<int> shape1, vector<int> shape2, vector<int> shape3, const char *file_path,
                        const char *indices_path, const char *updates_path, const char *output_path, int op,int align_to)
{
    printf("=====   test_scatterND3x4x3 start   ======\n");
    int ret;
    int w1 = shape1[2], h1 = shape1[1], c1 = shape1[0];
    int w2 = shape2[3], h2 = shape2[2], c2 = shape2[1], b2 = shape2[0];
    int w3 = shape3[2], h3 = shape3[1], c3 = shape3[0];
    int input_size = w1 * h1 * c1;
    int indices_size = w2 * h2 * c2 * b2;
    int update_size = w3 * h3 * c3;

    float *data = readFile(file_path, input_size);
    int *indices = readFileINT(indices_path, indices_size);
    float *updates = readFile(updates_path, update_size);
    rotateColumns(indices, h2 *c2*b2, w2);


    int block_size = input_size / shape1[shape1.size() - 1] * align_to;
    int blockNums = input_size / block_size;

    vector<vector<float>> newinput(blockNums);
    vector<vector<int>> newIndices(blockNums);
    vector<vector<float>> newUpdates(blockNums);
    vector<size_t> block_shape(shape1.size());
    for (int i = 0; i < shape1.size() - 1; i++)
    {
        block_shape[i] = shape1[i];
    }
    block_shape[shape1.size() - 1] = align_to;
    for (int i = 0; i < blockNums; i++)
    {
        newinput[i] = vector<float>(data + i * block_size, data + (i + 1) * block_size);
        // 将newinput写入文件我挨个看
    }
    // 处理indices和updates
    vector<int> vec_index(indices, indices + indices_size);
    int c_index = 0;
    int k = 0;
    // 将每一块的c和updates放到不同的数组中去
    for (int i = 2; i < indices_size; i += 3)
    {
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
    // 得写一个方法将blocknums块的索引分开,input,index，updates都得分开
    // input在读取的时候按块分开了，index和updates需要手动计算出来

    for (int j = 0; j < blockNums; j++)
    {
        input = InitMat3D_float(align_to, h1, c1, newinput[j].data());
        index = InitMat2D_int(3, newUpdates[j].size(), newIndices[j].data());
        update = InitMat1D_float(newUpdates[j].size(), newUpdates[j].data());
        // printMatWithIndex("input", j, input);   // 生成 input_0.txt, input_1.txt...
        // printMatWithIndex("index", j, index);   // 生成 index_0.txt, index_1.txt...
        // printMatWithIndex("update", j, update); // 生成 update_0.txt, update_1.txt...
        scatterND(result[j], input, index, update, op);
        // matToFileWithIndex(output_path, j, result[j]);
    }
    writeAllMatsToFile(output_path, result);
    printf("====    test_scatterND3x4x3 end    =======\n\n");
    return 0;
}
int test_scatterND4x5x4(vector<int> shape1, vector<int> shape2, vector<int> shape3, const char *file_path,
                        const char *indices_path, const char *updates_path, const char *output_path, int op)
{ // 目前只能处理indices是五维，但是第一维是1的情况
    printf("=====   test_scatterND4x5x4 start   ======\n");
    int ret;
    int w1 = shape1[3], h1 = shape1[2], c1 = shape1[1], b1 = shape1[0];
    int w2 = shape2[4], h2 = shape2[3], c2 = shape2[2], b2 = shape2[1], d2 = shape2[0];
    int w3 = shape3[3], h3 = shape3[2], c3 = shape3[1], b3 = shape3[0];
    int input_size = w1 * h1 * c1 * b1;
    int indices_size = w2 * h2 * c2 * b2 * d2;
    int update_size = w3 * h3 * c3 * b3;
    float *data = readFile(file_path, input_size);
    int *indices = readFileINT(indices_path, indices_size);
    float *updates = readFile(updates_path, update_size);
    
    ncnn::Mat input = InitMat4D_float(w1, h1, c1, b1, data);
    ncnn::Mat index = InitMat4D_int(w2, h2, c2, b2, indices);
    ncnn::Mat update = InitMat3D_float(w3, h3, c3, updates);
    ncnn::Mat result;
    scatterND(result, input, index, update, op);
    matToFIle4d(output_path, result);

    printf("====    test_scatterND4x5x4 end    =======\n\n");
    return 0;
}
int test_scatterND4x5x4channellast(vector<int> shape1, vector<int> shape2, vector<int> shape3, const char *file_path,
                        const char *indices_path, const char *updates_path, const char *output_path, int op)
{ // 目前只能处理indices是五维，但是第一维是1的情况
    printf("=====   test_scatterND4x5x4 start   ======\n");
    int ret;
    int w1 = shape1[3], h1 = shape1[2], c1 = shape1[1], b1 = shape1[0];
    int w2 = shape2[4], h2 = shape2[3], c2 = shape2[2], b2 = shape2[1], d2 = shape2[0];
    int w3 = shape3[3], h3 = shape3[2], c3 = shape3[1], b3 = shape3[0];
    int input_size = w1 * h1 * c1 * b1;
    int indices_size = w2 * h2 * c2 * b2 * d2;
    int update_size = w3 * h3 * c3 * b3;
    float *data = readFile(file_path, input_size);
    int *indices = readFileINT(indices_path, indices_size);
    float *updates = readFile(updates_path, update_size);
    moveSecondColumnToLast(indices, h2 * c2 * b2 * d2,w2);
    ncnn::Mat input = InitMat4D_float(w1, h1, c1, b1, data);
    ncnn::Mat index = InitMat4D_int(w2, h2, c2, b2, indices);
    ncnn::Mat update = InitMat3D_float(w3, h3, c3, updates);
    ncnn::Mat result;
    scatterND(result, input, index, update, op);
    matToFIle4d(output_path, result);

    printf("====    test_scatterND4x5x4 end    =======\n\n");
    return 0;
}


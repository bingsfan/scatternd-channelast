// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// COElementsITIONS OF ANY KIElements, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "scatterND.h"

//#include <algorithm>
#include <cmath>

namespace ncnn {
using namespace std;
ScatterND::ScatterND()
{
}
int ScatterND::forward(const Mat &input, const Mat &indices, const Mat &update, Mat &output)const
{
	output = input.clone();


	float* input_data = (float*)input.data;
	float* update_data = (float*)update.data;
	float* output_data = (float*)output.data;
	float* pIndex = (float*)indices.data;


	//for (int i = 0; i < indices.h * indices.w * indices.c; i++)
	//	printf("%f  ", pIndex[i]);

	int indice_rank = indices.dims;
	int last_indice_dimension = indices.dimsVec[indice_rank - 1];
	if (last_indice_dimension > input.dims) {
		NCNN_LOGE("Error: last dimension of indices larger than input blob dims size ");
		return -1;
	}

	int update_rank = update.dims;
	if (update_rank < indice_rank - 1) {
		NCNN_LOGE("Error: update_rank < indice_rank -1 ");
		return -1;
	}

	for (int i = 0; i < indice_rank - 1; ++i) {
		if (indices.dimsVec[i] != update.dimsVec[i]) {
			NCNN_LOGE("Error: indices_dims and update dims not equal before index indice_rank -1");
			return -1;
		}
	}


	if (dims_count(update.dimsVec, update.dims, indice_rank - 1) != dims_count(input.dimsVec, input.dims, last_indice_dimension)) {
		NCNN_LOGE("Error: indices_dims and update dims not equal before index indice_rank -1");
		return -1;
	}

	std::vector<int> element_counts(last_indice_dimension);
	for (int i = 0; i < last_indice_dimension; ++i) {
		element_counts[i] = dims_count(input.dimsVec, input.dims, i + 1);
	}

	int element_to_copy = dims_count(input.dimsVec, input.dims, last_indice_dimension);
	int offset_count = dims_count(indices.dimsVec, indices.dims, 0, indice_rank - 1);


	for (int i = 0; i < offset_count; ++i) {
		int offset = 0;
		for (int j = 0; j < last_indice_dimension; ++j) {
			auto indice = *(pIndex + i * last_indice_dimension + j);
			offset += (int)indice * element_counts[j];
		}
		float *pOut = output_data + offset;
		float *pUpdata = update_data + i * element_to_copy;

		switch (op)
		{
		case 0: //eq
		{
			memcpy(pOut, pUpdata, element_to_copy * sizeof(float));
		}
		    break;
		case 1: //add
		{
			for (int k = 0; k < element_to_copy; k++)
				pOut[k] += pUpdata[k];
		}
		    break;
		case 2: //mul
		{
			for (int k = 0; k < element_to_copy; k++)
				pOut[k] *= pUpdata[k];
		}
		    break;
		case 3: //max
		{
			for (int k = 0; k < element_to_copy; k++)
				pOut[k] = (std::fmaxf)(pOut[k], pUpdata[k]);
		}
		    break;
		case 4: //min
		{
			for (int k = 0; k < element_to_copy; k++)
				pOut[k] = (std::fminf)(pOut[k], pUpdata[k]);
		}
		    break;
		default:
			break;
		}
	}


	return 0;
}

#if 0
int ScatterND::forward(const Mat &input, const Mat &indices, const Mat &update, Mat &output)const
{
	output = input.clone();


	float* input_data = (float*)input.data;
	float* update_data = (float*)update.data;
	float* output_data = (float*)output.data;
	float* pIndex = (float*)indices.data;


	//for (int i = 0; i < indices.h * indices.w * indices.c; i++)
	//	printf("%f  ", pIndex[i]);

	int indice_rank = indices.dims;
	int last_indice_dimension = indices.dimsVec[indice_rank - 1];
	if (last_indice_dimension > input.dims) {
		NCNN_LOGE("Error: last dimension of indices larger than input blob dims size ");
		return -1;
	}

	int update_rank = update.dims;
	if (update_rank < indice_rank - 1) {
		NCNN_LOGE("Error: update_rank < indice_rank -1 ");
		return -1;
	}

	for (int i = 0; i < indice_rank - 1; ++i) {
		if (indices.dimsVec[i] != update.dimsVec[i]) {
			NCNN_LOGE("Error: indices_dims and update dims not equal before index indice_rank -1");
			return -1;
		}
	}
#if 0
	std::vector<int> input_dims(input.dimsVec, input.dimsVec + input.dims);
	std::vector<int> indices_dims(indices.dimsVec, indices.dimsVec + indices.dims);
	std::vector<int> update_dims(update.dimsVec, update.dimsVec + update.dims);

	if (count(update_dims, indice_rank - 1) != count(input_dims, last_indice_dimension)) {
		NCNN_LOGE("Error: indices_dims and update dims not equal before index indice_rank -1");
		return -1;
	}


	std::vector<int> element_counts(last_indice_dimension);
	for (int i = 0; i < last_indice_dimension; ++i) {
		element_counts[i] = count(input_dims, i + 1);
	}

	int element_to_copy = count(input_dims, last_indice_dimension);
	int offset_count = count(indices_dims, 0, indice_rank - 1);


#else if
	if (dims_count(update.dimsVec, update.dims, indice_rank - 1) != dims_count(input.dimsVec, input.dims, last_indice_dimension)) {
		NCNN_LOGE("Error: indices_dims and update dims not equal before index indice_rank -1");
		return -1;
	}

	std::vector<int> element_counts(last_indice_dimension);
	for (int i = 0; i < last_indice_dimension; ++i) {
		element_counts[i] = dims_count(input.dimsVec, input.dims, i + 1);
	}

	int element_to_copy = dims_count(input.dimsVec, input.dims, last_indice_dimension);
	int offset_count = dims_count(indices.dimsVec, indices.dims, 0, indice_rank - 1);
#endif

	for (int i = 0; i < offset_count; ++i) {
		int offset = 0;
		for (int j = 0; j < last_indice_dimension; ++j) {
			auto indice = *(pIndex + i * last_indice_dimension + j);
			offset += (int)indice * element_counts[j];
		}
		memcpy(output_data + offset, update_data + i * element_to_copy, element_to_copy * sizeof(float));
	}


	return 0;
}

#endif
}  // namespace ncnn

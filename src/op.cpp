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

#include "op.h"
#include <string>
#include <iostream>


int einsum(std::vector<ncnn::Mat>& result, const std::vector<ncnn::Mat>& input, const std::string& equation)
{
	int ret;
	ncnn::Einsum einsumLayer;
	ret = einsumLayer.load_param(equation);
	//std::vector<ncnn::Mat> b;
	result.resize(1);

	ret = einsumLayer.forward(input, result);
	return ret;
}

int scatterND(ncnn::Mat &result, const ncnn::Mat &input, const ncnn::Mat &indices, const ncnn::Mat &update, int op)
{
	ncnn::ScatterND scatterND;

	scatterND.op = op;

	scatterND.forward(input, indices, update, result);
	return 0;
}

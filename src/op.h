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

#ifndef OPERATOR_H
#define OPERATOR_H


#include "einsum.h"
#include "scatterND.h"
#include "mat.h"

#include <stdio.h>
#include <vector>



int einsum(std::vector<ncnn::Mat>& result, const std::vector<ncnn::Mat>& input, const std::string& equation);
int scatterND(ncnn::Mat &result, const ncnn::Mat &input, const ncnn::Mat &indices, const ncnn::Mat &update, int op);

#endif // OPERATOR_H

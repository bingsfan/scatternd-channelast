// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "mat.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include <math.h>
int count(std::vector<int> &dims, int start_index, int end_index) 
{
	if (-1 == end_index || end_index > dims.size()) {
		end_index = static_cast<int>(dims.size());
	}

	int result = 1;
	for (int index = start_index; index < end_index; ++index) {
		result *= dims[index];
	}
	return result;
}
int dims_count(int* dimsVec, int dims, int start_index, int end_index)
{
	if (-1 == end_index || end_index > dims) {
		end_index = static_cast<int>(dims);
	}

	int result = 1;
	for (int index = start_index; index < end_index; ++index) {
		result *= dimsVec[index];
	}
	return result;
}


namespace ncnn {

Mat Mat::clone(Allocator* _allocator) const
{
    if (empty())
        return Mat();

    Mat m;
    if (dims == 1)
        m.create(w, elemsize, elempack, _allocator);
    else if (dims == 2)
        m.create(w, h, elemsize, elempack, _allocator);
    else if (dims == 3)
        m.create(w, h, c, elemsize, elempack, _allocator);
    else if (dims == 4)
        m.create(w, h, d, c, elemsize, elempack, _allocator);

    if (total() > 0)
    {
        if (cstep == m.cstep)
            memcpy(m.data, data, total() * elemsize);
        else
        {
            // copy by channel for differnet cstep
            size_t size = (size_t)w * h * d * elemsize;
            for (int i = 0; i < c; i++)
            {
                memcpy(m.channel(i), channel(i), size);
            }
        }
    }

    return m;
}

void Mat::clone_from(const ncnn::Mat& mat, Allocator* allocator)
{
    *this = mat.clone(allocator);
}

Mat Mat::reshape(int _w, Allocator* _allocator) const
{
    if (w * h * d * c != _w)
        return Mat();

    if (dims >= 3 && cstep != (size_t)w * h * d)
    {
        Mat m;
        m.create(_w, elemsize, elempack, _allocator);

        // flatten
        for (int i = 0; i < c; i++)
        {
            const void* ptr = (unsigned char*)data + i * cstep * elemsize;
            void* mptr = (unsigned char*)m.data + (size_t)i * w * h * d * elemsize;
            memcpy(mptr, ptr, (size_t)w * h * d * elemsize);
        }

        return m;
    }

    Mat m = *this;

    m.dims = 1;
    m.w = _w;
    m.h = 1;
    m.d = 1;
    m.c = 1;
	m.dimsVec[0] = _w; //sclu

    m.cstep = _w;

    return m;
}

Mat Mat::reshape(int _w, int _h, Allocator* _allocator) const
{
    if (w * h * d * c != _w * _h)
        return Mat();

    if (dims >= 3 && cstep != (size_t)w * h * d)
    {
        Mat m;
        m.create(_w, _h, elemsize, elempack, _allocator);

        // flatten
        for (int i = 0; i < c; i++)
        {
            const void* ptr = (unsigned char*)data + i * cstep * elemsize;
            void* mptr = (unsigned char*)m.data + (size_t)i * w * h * d * elemsize;
            memcpy(mptr, ptr, (size_t)w * h * d * elemsize);
        }

        return m;
    }

    Mat m = *this;

    m.dims = 2;
    m.w = _w;
    m.h = _h;
    m.d = 1;
    m.c = 1;
	m.dimsVec[1] = _w; //sclu
	m.dimsVec[0] = _h; //sclu

    m.cstep = (size_t)_w * _h;

    return m;
}

Mat Mat::reshape(int _w, int _h, int _c, Allocator* _allocator) const
{
    if (w * h * d * c != _w * _h * _c)
        return Mat();

    if (dims < 3)
    {
        if ((size_t)_w * _h != alignSize((size_t)_w * _h * elemsize, 16) / elemsize)
        {
            Mat m;
            m.create(_w, _h, _c, elemsize, elempack, _allocator);

            // align channel
            for (int i = 0; i < _c; i++)
            {
                const void* ptr = (unsigned char*)data + (size_t)i * _w * _h * elemsize;
                void* mptr = (unsigned char*)m.data + i * m.cstep * m.elemsize;
                memcpy(mptr, ptr, (size_t)_w * _h * elemsize);
            }

            return m;
        }
    }
    else if (c != _c)
    {
        // flatten and then align
        Mat tmp = reshape(_w * _h * _c, _allocator);
        return tmp.reshape(_w, _h, _c, _allocator);
    }

    Mat m = *this;

    m.dims = 3;
    m.w = _w;
    m.h = _h;
    m.d = 1;
    m.c = _c;
	m.dimsVec[0] = _c; //sclu
	m.dimsVec[1] = _h; //sclu
	m.dimsVec[2] = _w; //sclu

    m.cstep = alignSize((size_t)_w * _h * elemsize, 16) / elemsize;

    return m;
}

Mat Mat::reshape(int _w, int _h, int _d, int _c, Allocator* _allocator) const
{
    if (w * h * d * c != _w * _h * _d * _c)
        return Mat();

    if (dims < 3)
    {
        if ((size_t)_w * _h * _d != alignSize((size_t)_w * _h * _d * elemsize, 16) / elemsize)
        {
            Mat m;
            m.create(_w, _h, _d, _c, elemsize, elempack, _allocator);

            // align channel
            for (int i = 0; i < _c; i++)
            {
                const void* ptr = (unsigned char*)data + (size_t)i * _w * _h * _d * elemsize;
                void* mptr = (unsigned char*)m.data + i * m.cstep * m.elemsize;
                memcpy(mptr, ptr, (size_t)_w * _h * _d * elemsize);
            }

            return m;
        }
    }
    else if (c != _c)
    {
        // flatten and then align
        Mat tmp = reshape(_w * _h * _d * _c, _allocator);
        return tmp.reshape(_w, _h, _d, _c, _allocator);
    }

    Mat m = *this;

    m.dims = 4;
    m.w = _w;
    m.h = _h;
    m.d = _d;
    m.c = _c;
	m.dimsVec[0] = _c; //sclu
	m.dimsVec[1] = _d; //sclu
	m.dimsVec[2] = _h; //sclu
	m.dimsVec[3] = _w; //sclu

    m.cstep = alignSize((size_t)_w * _h * _d * elemsize, 16) / elemsize;

    return m;
}

void Mat::create(int _w, size_t _elemsize, Allocator* _allocator)
{
    if (dims == 1 && w == _w && elemsize == _elemsize && elempack == 1 && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = 1;
    allocator = _allocator;

    dims = 1;
    w = _w;
    h = 1;
    d = 1;
    c = 1;
	dimsVec = (int*)malloc(dims * sizeof(int));
	dimsVec[0] = w;  //sclu

    cstep = w;

    size_t totalsize = alignSize(total() * elemsize, 4);
    if (totalsize > 0)
    {
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
    }

    if (data)
    {
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

void Mat::create(int _w, int _h, size_t _elemsize, Allocator* _allocator)
{
    if (dims == 2 && w == _w && h == _h && elemsize == _elemsize && elempack == 1 && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = 1;
    allocator = _allocator;

    dims = 2;
    w = _w;
    h = _h;
    d = 1;
    c = 1;
	dimsVec = (int*)malloc(dims * sizeof(int));
	dimsVec[1] = w;  //sclu
	dimsVec[0] = h;  //sclu

    cstep = (size_t)w * h;

    size_t totalsize = alignSize(total() * elemsize, 4);
    if (totalsize > 0)
    {
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
    }

    if (data)
    {
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

void Mat::create(int _w, int _h, int _c, size_t _elemsize, Allocator* _allocator)
{
    if (dims == 3 && w == _w && h == _h && c == _c && elemsize == _elemsize && elempack == 1 && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = 1;
    allocator = _allocator;

    dims = 3;
    w = _w;
    h = _h;
    d = 1;
    c = _c;
	dimsVec = (int*)malloc(dims * sizeof(int));

	dimsVec[2] = w;  //sclu
	dimsVec[1] = h;  //sclu
	dimsVec[0] = c;  //sclu

    cstep = alignSize((size_t)w * h * elemsize, 16) / elemsize;

    size_t totalsize = alignSize(total() * elemsize, 4);
    if (totalsize > 0)
    {
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
    }

    if (data)
    {
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

void Mat::create(int _w, int _h, int _d, int _c, size_t _elemsize, Allocator* _allocator)
{
    if (dims == 4 && w == _w && h == _h && d == _d && c == _c && elemsize == _elemsize && elempack == 1 && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = 1;
    allocator = _allocator;

    dims = 4;
    w = _w;
    h = _h;
    d = _d;
    c = _c;
	dimsVec = (int*)malloc(dims * sizeof(int));

	dimsVec[3] = w;  //sclu
	dimsVec[2] = h;  //sclu
	dimsVec[1] = d;  //sclu
	dimsVec[0] = c;  //sclu

    cstep = alignSize((size_t)w * h * d * elemsize, 16) / elemsize;

    size_t totalsize = alignSize(total() * elemsize, 4);
    if (totalsize > 0)
    {
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
    }

    if (data)
    {
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

void Mat::create(int _w, size_t _elemsize, int _elempack, Allocator* _allocator)
{
    if (dims == 1 && w == _w && elemsize == _elemsize && elempack == _elempack && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;

    dims = 1;
    w = _w;
    h = 1;
    d = 1;
    c = 1;
	dimsVec = (int*)malloc(dims * sizeof(int));

	dimsVec[0] = w;  //sclu


    cstep = w;

    size_t totalsize = alignSize(total() * elemsize, 4);
    if (totalsize > 0)
    {
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
    }

    if (data)
    {
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

void Mat::create(int _w, int _h, size_t _elemsize, int _elempack, Allocator* _allocator)
{
    if (dims == 2 && w == _w && h == _h && elemsize == _elemsize && elempack == _elempack && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;

    dims = 2;
    w = _w;
    h = _h;
    d = 1;
    c = 1;
	dimsVec = (int*)malloc(dims * sizeof(int));

	dimsVec[1] = w;  //sclu
	dimsVec[0] = h;  //sclu


    cstep = (size_t)w * h;

    size_t totalsize = alignSize(total() * elemsize, 4);
    if (totalsize > 0)
    {
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
    }

    if (data)
    {
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

void Mat::create(int _w, int _h, int _c, size_t _elemsize, int _elempack, Allocator* _allocator)
{
    if (dims == 3 && w == _w && h == _h && c == _c && elemsize == _elemsize && elempack == _elempack && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;

    dims = 3;
    w = _w;
    h = _h;
    d = 1;
    c = _c;
	dimsVec = (int*)malloc(dims * sizeof(int));

	dimsVec[2] = w;  //sclu
	dimsVec[1] = h;  //sclu
	dimsVec[0] = c;  //sclu

    cstep = alignSize((size_t)w * h * elemsize, 16) / elemsize;

    size_t totalsize = alignSize(total() * elemsize, 4);
    if (totalsize > 0)
    {
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
    }

    if (data)
    {
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

void Mat::create(int _w, int _h, int _d, int _c, size_t _elemsize, int _elempack, Allocator* _allocator)
{
    if (dims == 4 && w == _w && h == _h && d == _d && c == _c && elemsize == _elemsize && elempack == _elempack && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;

    dims = 4;
    w = _w;
    h = _h;
    d = _d;
    c = _c;
	dimsVec = (int*)malloc(dims * sizeof(int));

	dimsVec[3] = w;  //sclu
	dimsVec[2] = h;  //sclu
	dimsVec[1] = d;  //sclu
	dimsVec[0] = c;  //sclu

    cstep = alignSize((size_t)w * h * d * elemsize, 16) / elemsize;

    size_t totalsize = alignSize(total() * elemsize, 4);
    if (totalsize > 0)
    {
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
    }

    if (data)
    {
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

void Mat::create_like(const Mat& m, Allocator* _allocator)
{
    int _dims = m.dims;
    if (_dims == 1)
        create(m.w, m.elemsize, m.elempack, _allocator);
    if (_dims == 2)
        create(m.w, m.h, m.elemsize, m.elempack, _allocator);
    if (_dims == 3)
        create(m.w, m.h, m.c, m.elemsize, m.elempack, _allocator);
    if (_dims == 4)
        create(m.w, m.h, m.d, m.c, m.elemsize, m.elempack, _allocator);
}


Mat Mat::from_float16(const unsigned short* data, int size)
{
    Mat m(size);
    if (m.empty())
        return m;

    float* ptr = m; //.data;

#if __ARM_NEON && (__ARM_FP & 2)
    int nn = cpu_support_arm_vfpv4() ? size >> 2 : 0;
    int remain = size - (nn << 2);
#else
    int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON && (__ARM_FP & 2)
#if __aarch64__
    if (nn > 0)
    {
        asm volatile(
            "0:                             \n"
            "ld1    {v0.4h}, [%1], #8       \n"
            "fcvtl  v1.4s, v0.4h            \n"
            "subs   %w0, %w0, #1            \n"
            "st1    {v1.4s}, [%2], #16      \n"
            "bne    0b                      \n"
            : "=r"(nn),   // %0
            "=r"(data), // %1
            "=r"(ptr)   // %2
            : "0"(nn),
            "1"(data),
            "2"(ptr)
            : "cc", "memory", "v0", "v1");
    }
#else
    if (nn > 0)
    {
        asm volatile(
            "0:                             \n"
            "pld        [%1, #64]           \n"
            "vld1.s16   {d0}, [%1]!         \n"
            "vcvt.f32.f16 q1, d0            \n"
            "subs       %0, #1              \n"
            "vst1.f32   {d2-d3}, [%2 :128]! \n"
            "bne        0b                  \n"
            : "=r"(nn),   // %0
            "=r"(data), // %1
            "=r"(ptr)   // %2
            : "0"(nn),
            "1"(data),
            "2"(ptr)
            : "cc", "memory", "q0", "q1");
    }
#endif // __aarch64__
#endif // __ARM_NEON
    for (; remain > 0; remain--)
    {
        *ptr = float16_to_float32(*data);

        data++;
        ptr++;
    }

    return m;
}


unsigned short float32_to_float16(float value)
{
    // 1 : 8 : 23
    union
    {
        unsigned int u;
        float f;
    } tmp;

    tmp.f = value;

    // 1 : 8 : 23
    unsigned short sign = (tmp.u & 0x80000000) >> 31;
    unsigned short exponent = (tmp.u & 0x7F800000) >> 23;
    unsigned int significand = tmp.u & 0x7FFFFF;

    //     NCNN_LOGE("%d %d %d", sign, exponent, significand);

    // 1 : 5 : 10
    unsigned short fp16;
    if (exponent == 0)
    {
        // zero or denormal, always underflow
        fp16 = (sign << 15) | (0x00 << 10) | 0x00;
    }
    else if (exponent == 0xFF)
    {
        // infinity or NaN
        fp16 = (sign << 15) | (0x1F << 10) | (significand ? 0x200 : 0x00);
    }
    else
    {
        // normalized
        short newexp = exponent + (-127 + 15);
        if (newexp >= 31)
        {
            // overflow, return infinity
            fp16 = (sign << 15) | (0x1F << 10) | 0x00;
        }
        else if (newexp <= 0)
        {
            // Some normal fp32 cannot be expressed as normal fp16
            fp16 = (sign << 15) | (0x00 << 10) | 0x00;
        }
        else
        {
            // normal fp16
            fp16 = (sign << 15) | (newexp << 10) | (significand >> 13);
        }
    }

    return fp16;
}

float float16_to_float32(unsigned short value)
{
    // 1 : 5 : 10
    unsigned short sign = (value & 0x8000) >> 15;
    unsigned short exponent = (value & 0x7c00) >> 10;
    unsigned short significand = value & 0x03FF;

    //     NCNN_LOGE("%d %d %d", sign, exponent, significand);

    // 1 : 8 : 23
    union
    {
        unsigned int u;
        float f;
    } tmp;
    if (exponent == 0)
    {
        if (significand == 0)
        {
            // zero
            tmp.u = (sign << 31);
        }
        else
        {
            // denormal
            exponent = 0;
            // find non-zero bit
            while ((significand & 0x200) == 0)
            {
                significand <<= 1;
                exponent++;
            }
            significand <<= 1;
            significand &= 0x3FF;
            tmp.u = (sign << 31) | ((-exponent + (-15 + 127)) << 23) | (significand << 13);
        }
    }
    else if (exponent == 0x1F)
    {
        // infinity or NaN
        tmp.u = (sign << 31) | (0xFF << 23) | (significand << 13);
    }
    else
    {
        // normalized
        tmp.u = (sign << 31) | ((exponent + (-15 + 127)) << 23) | (significand << 13);
    }

    return tmp.f;
}

} // namespace ncnn

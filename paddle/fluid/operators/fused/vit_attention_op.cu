// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <paddle/fluid/platform/device_context.h>
#include <algorithm>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/operators/math/bert_encoder_functor.h"
#include "paddle/phi/kernels/gpudnn/softmax_gpudnn.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/fluid/platform/dynload/cublas.h"
#define FINAL_MASK 0xffffffff

namespace paddle {
namespace operators {

template <typename T>
__inline__ __device__
T warpReduceSum(T val)
{
  for(int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}


template <typename T>
__inline__ __device__
T warpReduceMax(T val)
{
  for(int mask = 16; mask > 0; mask >>= 1)
    val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
  return val;
}


template <typename T>
__inline__ __device__
T blockReduceMax(T val)
{
  static __shared__ T shared[32]; 
//  __shared__ T shared[32]; 
  int lane = threadIdx.x & 0x1f; // in-warp idx
  int wid = threadIdx.x >> 5;  // warp idx

  val = warpReduceMax(val); // get maxx in each warp

  if(lane == 0) // record in-warp maxx by warp Idx
    shared[wid] = val;

  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : 0;
  val = warpReduceMax(val);

  return val;
}


template <typename T>
  __inline__ __device__
T blockReduceSum(T val)
{
  static __shared__ T shared[32]; 
  //__shared__ T shared[32]; 
  int lane = threadIdx.x & 0x1f; 
  int wid = threadIdx.x >> 5;  

  val = warpReduceSum<T>(val);

  if(lane == 0)
    shared[wid] = val;

  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : (T)(0.0f);
  val = warpReduceSum<T>(val);
                              
  return val;
}

template <typename T>
__global__
void set_ptr_kernel(const void** array_a, 
                    const void** array_b, 
                    void** array_c, 
                    T* device_a,
                    T* device_b,
                    T* device_c,
                    int batch_a, int batch_b, int batch_c, 
                    int head_a, int head_b, int head_c){
     int batch_id = blockIdx.x;
     int head_id = threadIdx.x;
     array_a[batch_id * blockDim.x + head_id] = device_a + batch_id * batch_a + head_id * head_a;
     array_b[batch_id * blockDim.x + head_id] = device_b + batch_id * batch_b + head_id * head_b; 
     array_c[batch_id * blockDim.x + head_id] = device_c + batch_id * batch_c + head_id * head_c;
     
}

template <typename T>
__global__
void softmax_kernel(T* qk_buf, T* softmax_buf, const int batch_size, const int head_num, const int length)
{
    int batch_id = blockIdx.x / head_num;
    int head_offset = blockIdx.x % head_num * length;
    int batch_offset = batch_id * length * head_num * length;
    __shared__ float s_sum, s_max;

    for(int i = 0; i < length; ++i)
    {
      int length_offset = i * length * head_num;
      float qk = threadIdx.x < length ? (float)qk_buf[threadIdx.x + batch_offset + head_offset + length_offset]  : 0.0f;
      __syncthreads();
      float tmp = threadIdx.x < length ? (float)(qk): -1e20f;
      float max_val = blockReduceMax<float>(tmp);

      if(threadIdx.x == 0) s_max = max_val;
       __syncthreads();

      qk = threadIdx.x < length ? __expf(tmp - s_max) : 0.0f;

      float sum_val = blockReduceSum<float>(qk);

      if(threadIdx.x == 0) s_sum = sum_val + 1e-6f;
      __syncthreads();

      if(threadIdx.x < length) softmax_buf[threadIdx.x + batch_offset + head_offset + length_offset] = (T)(qk / s_sum);

    }
}

int round_up(int seq_len)
{
    int val =32;
    if(seq_len <= 32)
      val = 32;
    else if(seq_len > 32 && seq_len <= 64)
      val = 64;
    else if(seq_len > 64 && seq_len <= 128)
      val = 128;
    else if(seq_len > 128 && seq_len <= 256)
      val = 256;
    else if(seq_len > 256 && seq_len <= 512)
      val = 512;
    else
      val = 1024;
    return val;
}


template <typename DeviceContext, typename T>
class VitAttentionKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    using Tensor = framework::Tensor;
    //input
    /*
    demo:
    input => {batch, seq_len, hidden_size*3} => [Q,K,V]
    Q=[aaa,bbb,ccc,...yyy,zzz]=>head_number=24,head_size=3,hidden_size=72 
    */
    auto *input = context.Input<framework::Tensor>("Input");
    auto *input_d = input->data<T>();
    auto input_dims = input->dims();
    int batch = input_dims[0];
    int seq_len = input_dims[1];
    int hidden_three = input_dims[2];
    int hidden_size = hidden_three / 3;
    //prepare attr and context
    int head_number = context.Attr<int>("head_number");
    int head_size = hidden_size / head_number;
    float scale = context.Attr<float>("scale");
    auto &device_ctx = context.template device_context<DeviceContext>();
    auto stream = device_ctx.stream();
    // out
    auto *out = context.Output<framework::Tensor>("Out");
    out->Resize({batch, seq_len, head_number*head_size});
    auto *output_d = out->mutable_data<T>(context.GetPlace());
    // prepare tmp tensor(softmax_d)
    Tensor temp_tensor_in;
    temp_tensor_in.Resize({batch,head_number,seq_len,seq_len});
    auto *temp_softmax_d_in = temp_tensor_in.mutable_data<T>(context.GetPlace());
    //Tensor temp_tensor_out;
    //temp_tensor_out.Resize({batch, head_number, seq_len, seq_len});
    //auto *temp_softmax_d_out = temp_tensor_out.mutable_data<T>(context.GetPlace());
    // qkv ptr
    auto *input_q_d = const_cast<T*>(input_d + hidden_size * 0);
    auto *input_k_d = const_cast<T*>(input_d + hidden_size * 1);
    auto *input_v_d = const_cast<T*>(input_d + hidden_size * 2); 

    // compute q * k
    int batch_count = batch * head_number;

    const void **d_a_array, **d_b_array, **array;
    void **d_c_array;
    cudaMalloc((void**)&array, 3 * batch_count * sizeof(T *));
    d_a_array = array;
    d_b_array = &array[batch_count];
    d_c_array = const_cast<void**>(&array[2 * batch_count]);
    // set_ptr_kernel
    dim3 grid_ptr(batch);
    dim3 block_ptr(head_number);
    set_ptr_kernel<<<grid_ptr,block_ptr,0,stream>>>(d_a_array, 
                                                d_b_array,
                                                d_c_array,
                                                input_q_d,
                                                input_k_d,
                                                temp_softmax_d_in,
                                                seq_len * hidden_three,
                                                seq_len * hidden_three,
                                                seq_len * seq_len * head_number, 
                                                head_size, 
                                                head_size, 
                                                seq_len);
    auto alpha = (T)scale;
    auto beta = (T)0.0f;
    int lda = hidden_three;
    int ldb = hidden_three;
    int ldc = seq_len * head_number;


    auto blas = phi::funcs::GetBlas<platform::CUDADeviceContext, T>(device_ctx);
    blas.BatchedGemmArray(CblasTrans, 
                     CblasNoTrans, 
                     seq_len,
                     seq_len,
                     head_size,
                     alpha,
                     d_b_array, ldb,
                     d_a_array, lda,
                     beta, 
                     d_c_array, ldc,
                     batch_count);
    // softmax 
    //dim3 grid(batch * head_number);
    //dim3 block;
    //int val = round_up(seq_len);
    //block.x = val;
    //softmax_kernel<T> <<<grid, block, 0, stream>>>(temp_softmax_d_in, temp_softmax_d_in, batch, head_number, seq_len);
    phi::SoftmaxForwardCUDAKernelDriver<T>(device_ctx, temp_tensor_in, -1, &temp_tensor_in);
    // softmax * v
    set_ptr_kernel<<<grid_ptr,block_ptr,0,stream>>>(d_a_array, 
                                                d_b_array,
                                                d_c_array,
                                                temp_softmax_d_in,
                                                input_v_d,
                                                output_d,
                                                seq_len * seq_len * head_number,
                                                seq_len * hidden_three,
                                                seq_len * hidden_size, 
                                                seq_len, 
                                                head_size, 
                                                head_size);

    alpha = (T)1.0f;
    lda = seq_len * head_number;
    ldb = hidden_three;
    ldc = hidden_size;

    blas.BatchedGemmArray(CblasNoTrans, 
                     CblasNoTrans, 
                     head_size,
                     seq_len,
                     seq_len,
                     alpha,
                     d_b_array, ldb,
                     d_a_array, lda,
                     beta, 
                     d_c_array, ldc,
                     batch_count);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    vit_attention,
    ops::VitAttentionKernel<paddle::platform::CUDADeviceContext, float>,
    ops::VitAttentionKernel<paddle::platform::CUDADeviceContext, paddle::platform::float16>);

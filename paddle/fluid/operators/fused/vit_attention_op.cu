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
    Tensor temp_tensor;
    temp_tensor.Resize({batch * head_number * seq_len * seq_len});
    auto *temp_softmax_d = temp_tensor.mutable_data<T>(context.GetPlace());
    // qkv ptr
    auto *input_q_d = const_cast<T*>(input_d + hidden_size * 0);
    auto *input_k_d = const_cast<T*>(input_d + hidden_size * 1);
    auto *input_v_d = const_cast<T*>(input_d + hidden_size * 2); 

    // compute q * k
    int batch_count = batch * head_number;
    // // host
    T** h_a_array = new T*[batch_count];
    T** h_b_array = new T*[batch_count];
    T** h_c_array = new T*[batch_count];
    // // device
    const void **d_a_array, **d_b_array;
    void **d_c_array;

    cudaMalloc((void**)&d_a_array, batch_count * sizeof(T *));
    cudaMalloc((void**)&d_b_array, batch_count * sizeof(T *));
    cudaMalloc((void**)&d_c_array, batch_count * sizeof(T *));
    // // set batch ptr
    for(int i=0; i<batch; ++i)
    {
	     for(int j=0; j<head_number; j++)
	     {
    		h_a_array[i * head_number + j] = input_q_d + i * seq_len * hidden_three + j * head_size;
    		h_b_array[i * head_number + j] = input_k_d + i * seq_len * hidden_three + j * head_size;
    		h_c_array[i * head_number + j] = temp_softmax_d + i * seq_len * (seq_len * head_number) + j * seq_len;
	     }
    }
    // // copy host to device
    cudaMemcpy(d_a_array, h_a_array, batch_count * sizeof(T*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_array, h_b_array, batch_count * sizeof(T*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c_array, h_c_array, batch_count * sizeof(T*), cudaMemcpyHostToDevice);

    auto alpha = 1.25f;
    auto beta = 0.0f;
    int lda = hidden_three;
    int ldb = hidden_three;
    int ldc = seq_len * head_number;


    auto blas = phi::funcs::GetBlas<platform::CUDADeviceContext, T>(device_ctx);
    // CBLAS_TRANSPOSE transA = CblasTrans;
    // CBLAS_TRANSPOSE transB = CblasNoTrans;
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
    /*
    blasHandle_t handle{nullptr};
    paddle::platform::dynload::cublasCreate_v2(&handle);
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DFALT;
    paddle::platform::dynload::cublasGemmBatchedEx(handle,
			CUBLAS_OP_T,
			CUBLAS_OP_N,
			seq_len,
			seq_len,
			head_size,
			&alpha,
		        d_b_array, CUDA_R_32F, ldb,
			d_a_array, CUDA_R_32F, lda,
			&beta,
			d_c_array, CUDA_R_32F, ldc,
			batch_count,
			CUDA_R_32F,
			algo);
    */
    // softmax 
    dim3 grid(batch * head_number);
    dim3 block;
    if(seq_len <= 32)
      block.x = 32;
    else if(seq_len > 32 && seq_len <= 64)
      block.x = 64;
    else if(seq_len > 64 && seq_len <= 128)
      block.x = 128;
    else if(seq_len > 128 && seq_len <= 256)
      block.x = 256;
    else if(seq_len > 256 && seq_len <= 512)
      block.x = 512;
    else
      block.x = 1024;
    softmax_kernel<T> <<<grid, block, 0, stream>>>(temp_softmax_d, temp_softmax_d, batch, head_number, seq_len);
    // softmax * v
    // cudaMemcpy(output_d, temp_softmax_d, batch * seq_len * seq_len * head_number * sizeof(T), cudaMemcpyDeviceToDevice);
    // // set batch ptr
    for(int i=0; i<batch; ++i)
    {
	     for(int j=0; j<head_number; j++)
	     {
    		h_a_array[i * head_number + j] = temp_softmax_d + i * seq_len * seq_len * head_number + j * seq_len;
    		h_b_array[i * head_number + j] = input_v_d + i * seq_len * hidden_three + j * head_size;
    		h_c_array[i * head_number + j] = output_d + i * seq_len * hidden_size + j * head_size;
	     }
    }
    // // copy host to device
    cudaMemcpy(d_a_array, h_a_array, batch_count * sizeof(T*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_array, h_b_array, batch_count * sizeof(T*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c_array, h_c_array, batch_count * sizeof(T*), cudaMemcpyHostToDevice);

    alpha = 1.0f;
    beta = 0.0f;
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
    /*
    paddle::platform::dynload::cublasGemmBatchedEx(handle,
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			head_size,
			seq_len,
			seq_len,
			&alpha,
		        d_b_array, CUDA_R_32F, ldb,
			d_a_array, CUDA_R_32F, lda,
			&beta,
			d_c_array, CUDA_R_32F, ldc,
			batch_count,
			CUDA_R_32F,
			algo);
    */
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    vit_attention,
    ops::VitAttentionKernel<paddle::platform::CUDADeviceContext, float>);

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
#pragma once

#include <glog/logging.h>
#include <mlir/IR/Builders.h>
#include <mlir/Transforms/DialectConversion.h>

#include "paddle/infrt/dialect/infrt/common/types.h"
#include "paddle/infrt/dialect/infrt/ir/infrt_dialect.h"
#include "paddle/infrt/dialect/pd/ir/pd_ops.h"
#include "paddle/infrt/dialect/phi/ir/infrt_phi_tensor.h"
#include "paddle/infrt/dialect/tensorrt/trt_ops.h"

namespace infrt {
namespace trt {
static mlir::Value createTRTConv2dOp(mlir::PatternRewriter &rewriter,  // NOLINT
                                     mlir::Operation *op) {
  auto conv_op = ::llvm::dyn_cast<infrt::pd::Conv2dOp>(op);
  ::mlir::SmallVector<::mlir::Value, 4> operands;
  ::mlir::Operation::operand_range Input = conv_op.getODSOperands(0);
  ::mlir::Operation::operand_range Filter = conv_op.getODSOperands(1);
  operands.push_back((*Input.begin()));
  operands.push_back((*Filter.begin()));

  ::mlir::SmallVector<::mlir::Type, 4> resultTypes;
  for (auto v : conv_op.getODSResults(0)) {
    resultTypes.push_back(v.getType());
  }
  ::mlir::SmallVector<::mlir::NamedAttribute, 8> attributes;
  {
    // TODO(weishengying) :  get out_channel_num for filter shape
    auto tblgen_attr = rewriter.getSI32IntegerAttr(3);
    attributes.emplace_back(rewriter.getStringAttr("out_channel_num"),
                            tblgen_attr);
  }
  {
    // TODO(weishengying) :  get kernel_size for filter shape
    auto tblgen_attr = rewriter.getI32ArrayAttr({3, 3});
    attributes.emplace_back(rewriter.getStringAttr("kernel_size"), tblgen_attr);
  }
  {
    auto tblgen_attr = op->getAttrOfType<::mlir::ArrayAttr>("strides");
    attributes.emplace_back(rewriter.getStringAttr("strides"), tblgen_attr);
  }
  {
    auto tblgen_attr = op->getAttrOfType<::mlir::ArrayAttr>("paddings");
    attributes.emplace_back(rewriter.getStringAttr("paddings"), tblgen_attr);
  }
  {
    auto tblgen_attr =
        op->getAttrOfType<::mlir::StringAttr>("padding_algorithm");
    attributes.emplace_back(rewriter.getStringAttr("padding_mode"),
                            tblgen_attr);
  }
  {
    auto tblgen_attr = op->getAttrOfType<::mlir::IntegerAttr>("groups");
    attributes.emplace_back(rewriter.getStringAttr("groups"), tblgen_attr);
  }
  {
    auto tblgen_attr = op->getAttrOfType<::mlir::ArrayAttr>("dilations");
    attributes.emplace_back(rewriter.getStringAttr("dilations"), tblgen_attr);
  }
  {
    auto tblgen_attr = op->getAttrOfType<::mlir::StringAttr>("data_format");
    attributes.emplace_back(rewriter.getStringAttr("data_format"), tblgen_attr);
  }
  return rewriter.create<trt::ConvolutionOp>(
      op->getLoc(), resultTypes, operands, attributes);
}

static inline mlir::ArrayAttr TransposeWeight(
    mlir::PatternRewriter &builder,  // NOLINT
    const mlir::ArrayAttr &weight,
    const mlir::ArrayAttr &dims) {
  CHECK_EQ(dims.size(), 2U);
  CHECK(!dims.empty());
  CHECK(dims[0].getType().isInteger(64));
  CHECK(!weight.empty());
  CHECK(weight[0].getType().isF32());

  int row = dims[0].cast<mlir::IntegerAttr>().getInt();
  int col = dims[1].cast<mlir::IntegerAttr>().getInt();
  std::vector<float> trans_weight(weight.size());
  for (int i = 0; i < row; ++i) {
    for (int j = 0; j < col; ++j) {
      trans_weight[j * row + i] =
          weight[i * col + j].cast<mlir::FloatAttr>().getValueAsDouble();
    }
  }
  return builder.getF32ArrayAttr(trans_weight);
}

// matmul_y and elt_y is weights.
inline ::llvm::SmallVector<::mlir::Value, 4> createTrtFcOp(
    mlir::PatternRewriter &builder,  // NOLINT
    mlir::Value matmul_x,
    mlir::Value matmul_y,
    mlir::Value elt_y,
    mlir::Value elt_out) {
  ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

  auto *y_producer = matmul_y.getDefiningOp();
  auto create_inited_tensor_op =
      llvm::dyn_cast<::infrt::phi::CreateHostInitedDenseTensorOp>(y_producer);
  CHECK_NOTNULL(create_inited_tensor_op);

  mlir::ArrayAttr dims = create_inited_tensor_op.dims();
  CHECK_EQ(dims.size(), 2U);

  std::vector<int64_t> new_dims(dims.size());
  CHECK(!dims.empty());
  CHECK(dims[0].getType().isIntOrIndex());
  for (size_t i = 0; i < new_dims.size(); ++i) {
    new_dims[i] = dims[dims.size() - 1 - i].cast<mlir::IntegerAttr>().getInt();
  }
  auto insert_point = builder.saveInsertionPoint();
  builder.setInsertionPoint(create_inited_tensor_op);
  auto new_inited_op =
      builder.create<::infrt::phi::CreateHostInitedDenseTensorOp>(
          create_inited_tensor_op->getLoc(),
          create_inited_tensor_op.output().getType(),
          create_inited_tensor_op.context(),
          builder.getI64ArrayAttr(new_dims),
          ::infrt::LayoutAttr::get(builder.getContext(),
                                   ::infrt::LayoutType::NCHW),
          create_inited_tensor_op.lod(),
          TransposeWeight(builder, create_inited_tensor_op.values(), dims));
  builder.replaceOp(create_inited_tensor_op, new_inited_op->getResults());
  builder.restoreInsertionPoint(insert_point);

  auto ods_loc = builder.getFusedLoc({y_producer->getLoc()});
  ::infrt::trt::FullyConnectedOp fc_op;
  {
    ::mlir::SmallVector<::mlir::Type, 4> tblgen_types;

    fc_op = builder.create<::infrt::trt::FullyConnectedOp>(
        ods_loc,
        elt_out.getType(),
        matmul_x,
        new_inited_op.output(),
        elt_y,
        builder.getSI32IntegerAttr(new_dims[0]));
  }

  ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
  for (auto v : ::llvm::SmallVector<::mlir::Value, 4>{fc_op.getODSResults(0)}) {
    tblgen_repl_values.push_back(v);
  }
  return tblgen_repl_values;
}

static mlir::Value createTRTShuffledOp(
    mlir::PatternRewriter &rewriter,  // NOLINT
    mlir::Operation *op,
    const mlir::Value &input,
    const mlir::Attribute &start,
    const mlir::Attribute &stop) {
  auto flatten_op = ::llvm::dyn_cast<infrt::pd::Flatten_contiguous_rangeOp>(op);
  ::mlir::SmallVector<::mlir::Value, 4> operands;
  operands.push_back(input);

  ::mlir::SmallVector<::mlir::Type, 4> resultTypes;
  for (auto v : flatten_op.getODSResults(0)) {
    resultTypes.push_back(v.getType());
  }

  ::mlir::SmallVector<::mlir::NamedAttribute, 8> attributes;
  mlir::IntegerAttr start_attr = start.dyn_cast<mlir::IntegerAttr>();
  mlir::IntegerAttr stop_attr = stop.dyn_cast<mlir::IntegerAttr>();

  int start_axis = start_attr.getSInt();
  int stop_axis = stop_attr.getSInt();
  // TODO(weishengying) : get dim form DenseTonsor
  int dims = 4;
  // TODO(weishengying) : get input_dims form DenseTonsor
  int input_dims[4] = {1, 2048, 1, 1};
  int dim_prod = 1;

  std::vector<int> flatten_dim(dims - (stop_axis - start_axis));
  for (int i = 0, j = 0; i < dims; ++i) {
    if (start_axis <= i + 1 && i + 1 <= stop_axis) {
      int dim_i = input_dims[i];
      dim_prod *= dim_i;
      if (i + 1 == stop_axis) {
        flatten_dim[j++] = dim_prod;
      }
    } else {
      flatten_dim[j++] = input_dims[i];
    }
  }
  auto reshape_arrt = rewriter.getI32ArrayAttr(flatten_dim);
  attributes.emplace_back(rewriter.getStringAttr("reshape"), reshape_arrt);
  return rewriter.create<trt::ShuffleOp>(
      op->getLoc(), resultTypes, operands, attributes);
}
}  // namespace trt
}  // namespace infrt

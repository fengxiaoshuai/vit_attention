// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/vit_attention_fuse_pass.h"

#include <string>
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

#define GET_IR_NODE(node__) GET_IR_NODE_FROM_SUBGRAPH(node__, node__, pattern);
#define GET_NODES                    \
  GET_IR_NODE(reshape1_op);          \
  GET_IR_NODE(transpose1_op);        \
  GET_IR_NODE(slice1_op);            \
  GET_IR_NODE(slice2_op);            \
  GET_IR_NODE(slice3_op);            \
  GET_IR_NODE(matmul1_op);           \
  GET_IR_NODE(scale1_op);            \
  GET_IR_NODE(transpose2_op);        \
  GET_IR_NODE(softmax1_op);          \
  GET_IR_NODE(matmul2_op);           \
  GET_IR_NODE(transpose3_op);        \
  GET_IR_NODE(reshape2_op);        


namespace paddle {
namespace framework {
namespace ir {

void VitAttentionFusePass::ApplyImpl(ir::Graph* graph) const {
  GraphPatternDetector gpd;
  const std::string pattern_name = "vit_attention_fuse";
  FusePassBase::Init(pattern_name, graph);

  // pattern
  PDNode* x = gpd.mutable_pattern()->NewNode("x")
                                   ->assert_is_op_input("reshape2", "X")
                                   ->AsInput();
  patterns::VitAttention pattern(gpd.mutable_pattern(), pattern_name);
  pattern(x);

  int fusion_count=0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph, Graph* g) 
  {
    GET_NODES;
    ++fusion_count;
  };
  gpd(graph, handler);
  AddStatis(fusion_count);


}


}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(vit_attention_fuse_pass, paddle::framework::ir::VitAttentionFusePass);
REGISTER_PASS_CAPABILITY(vit_attention_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .GE("reshape2", 0)
            .GE("transpose2", 0)
            .GE("slice", 0)
            .GE("scale", 0)
            .GE("softmax", 0)
            .GE("matmul_v2", 0));


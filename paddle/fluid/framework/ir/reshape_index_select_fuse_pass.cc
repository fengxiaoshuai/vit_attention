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

#include "paddle/fluid/framework/ir/reshape_index_select_fuse_pass.h"

#include <string>
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

#define GET_IR_NODE(node__) GET_IR_NODE_FROM_SUBGRAPH(node__, node__, pattern);
#define GET_NODES                    \
  GET_IR_NODE(reshape1_op);          \
  GET_IR_NODE(index_select_op);      \
  GET_IR_NODE(index_select_w);       \
  GET_IR_NODE(reshape2_op);          \
  GET_IR_NODE(transpose_op);         \
  GET_IR_NODE(unsqueeze2_op);        \
  GET_IR_NODE(unsqueeze2_out);  


namespace paddle {
namespace framework {
namespace ir {

void ReshapeIndexSelectFusePass::ApplyImpl(ir::Graph* graph) const {
  GraphPatternDetector gpd;
  const std::string pattern_name = "reshape_index_select_fuse";
  FusePassBase::Init(pattern_name, graph);

  // pattern
  PDNode* x = gpd.mutable_pattern()->NewNode("x")
                                   ->assert_is_op_input("reshape2", "X")
                                   ->AsInput();
  patterns::ReshapeIndexSelect pattern(gpd.mutable_pattern(), pattern_name);
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

REGISTER_PASS(reshape_index_select_fuse_pass, paddle::framework::ir::ReshapeIndexSelectFusePass);
REGISTER_PASS_CAPABILITY(reshape_index_select_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .GE("reshape2", 0)
            .GE("transpose2", 0)
            .GE("index_select", 0)
            .GE("unsqueeze2", 0));


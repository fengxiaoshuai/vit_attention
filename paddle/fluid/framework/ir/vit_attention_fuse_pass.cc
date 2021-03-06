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
#define GET_NODES                     \
  GET_IR_NODE(reshape1_op);           \
  GET_IR_NODE(reshape1_out);          \
  GET_IR_NODE(transpose1_op);         \
  GET_IR_NODE(transpose1_out);        \
  GET_IR_NODE(slice1_op);             \
  GET_IR_NODE(slice1_out);            \
  GET_IR_NODE(slice2_op);             \
  GET_IR_NODE(slice2_out);            \
  GET_IR_NODE(slice3_op);             \
  GET_IR_NODE(slice3_out);            \
  GET_IR_NODE(matmul1_op);            \
  GET_IR_NODE(matmul1_out);           \
  GET_IR_NODE(scale1_op);             \
  GET_IR_NODE(scale1_out);            \
  GET_IR_NODE(transpose2_op);         \
  GET_IR_NODE(transpose2_out);        \
  GET_IR_NODE(softmax1_op);           \
  GET_IR_NODE(softmax1_out);          \
  GET_IR_NODE(matmul2_op);            \
  GET_IR_NODE(matmul2_out);           \
  GET_IR_NODE(transpose3_op);         \
  GET_IR_NODE(transpose3_out);        \
  GET_IR_NODE(reshape2_op);           \
  GET_IR_NODE(reshape2_out);        


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
    //do something;
    OpDesc desc(matmul1_op->Op()->Block());
    desc.SetType("vit_attention");
    desc.SetInput("Input", {subgraph.at(x)->Name()});
    desc.SetOutput("Out", {reshape2_out->Name()});
    std::vector<int64_t> shape = softmax1_out->Var()->GetShape();
    desc.SetAttr("head_number", int(shape[1]));
    desc.SetAttr("scale", scale1_op->Op()->GetAttr("scale"));

    // Create a new node for the fused op.
    auto vit_attention_node = graph->CreateOpNode(&desc);

    // Link inputs and outputs.
    PADDLE_ENFORCE_NE(subgraph.count(x), 0, platform::errors::NotFound("Detector did not find input x of vit_attention."));

    IR_NODE_LINK_TO(subgraph.at(x), vit_attention_node);          // Input
    IR_NODE_LINK_TO(vit_attention_node, reshape2_out);                        // Output

    // Delete the unneeded nodes.
    std::unordered_set<const Node*> marked_nodes({reshape1_op,
                                                  reshape1_out,
                                                  transpose1_op,
                                                  transpose1_out,
                                                  slice1_op,
						  slice1_out,
						  slice2_op,
						  slice2_out,
						  slice3_op,
						  slice3_out,
						  matmul1_op,
						  matmul1_out,
						  scale1_op,
						  scale1_out,
						  transpose2_op,
						  transpose2_out,
						  softmax1_op,
						  softmax1_out,
						  matmul2_op,
						  matmul2_out,
						  transpose3_op,
						  transpose3_out,
						  reshape2_op});

					          
                                             
    GraphSafeRemoveNodes(graph, marked_nodes);
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


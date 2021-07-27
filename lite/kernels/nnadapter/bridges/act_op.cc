// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/nnadapter/bridges/converter.h"
#include "lite/kernels/nnadapter/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace nnadapter {

int ActConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto converter = static_cast<Converter*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "Converting " << op_type << " ...";

  // Get input and output vars and op attributes
  auto x_name = op_info->Input("X").front();
  auto x_scale_name = "X0_scale";
  auto has_x_scale = op_info->HasInputScale(x_scale_name, true);
  auto x_scale =
      has_x_scale ? op_info->GetInputScale(x_scale_name, true)[0] : 0.f;
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();
  auto out_name = op_info->Output("Out").front();
  auto out_scale_name = "Out0_scale";
  auto has_out_scale = op_info->HasOutputScale(out_scale_name, true);
  auto out_scale =
      has_out_scale ? op_info->GetOutputScale(out_scale_name, true)[0] : 0.f;
  auto out = scope->FindMutableTensor(out_name);
  auto out_dims = out->dims();

  // Input operand
  NNAdapterOperand* input_operand = nullptr;
  if (converter->HasOperand(x_name)) {
    input_operand = converter->GetOperand(x_name);
  } else {
    if (has_x_scale) {
      input_operand =
          converter->AddQuant8VariableOperand(x_dims, x_scale, x_name);
    } else {
      input_operand = converter->AddFloat32VariableOperand(x_dims, x_name);
    }
  }

  // Output operand
  NNAdapterOperand* output_operand = nullptr;
  if (has_out_scale) {
    output_operand =
        converter->AddQuant8VariableOperand(out_dims, out_scale, out_name);
  } else {
    output_operand = converter->AddFloat32VariableOperand(out_dims, out_name);
  }

  // Activation operation
  std::vector<NNAdapterOperand*> input_operands = {input_operand};
  std::vector<NNAdapterOperand*> output_operands = {output_operand};
  NNAdapterOperation* activation_operation = nullptr;
  if (op_type == "sigmoid") {
    activation_operation = converter->AddOperation(NNADAPTER_SIGMOID);
  } else if (op_type == "relu") {
    activation_operation = converter->AddOperation(NNADAPTER_RELU);
  } else if (op_type == "relu6") {
    activation_operation = converter->AddOperation(NNADAPTER_RELU6);
  } else if (op_type == "tanh") {
    activation_operation = converter->AddOperation(NNADAPTER_TANH);
  } else if (op_type == "hard_swish") {
    // Hard-Swish(x) = x * RELU6(x+3) / 6
    float shift_factor = 3.0f;
    float div_factor = 6.0f;
    auto fuse_code_operand =
      converter->AddInt32ConstantOperand(NNADAPTER_FUSED_NONE);
    // x+3
    NNAdapterOperation* add_operation = nullptr;
    add_operation = converter->AddOperation(NNADAPTER_ADD);
    NNAdapterOperand* shift_operand = nullptr;
    NNAdapterOperand* shift_out = nullptr;
    shift_operand = converter->AddFloat32ConstantOperand(&shift_factor, x_dims, false);
    shift_out = converter->AddFloat32VariableOperand(x_dims, "Shift");
    std::vector<NNAdapterOperand*> shift_out_operands = {shift_out};
    std::vector<NNAdapterOperand*> shift_operands = {input_operand, shift_operand, fuse_code_operand};
    converter->SetOperation(add_operation, &shift_operands, &shift_out_operands);

    // relu(x+3)
    NNAdapterOperation* relu_operation = nullptr;
    relu_operation = converter->AddOperation(NNADAPTER_RELU6);
    NNAdapterOperand* relu_out = nullptr;
    relu_out = converter->AddFloat32VariableOperand(x_dims, "Relu6");
    std::vector<NNAdapterOperand*> relu_out_operands = {relu_out};
    converter->SetOperation(relu_operation, &shift_out_operands, &relu_out_operands);

    // relu(x+3) * x
    NNAdapterOperation* mul_operation = nullptr;
    mul_operation = converter->AddOperation(NNADAPTER_MUL);
    std::vector<NNAdapterOperand*> mul_operands = {input_operand, shift_out, fuse_code_operand};
    NNAdapterOperand* mul_out = nullptr;
    mul_out = converter->AddFloat32VariableOperand(x_dims, "Mul");
    std::vector<NNAdapterOperand*> mul_out_operands = {mul_out};
    converter->SetOperation(mul_operation, &mul_operands, &mul_out_operands);

    // relu(x+3) * x / 6
    NNAdapterOperation* div_operation = nullptr;
    div_operation = converter->AddOperation(NNADAPTER_DIV);
    NNAdapterOperand* div_operand = nullptr;
    div_operand = converter->AddFloat32ConstantOperand(&div_factor, x_dims, false);
    std::vector<NNAdapterOperand*> div_operands = {mul_out, div_operand, fuse_code_operand};
    converter->SetOperation(div_operation, &div_operands, &output_operands);

    return REBUILD_WHEN_SHAPE_CHANGED;
  } else {
    LOG(WARNING) << "Unsupported activation type: " << op_type;
    return FAILED;
  }
  converter->SetOperation(
      activation_operation, &input_operands, &output_operands);
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace nnadapter
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(relu,
                         kNNAdapter,
                         paddle::lite::subgraph::nnadapter::ActConverter);
REGISTER_SUBGRAPH_BRIDGE(sigmoid,
                         kNNAdapter,
                         paddle::lite::subgraph::nnadapter::ActConverter);
REGISTER_SUBGRAPH_BRIDGE(relu6,
                         kNNAdapter,
                         paddle::lite::subgraph::nnadapter::ActConverter);
REGISTER_SUBGRAPH_BRIDGE(tanh,
                         kNNAdapter,
                         paddle::lite::subgraph::nnadapter::ActConverter);

// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto converter = static_cast<Converter*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "Converting " << op_type << " ...";

  // Get input and output vars and op attributes
  auto x_name = op_info->Input("Start");
  auto x_scale_name = "Start0_scale";
  auto has_x_scale = op_info->HasInputScale(x_scale_name, true);
  auto x_scale =
      has_x_scale ? op_info->GetInputScale(x_scale_name, true)[0] : 0.f;
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();

  auto end_name = op_info->Input("End").front();
  auto end_scale_name = "End0_scale";
  auto has_end_scale = op_info->HasInputScale(end_scale_name, true);
  auto end_scale =
      has_end_scale ? op_info->GetInputScale(end_scale_name, true)[0] : 0.f;
  auto end = scope->FindMutableTensor(end_name);
  auto end_dims = end->dims();

  auto step_name = op_info->Input("Step").front();
  auto step_scale_name = "Step0_scale";
  auto has_step_scale = op_info->HasInputScale(step_scale_name, true);
  auto step_scale =
      has_step_scale ? op_info->GetInputScale(step_scale_name, true)[0] : 0.f;
  auto step = scope->FindMutableTensor(step_name);
  auto step_dims = step->dims();

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

  // End operand
  NNAdapterOperand* end_operand = nullptr;
  if (converter->HasOperand(end_name)) {
    end_operand = converter->GetOperand(end_name);
  } else {
    if (has_end_scale) {
      end_operand =
          converter->AddQuant8VariableOperand(end_dims, end_scale, end_name);
    } else {
      end_operand = converter->AddFloat32VariableOperand(end_dims, end_name);
    }
  }

  // Step operand
  NNAdapterOperand* input_operand = nullptr;
  if (converter->HasOperand(step_name)) {
    step_operand = converter->GetOperand(step_name);
  } else {
    if (has_step_scale) {
      step_operand =
          converter->AddQuant8VariableOperand(step_dims, step_scale, step_name);
    } else {
      step_operand = converter->AddFloat32VariableOperand(step_dims, step_name);
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

  // Range operation
  std::vector<NNAdapterOperand*> input_operands = {
      input_operand,
      end_operand,
      step_operand}
  std::vector<NNAdapterOperand*> output_operands = {output_operand};
  auto range_operation = converter->AddOperation(NNADAPTER_RANGE);
  converter->SetOperation(range_operation, &input_operands, &output_operands);
  return REBUILD_WHEN_SHAPE_CHANGED;

}  // namespace nnadapter
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(relu,
                         kNNAdapter,
                         paddle::lite::subgraph::nnadapter::RangeConverter);
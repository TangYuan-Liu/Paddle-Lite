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

#include "driver/huawei_ascend_npu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int Program::ConvertActivation(hal::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();

  if (operation->type == NNADAPTER_HARD_SWISH) {
    NNADAPTER_CHECK_EQ(input_count, 4);
    NNADAPTER_CHECK_EQ(output_count, 1);
  } else {
    NNADAPTER_CHECK_EQ(input_count, 1);
    NNADAPTER_CHECK_EQ(output_count, 1);
  }
  // Input
  auto input_operand = input_operands[0];
  NNADAPTER_VLOG(5) << "input: " << OperandToString(input_operand);
 
  // output
  auto output_operand = output_operands[0];
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);

  // Convert to GE operators
  auto input_operator = GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = ConvertOperand(input_operand);
  }

  auto act_name = GetOperatorName(output_operand);
  switch (operation->type) {
#define CONVERT_UNARY_ACTIVATION(type, class_name)                \
  case NNADAPTER_##type: {                                        \
    auto act_op = std::make_shared<ge::op::class_name>(act_name); \
    SET_INPUT(act_op, x, input_operator);                         \
    MAP_OUTPUT(act_op, y, output_operand);                        \
  } break;
    CONVERT_UNARY_ACTIVATION(SIGMOID, Sigmoid);
    CONVERT_UNARY_ACTIVATION(RELU, Relu);
    CONVERT_UNARY_ACTIVATION(RELU6, Relu6);
    CONVERT_UNARY_ACTIVATION(TANH, Tanh);
  
  case NNADAPTER_HARD_SWISH: {
    // offset
    auto offset_operand = input_operands[1];
    if (offset_operand)
      NNADAPTER_VLOG(5) << "input: " << OperandToString(offset_operand);

    // threshold
    auto threshold_operand = input_operands[2];
    if (threshold_operand)
      NNADAPTER_VLOG(5) << "input: " << OperandToString(threshold_operand);

    // scale
    auto scale_operand = input_operands[3];
    if (scale_operand)
      NNADAPTER_VLOG(5) << "input: " << OperandToString(scale_operand);

    auto offset_operator = GetMappedOperator(offset_operand);
    if (!offset_operator) {
      offset_operator = ConvertOperand(offset_operand);
    }

    auto threshold_operator = GetMappedOperator(threshold_operand);
    if (!threshold_operator) {
      threshold_operator = ConvertOperand(threshold_operand);
    }

    auto scale_operator = GetMappedOperator(scale_operand);
    if (!scale_operator) {
      scale_operator = ConvertOperand(scale_operand);
    }

    //auto x_size = input_operand->type.dimensions[0];
    auto x_size = input_operand->type.dimension_count;
    std::vector<float> scale1_vec(x_size, 1);
    std::vector<float> clip_vec(x_size, 0);
    auto scale1 = AddFloat32ConstantOperator(scale1_vec);
    auto clip_min = AddFloat32ConstantOperator(clip_vec);

    auto sca_op = std::make_shared<ge::op::Scale>("scale");
    SET_INPUT(sca_op, x, input_operator);
    SET_INPUT(sca_op, scale, scale1);
    std::shared_ptr<Operator> sca_operator = MAP_OUTPUT(sca_op, y, output_operand);

    auto clip_op = std::make_shared<ge::op::ClipByValue>("clip");
    SET_INPUT(clip_op, x, sca_operator);
    SET_INPUT(clip_op, clip_value_min, clip_min);
    SET_INPUT(clip_op, clip_value_max, threshold_operator);
    std::shared_ptr<Operator> clip_operator = MAP_OUTPUT(clip_op, y, output_operand);
    
    auto mul_op = std::make_shared<ge::op::Mul>("mul");
    SET_INPUT(mul_op, x1, input_operator);
    SET_INPUT(mul_op, x2, clip_operator);
    std::shared_ptr<Operator> mul_operator = MAP_OUTPUT(mul_op, y, output_operand);

    auto div_op = std::make_shared<ge::op::Div>("div");
    SET_INPUT(div_op, x1, mul_operator);
    SET_INPUT(div_op, x2, scale_operator);
    MAP_OUTPUT(div_op, y, output_operand);

    return NNADAPTER_NO_ERROR;
  }

#undef CONVERT_UNARY_ACTIVATION
    default:
      NNADAPTER_LOG(FATAL) << "Unsupported activation operation type "
                           << OperationTypeToString(operation->type)
                           << " is found.";
      break;
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter

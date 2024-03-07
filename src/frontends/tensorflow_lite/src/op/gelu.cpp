// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "op_translation_utils.hpp"
#include "utils.hpp"

using namespace std;

namespace ov {
namespace frontend {
namespace tensorflow_lite {
namespace op {

OutputVector gelu(const ov::frontend::tensorflow_lite::NodeContext& node) {
    const auto& decoder = get_decoder(node);
    printf("Keyon: Running from %s:%d\n", __FILE__, __LINE__);
//    std::cout << " Keyon: approximate " << decoder->get_attribute(&tflite::GeluOptions::approximate) << std::endl;
    std::map<std::string, ov::Any> attrs{
        {"approximate", false},//decoder->get_attribute(&tflite::GeluOptions::approximate)},
    };
    return attribute_helper(node, attrs, ov::frontend::tensorflow::op::translate_gelu_op);
}


}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov

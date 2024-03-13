// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gelu.hpp"

//#include "openvino/frontend/pytorch/node_context.hpp"
//#include "utils.hpp"

#include "common_op_table.hpp"
#include "openvino/opsets/opset10.hpp"
using namespace std;
using namespace ov;
using namespace ov::opset10;



namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_gelu_op(const NodeContext& node) {
    default_op_checks(node, 1, {"GELU"});
    auto input = node.get_input(0);
//    bool att = node.has_attribute<bool>("approximate");
//	std::cout << "Keyon: has attribut  " << att << std::endl;
    auto approximate = node.get_attribute<bool>("approximate");
//    auto approximate = node.get_attribute<element::boolean>("approximate");
    const auto mode = (approximate == true) ? ov::op::GeluApproximationMode::TANH : ov::op::GeluApproximationMode::ERF;
    auto res = make_shared<ov::op::v7::Gelu>(input, mode);
    set_node_name(node.get_name(), res);
    return res->outputs();
};

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov

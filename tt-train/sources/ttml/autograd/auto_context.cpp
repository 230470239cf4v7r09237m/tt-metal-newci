// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "auto_context.hpp"

#include <optional>

namespace ttml::autograd {

std::mt19937& AutoContext::get_generator() {
    return m_generator;
}

void AutoContext::set_seed(uint32_t seed) {
    m_seed = seed;
    m_generator = std::mt19937(m_seed);
}

uint32_t AutoContext::get_seed() const {
    return m_seed;
}

AutoContext& AutoContext::get_instance() {
    static AutoContext& instance = core::Indestructible<AutoContext>::get_instance();
    return instance;
}
std::optional<NodeId> AutoContext::add_backward_node(GradFunction&& grad_function, std::span<NodeId> links) {
    if (m_grads_mode == GradMode::DISABLED) {
        return std::nullopt;
    }
    return m_graph.add_node(std::move(grad_function), links);
}
void AutoContext::set_gradient_mode(GradMode mode) {
    m_grads_mode = mode;
}
GradMode AutoContext::get_gradient_mode() const {
    return m_grads_mode;
}

void AutoContext::reset_graph() {
    m_graph.reset();
}

void AutoContext::init_device(tt::tt_metal::distributed::MeshShape shape) {
    m_device = std::make_unique<core::MeshDevice>(shape);
}

ttnn::distributed::MeshDevice& AutoContext::get_device() {
    if (!m_device) {
        init_device(m_mesh_shape);
    }

    return m_device->get_device();
}

AutoContext::AutoContext() : m_generator(m_seed) {
}

void AutoContext::set_mesh_shape(tt::tt_metal::distributed::MeshShape shape) {
    if (m_device) {
        throw std::runtime_error("set_mesh_shape was called after the device was created.");
    }
    m_mesh_shape = shape;
}

tt::tt_metal::distributed::MeshShape AutoContext::get_mesh_shape() const {
    return m_mesh_shape;
}
}  // namespace ttml::autograd

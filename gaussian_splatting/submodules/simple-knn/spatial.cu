/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "spatial.h"
#include "simple_knn.h"

std::tuple<torch::Tensor, torch::Tensor> distCUDA2(const torch::Tensor& points)
{
  const int P = points.size(0);

  auto float_opts = points.options().dtype(torch::kFloat32);
  auto int_opts = points.options().dtype(torch::kInt32);
  torch::Tensor nearestIndices = torch::empty({P, 3}, int_opts).set_requires_grad(false);
  torch::Tensor means = torch::full({P}, 0.0, float_opts);
  
  SimpleKNN::knn(P, (float3*)points.contiguous().data<float>(), means.contiguous().data<float>(), nearestIndices.contiguous().data<int32_t>());

  return std::make_tuple(means, nearestIndices);
}
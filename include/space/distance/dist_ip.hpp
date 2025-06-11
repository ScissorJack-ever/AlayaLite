/*
 * Copyright 2025 AlayaDB.AI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include "dist_config.hpp"

namespace alaya {

FAST_BEGIN
template <typename DataType = float, typename DistanceType = float>
inline auto ip_sqr(DataType *x, DataType *y, size_t dim) -> DistanceType { // 普通内积运算
  DistanceType sum = 0;
  for (size_t i = 0; i < dim; ++i) {
    sum += x[i] * y[i];
  }
  return -sum;
}
FAST_END

FAST_BEGIN
template <typename DataType = float, typename DistanceType = float>
inline auto ip_sqr_sq4(const uint8_t *encoded_x, const uint8_t *encoded_y, size_t dim,
                       const DataType *min, const DataType *max) -> DistanceType {
  DistanceType sum = 0;

  for (size_t i = 0; i < dim; i += 2) { // 每个byte存了两维的量化数据,所以i+=2
    //应该分x1,x2高低8位都要算
    auto x = (encoded_x[i] >> 4) & 0x0F;
    auto y = encoded_y[i] & 0x0F;
    sum += (x * (max[i] - min[i]) + min[i]) * (y * (max[i] - min[i]) + min[i]); // x是一个[0,15]的整数,想算原内积ip(损失精度版),sum应该除于一个15*15，只是比大小就不用
  }

  return -sum; // 最大内积问题转换为最小化负内积问题
}
FAST_END

FAST_BEGIN
template <typename DataType = float, typename DistanceType = float>
inline auto ip_sqr_sq8(const uint8_t *encoded_x, const uint8_t *encoded_y, size_t dim,
                       const DataType *min, const DataType *max) -> DistanceType {
  DistanceType sum = 0;

  for (size_t i = 0; i < dim; i += 1) {
    sum +=
        (encoded_x[i] * (max[i] - min[i]) + min[i]) * (encoded_y[i] * (max[i] - min[i]) + min[i]); // 同上,想算原内积ip(损失精度版),sum应该除于一个255*255,只是比大小就不用
  }

  return -sum; // 最大内积问题转换为最小化负内积问题
}
FAST_END

}  // namespace alaya

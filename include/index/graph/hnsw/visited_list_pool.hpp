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

#include <cstdint>
#include <cstring>
#include <deque>
#include <mutex>  //NOLINT [build/c++11]

namespace alaya {
using vl_type = uint16_t;

class VisitedList {
 public:
  vl_type cur_v_; // 当前标记值，表示当前这次搜索的“版本号”或“轮次标识”，避免每次搜索都重新初始化数组。
  vl_type *mass_; // 动态数组，用来记录每个节点是否被访问过（通过比较 cur_v_ 和对应位置的值）。
  uint32_t numelements_; // 图中节点的最大数量，决定了 mass_ 数组的大小。

  explicit VisitedList(int numelements1) {
    cur_v_ = -1;
    numelements_ = numelements1;
    mass_ = new vl_type[numelements_];
  }

  void reset() {
    cur_v_++;
    if (cur_v_ == 0) {
      memset(mass_, 0, sizeof(vl_type) * numelements_);
      cur_v_++;
    }
  }

  ~VisitedList() { delete[] mass_; }
}; // visited list

///////////////////////////////////////////////////////////
//
// Class for multi-threaded pool-management of VisitedLists
//
/////////////////////////////////////////////////////////

class VisitedListPool {
  std::deque<VisitedList *> pool_;
  std::mutex poolguard_;
  int numelements_; // 每个 VisitedList 中维护的数组大小。

 public:
  VisitedListPool(int initmaxpools, int numelements1) {
    numelements_ = numelements1;
    for (int i = 0; i < initmaxpools; i++) {
      pool_.push_front(new VisitedList(numelements_));
    }
  }

  auto get_free_visited_list() -> VisitedList * {
    VisitedList *rez;
    {
      std::unique_lock<std::mutex> lock(poolguard_);
      if (!pool_.empty()) {
        rez = pool_.front();
        pool_.pop_front();
      } else {
        rez = new VisitedList(numelements_);
      }
    }
    rez->reset();
    return rez;
  }

  void release_visited_list(VisitedList *vl) {
    std::unique_lock<std::mutex> lock(poolguard_);
    pool_.push_front(vl);
  }

  ~VisitedListPool() {
    while (!pool_.empty()) {
      VisitedList *rez = pool_.front();
      pool_.pop_front();
      delete rez;
    }
  }
}; // visited list pool

}  // namespace alaya

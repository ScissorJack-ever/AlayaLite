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
#include <memory>

#include "../../index/graph/graph.hpp"
#include "../../space/space_concepts.hpp"
#include "../../utils/prefetch.hpp"
#include "../../utils/query_utils.hpp"
#include "coro/task.hpp"
#include "job_context.hpp"

namespace alaya {

template <typename DistanceSpaceType, typename DataType = DistanceSpaceType::DataTypeAlias,
          typename DistanceType = DistanceSpaceType::DistanceTypeAlias,
          typename IDType = DistanceSpaceType::IDTypeAlias>
requires Space<DistanceSpaceType, DataType, DistanceType, IDType>  //硬性要求了DistanceSpaceType的实现
struct GraphSearchJob {
  std::shared_ptr<DistanceSpaceType> space_ = nullptr;        ///< The is a data manager interface .
  std::shared_ptr<Graph<DataType, IDType>> graph_ = nullptr;  ///< The search graph.
  std::shared_ptr<JobContext<IDType>> job_context_;           ///< The shared job context

  explicit GraphSearchJob(std::shared_ptr<DistanceSpaceType> space,
                          std::shared_ptr<Graph<DataType, IDType>> graph,
                          std::shared_ptr<JobContext<IDType>> job_context = nullptr)
      : space_(space), graph_(graph) {
    if (!job_context_) {
      job_context_ = std::make_shared<JobContext<IDType>>();
    }
  }

  auto search(DataType *query, uint32_t k, IDType *ids, uint32_t ef) -> coro::task<> { // 这就是一个coroutine
    auto query_computer = space_->get_query_computer(query);
    LinearPool<DistanceType, IDType> pool(space_->get_data_num(), ef);
    graph_->initialize_search(pool, query_computer);

    space_->prefetch_by_address(query);

    while (pool.has_next()) {
      auto u = pool.pop(); // 拿id

      mem_prefetch_l1(graph_->edges(u), graph_->max_nbrs_ * sizeof(IDType) / 64);
      co_await std::suspend_always{}; // 等待外部resume

      for (int i = 0; i < graph_->max_nbrs_; ++i) {
        int v = graph_->at(u, i);

        if (v == -1) {
          break;
        }

        if (pool.vis_.get(v)) { // visited
          continue;
        }
        pool.vis_.set(v);

        space_->prefetch_by_id(v);
        co_await std::suspend_always{};

        auto cur_dist = query_computer(v);
        pool.insert(v, cur_dist);
      }
    }

    for (int i = 0; i < k; i++) {
      ids[i] = pool.id(i);
    }
    co_return; // 用于触发return_void？
  }

  auto search(DataType *query, uint32_t k, IDType *ids, DistanceType *distances, uint32_t ef)
      -> coro::task<> { //多给了一个distances参数
    auto query_computer = space_->get_query_computer(query);
    LinearPool<DistanceType, IDType> pool(space_->get_data_num(), ef);
    graph_->initialize_search(pool, query_computer);

    space_->prefetch_by_address(query);

    while (pool.has_next()) {
      auto u = pool.pop();

      mem_prefetch_l1(graph_->edges(u), graph_->max_nbrs_ * sizeof(IDType) / 64);
      co_await std::suspend_always{};

      for (int i = 0; i < graph_->max_nbrs_; ++i) {
        int v = graph_->at(u, i);

        if (v == -1) {
          break;
        }

        if (pool.vis_.get(v)) {
          continue;
        }
        pool.vis_.set(v);

        space_->prefetch_by_id(v);
        co_await std::suspend_always{};

        auto cur_dist = query_computer(v);
        pool.insert(v, cur_dist);
      }
    }

    for (int i = 0; i < k; i++) {
      ids[i] = pool.id(i);
      distances[i] = pool.dist(i); //唯一多出来的一行
    }
    co_return;
  }

  void search_solo(DataType *query, uint32_t k, IDType *ids, uint32_t ef) {
    auto query_computer = space_->get_query_computer(query);
    LinearPool<DistanceType, IDType> pool(space_->get_data_num(), ef);
    graph_->initialize_search(pool, query_computer);

    // 没有space->prefetch（query）了

    while (pool.has_next()) {
      auto u = pool.pop();

      // 没有memfetch和coawait了

      for (int i = 0; i < graph_->max_nbrs_; ++i) {
        int v = graph_->at(u, i);

        if (v == -1) {
          break;
        }

        if (pool.vis_.get(v)) {
          continue;
        }
        pool.vis_.set(v);

        // 多出的prefetch
        auto jump_prefetch = i + 3;
        if (jump_prefetch < graph_->max_nbrs_) {
          auto prefetch_id = graph_->at(u, jump_prefetch);
          if (prefetch_id != -1) {
            space_->prefetch_by_id(prefetch_id);
          }
        }
        auto cur_dist = query_computer(v);
        pool.insert(v, cur_dist);
      }
    }
    for (int i = 0; i < k; i++) {
      ids[i] = pool.id(i);
    }
  }

  void search_solo_updated(DataType *query, uint32_t k, IDType *ids, uint32_t ef) {
    auto query_computer = space_->get_query_computer(query);
    LinearPool<DistanceType, IDType> pool(space_->get_data_num(), ef);
    graph_->initialize_search(pool, query_computer);

    //没有space->prefetch

    while (pool.has_next()) {
      auto u = pool.pop();

      //代替mem_prefetch与coawait 
      if (job_context_->removed_node_nbrs_.find(u) != job_context_->removed_node_nbrs_.end()) {
        for (auto &second_hop_nbr : job_context_->removed_node_nbrs_.at(u)) {
          if (pool.vis_.get(u)) {
            continue;
          }
          pool.vis_.set(u);
          auto dist = query_computer(u);
          pool.insert(u, dist);
        }
        continue;
      }

      for (int i = 0; i < graph_->max_nbrs_; ++i) {
        int v = graph_->at(u, i);

        if (v == -1) {
          break;
        }

        if (pool.vis_.get(v)) {
          continue;
        }
        pool.vis_.set(v);

        // 代替space_prefetch与coawait
        auto jump_prefetch = i + 3;
        if (jump_prefetch < graph_->max_nbrs_) {
          auto prefetch_id = graph_->at(u, jump_prefetch);
          if (prefetch_id != -1) {
            space_->prefetch_by_id(prefetch_id);
          }
        }

        auto cur_dist = query_computer(v);
        pool.insert(v, cur_dist);
      }
    }
    for (int i = 0; i < k; i++) {
      ids[i] = pool.id(i);
    }
  }
};

}  // namespace alaya

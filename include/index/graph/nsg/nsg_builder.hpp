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

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <memory>
#include <random>
#include <stack>
#include <thread>
#include <utility>
#include <vector>
#include "index/graph/graph.hpp"
#include "index/graph/knng/nndescent.hpp"
#include "index/neighbor.hpp"
#include "utils/log.hpp"
#include "utils/random.hpp"
#include "utils/thread_pool.hpp"
#include "space/space_concepts.hpp"
#include "utils/timer.hpp"

namespace alaya {

template <typename DistanceSpaceType>
  requires Space<DistanceSpaceType, typename DistanceSpaceType::DataTypeAlias,
                 typename DistanceSpaceType::DistanceTypeAlias,
                 typename DistanceSpaceType::IDTypeAlias>
struct NSGBuilder {
  using DataType = typename DistanceSpaceType::DataTypeAlias;
  using DistanceType = typename DistanceSpaceType::DistanceTypeAlias;
  using IDType = typename DistanceSpaceType::IDTypeAlias;
  using DistanceSpaceTypeAlias = DistanceSpaceType;

  std::shared_ptr<DistanceSpaceType> space_ =
      nullptr;                ///< The data manager interface for the NSG graph.
  uint32_t dim_;              ///< Dimension of the vectors.
  uint32_t max_nbrs_;         ///< Maximum number of neighbors for each node.
  uint32_t ef_construction_;  ///< Size of the search pool during graph construction.
  uint32_t cut_len_;          ///< Length used for pruning during graph construction.
  IDType vector_num_;         ///< Total number of vectors.
  IDType ep_;                 ///< Entry point for the graph.
  std::unique_ptr<Graph<DataType, IDType>> final_graph_;  ///< Final NSG graph.
  RandomGenerator rng_;                                   ///< Random number generator.

  uint32_t nndescent_max_nbrs_{};             ///< Maximum number of neighbors for Nndescent.
  uint32_t nndescent_selected_sample_num_{};  ///< Number of selected samples for Nndescent.
  uint32_t nndescent_radius_{};  ///< Radius for reverse nearest neighbors in Nndescent.
  uint32_t nndescent_candidate_pool_size_{};  ///< Size of candidate pool in Nndescent.
  uint32_t nndescent_iters_{};                ///< Number of iterations for Nndescent.

  /**
   * @brief Constructor for NSGBuilder.
   *
   * Initializes the NSGBuilder with the given distance space and optional parameters.
   *
   * @param space The data manager interface for the NSG graph.
   * @param R Maximum number of neighbors for each node (default is 32).
   * @param L Size of the search pool during graph construction (default is 200).
   */
  explicit NSGBuilder(std::shared_ptr<DistanceSpaceType> &space, uint32_t R = 32, uint32_t L = 200)
      : space_(space),
        dim_(space->get_dim()),
        max_nbrs_(R),
        ef_construction_(L),
        rng_(0x0903),
        nndescent_max_nbrs_(64),
        nndescent_selected_sample_num_(10),
        nndescent_radius_(100),
        nndescent_candidate_pool_size_(nndescent_max_nbrs_ + 50),
        nndescent_iters_(10) {
    cut_len_ = max_nbrs_ + 100;
    srand(0x1998); // 使后续rand()调用产生可重复的随机序列。
    vector_num_ = space_->get_data_num();
  }

  /**
   * @brief Build the NSG graph.
   *
   * Constructs the NSG graph using the Nndescent algorithm and subsequent linking and pruning
   * steps.
   *
   * @return A unique pointer to the final NSG graph.
   */
  auto build_graph(uint32_t thread_num = 1) -> std::unique_ptr<Graph<DataType, IDType>> {
    NndescentImpl<DistanceSpaceType> nndescent(space_, nndescent_max_nbrs_);
    nndescent.selected_sample_num_ = nndescent_selected_sample_num_;
    nndescent.radius_ = nndescent_radius_;
    nndescent.candidate_pool_size_ = nndescent_candidate_pool_size_;
    nndescent.iterations_ = nndescent_iters_;

    auto nndescent_graph = nndescent.build_graph();

    init(nndescent_graph);
    std::vector<int> degrees(vector_num_, 0);

    {
      Graph<DataType, IDType> tmp_graph(space_->get_capacity(), max_nbrs_);
      link(nndescent_graph, tmp_graph);

      final_graph_ = std::make_unique<Graph<DataType, IDType>>(space_->get_capacity(), max_nbrs_);
      // std::fill_n(final_graph_->data_, vector_num_ * max_nbrs_,
      // Graph<DataType, IDType>::kEmptyId);
      final_graph_->eps_.push_back(ep_);

      unsigned int num_cores = std::thread::hardware_concurrency();
      ThreadPool pool(num_cores);

      IDType per_core_num = (vector_num_ + num_cores - 1) / num_cores;
      for (int thread_id = 0; thread_id < num_cores; thread_id++) {
        pool.enqueue([this, &tmp_graph, &degrees, per_core_num, thread_id]() { // copy tem_graph and count degree
          IDType start = thread_id * per_core_num;
          IDType end = std::min((thread_id + 1) * per_core_num, vector_num_);

          for (IDType i = start; i < end; i++) {
            int cnt = 0;
            for (int j = 0; j < max_nbrs_; j++) {
              IDType id = tmp_graph.at(i, j);
              if (id != Graph<DataType, IDType>::kEmptyId) {
                final_graph_->at(i, cnt) = id;
                cnt += 1;
              }
            }
            degrees[i] = cnt;
          }
        });
      }
      pool.wait_until_all_tasks_completed(num_cores);
    }

    // 确保vector_num_的接入数
    int num_attached = tree_grow(degrees);
    int max_degree = 0;
    int min_degree = 1e6;
    double avg_degree = 0;
    for (int i = 0; i < vector_num_; i++) { // 计算每个点的邻居数，即出度数
      int size = 0;
      while (size < max_nbrs_ && final_graph_->at(i, size) != Graph<DataType, IDType>::kEmptyId) {
        size += 1;
      }
      max_degree = std::max(size, max_degree);
      min_degree = std::min(size, min_degree);
      avg_degree += size;
    }
    avg_degree /= vector_num_;
    LOG_INFO("Degree Statistics: Max = {}, Min = {}, Avg = {}", max_degree, min_degree, avg_degree);

    return std::move(final_graph_);
  }

  /**
   * @brief Initialize the NSG graph.
   *
   * Computes the center of the dataset and selects an entry point using a search on the graph.
   *
   * @param knng The initial graph built by Nndescent.
   */
  void init(const std::unique_ptr<Graph<DataType, IDType>> &knng) { // 主要是设置entry point
    // 求形心center
    std::vector<DataType> center(dim_);
    for (size_t i = 0; i < dim_; ++i) {
      center[i] = 0.0;
    }
    for (size_t i = 0; i < vector_num_; i++) {
      for (size_t j = 0; j < dim_; j++) {
        center[j] += space_->get_data_by_id(i)[j];
      }
    }
    for (size_t i = 0; i < dim_; i++) {
      center[i] /= vector_num_;
    }

    size_t ep_init = rng_.rand_int(vector_num_);
    std::vector<Neighbor<IDType>> retset;
    std::vector<Node<IDType>> tmpset; // 没用
    std::vector<bool> vis(vector_num_);
    // 给center找nn
    search_on_graph<false>(center.data(), knng, vis, ep_init, ef_construction_, retset, tmpset);
    // set enterpoint
    this->ep_ = retset[0].id_;
  }

  /**
   * @brief Perform a search on the graph.
   *
   * Searches for the nearest neighbors of a query vector on the graph.
   *
   * @tparam collect_full_set Flag to indicate whether to collect the full set of neighbors.
   * @param q Query vector.
   * @param graph The graph to search on.
   * @param vis Visited nodes.
   * @param ep Entry point for the search.
   * @param pool_size Size of the search pool.
   * @param retset Result set of nearest neighbors.
   * @param full_set Full set of neighbors (if collect_full_set is true).
   */
  template <bool collect_full_set>
  void search_on_graph(const DataType *q, const std::unique_ptr<Graph<DataType, IDType>> &graph,
                       std::vector<bool> &vis, IDType ep, int pool_size,
                       std::vector<Neighbor<IDType>> &retset,
                       std::vector<Node<IDType>> &full_set) const {
    RandomGenerator gen(0x1234);
    retset.resize(pool_size + 1);

    std::vector<IDType> init_ids(pool_size);
    int num_ids = 0;
    for (int i = 0; i < init_ids.size() && i < graph->max_nbrs_; i++) { // 遍历entry point的邻居作为初始id
      IDType id = graph->at(ep, i);
      if (id < 0 || id >= vector_num_) {
        continue;
      }
      init_ids[i] = id;
      vis[id] = true;
      num_ids += 1;
    }

    while (num_ids < pool_size) { // 随机id来补满init_ids
      int id = gen.rand_int(vector_num_);
      if (vis[id]) {
        continue;
      }
      init_ids[num_ids] = id;
      num_ids++;
      vis[id] = true;
    }

    for (int i = 0; i < init_ids.size(); i++) { // 遍历init_id计算与q的距离并记录在retset(和full_set)中
      int id = init_ids[i];
      DistanceType dist =
          space_->get_dist_func()(const_cast<DataType *>(q), space_->get_data_by_id(id), dim_);
      retset[i] = Neighbor<IDType>(id, dist, true); // true表示还得遍历
      if (collect_full_set) {
        full_set.emplace_back(id, dist);
      }
    }

    std::sort(retset.begin(), retset.begin() + pool_size);

    int k = 0;
    while (k < pool_size) {
      int updated_pos = pool_size;
      if (retset[k].flag_) {
        retset[k].flag_ = false;
        int n = retset[k].id_;
        for (int m = 0; m < graph->max_nbrs_; m++) { // 遍历retset中合法未访问的点
          int id = graph->at(n, m);
          if (id < 0 || id >= vector_num_ || vis[id]) {
            continue;
          }
          vis[id] = true;
          DistanceType dist =
              space_->get_dist_func()(const_cast<DataType *>(q), space_->get_data_by_id(id), dim_);
          Neighbor<IDType> nn(id, dist, true); // 表示nei的nei
          if (collect_full_set) {
            full_set.emplace_back(id, dist);
          }
          if (dist >= retset[pool_size - 1].distance_) {
            continue;
          }
          int r = insert_into_pool(retset.data(), pool_size, nn); // 合法则返回在retset的插入位置
          updated_pos = std::min(updated_pos, r); // 记录距离最小的插入位置，retset随便插，反正只有前pool_size个是被承认的
        } // for
      } // if
      k = (updated_pos <= k) ? updated_pos : (k + 1); // 到retset中刚插入还未vis的位置，没有就正常推进
    } // while

  }

  /**
   * @brief Link nodes in the graph.
   *
   * Links nodes in the graph using the initial graph built by Nndescent.
   *
   * @param knng The initial graph built by Nndescent.
   * @param graph The graph to be linked.
   */
  void link(const std::unique_ptr<Graph<DataType, IDType>> &knng, Graph<DataType, IDType> &graph) {
    auto t1 = Timer();
    std::atomic<int> cnt{0};
    unsigned int num_cores = std::thread::hardware_concurrency();
    ThreadPool pool(num_cores);
    // 每个thread处理的vector数,num_cores*per_core_num刚好>=vector_num
    IDType per_core_num = (vector_num_ + num_cores - 1) / num_cores; 
    for (int thread_id = 0; thread_id < num_cores; thread_id++) {
      pool.enqueue([this, &knng, &graph, &cnt, per_core_num, thread_id]() { 
        // 连续的一整块(per_core_num个)idtype交给此thread处理
        for (IDType i = thread_id * per_core_num;
             i < (thread_id + 1) * per_core_num && i < vector_num_; i++) { // (knng)剪枝,结果存于graph
          std::vector<Node<IDType>> pool;
          std::vector<Neighbor<IDType>> tmp; // 占位的东西，没用
          std::vector<bool> vis(vector_num_);
          // Searches for the nearest neighbors of a query vector in knng and push it into pool
          search_on_graph<true>(space_->get_data_by_id(i), knng, vis, ep_, ef_construction_, tmp,
                                pool);
          sync_prune(i, pool, vis, knng, graph);
          pool.clear();
          tmp.clear();
          int cur = cnt.fetch_add(1);
          if ((cur + 1) % 10000 == 0) {
            LOG_INFO("NSG building progress: [{}/{}]", cur + 1, vector_num_);
          }
        }
      });
    }
    pool.wait_until_all_tasks_completed(num_cores);

    pool.reset_task();
    std::vector<std::mutex> locks(vector_num_);
    for (int thread_id = 0; thread_id < num_cores; thread_id++) {
      pool.enqueue([this, &graph, &locks, per_core_num, thread_id]() { // (second)graph反向连接
        for (IDType i = thread_id * per_core_num;
             i < (thread_id + 1) * per_core_num && i < vector_num_; ++i) {
          add_reverse_links(i, locks, graph);
        }
      });
    }
    pool.wait_until_all_tasks_completed(num_cores);

    LOG_INFO("NSG building cost: {}", t1.elapsed() * 1.0 / 1000 / 1000);
  }

  /**
   * @brief Synchronize and prune the graph.
   *
   * Synchronizes and prunes the graph to ensure valid and efficient connections.
   *
   * @param q Query node.
   * @param pool Pool of neighbors.
   * @param vis Visited nodes.
   * @param knng The initial graph built by Nndescent.
   * @param graph The graph to be synchronized and pruned.
   */
  void sync_prune(IDType q, std::vector<Node<IDType>> &pool, std::vector<bool> &vis,
                  const std::unique_ptr<Graph<DataType, IDType>> &knng,
                  Graph<DataType, IDType> &graph) {
    for (int i = 0; i < knng->max_nbrs_; i++) { // 遍历q在knng中的邻居，并把其中未访问的有效邻居插入pool
      IDType id = knng->at(q, i);
      if (id < 0 || id >= vector_num_ || vis[id]) {
        continue;
      }

      DistanceType dist = space_->get_distance(q, id);
      pool.emplace_back(id, dist);
    }

    std::sort(pool.begin(), pool.end()); // 按与q的dist升序排列，此时q作为剪枝时的"A"，保证AB<AC

    std::vector<Node<IDType>> result;

    int start = 0;
    if (pool[start].id_ == q) { // pool传入时不是空的，但按理说也不会有q...感觉多余
      start++;
    }
    result.push_back(pool[start]); // 先行插入一个"B"，以供prune C时比较

    // cut_len_:Length used for pruning during graph construction. 数值上 = max_nbrs_ + 100
    // 只有pool中前cut_len_个才参与判断,即使它们最终没有录入result
    while (result.size() < max_nbrs_ && (++start) < pool.size() && start < cut_len_) {
      auto &p = pool[start];
      bool occlude = false;
      for (int t = 0; t < result.size(); t++) {
        if (p.id_ == result[t].id_) { // pool中的id会duplicate嘛？
          occlude = true;
          break;
        }

        DistanceType djk = space_->get_distance(result[t].id_, p.id_); // BC
        if (djk < p.distance_) { // BC<AC,prune C
          occlude = true;
          break;
        }
      }

      if (!occlude) {
        result.push_back(p);
      }
    }

    for (int i = 0; i < max_nbrs_; i++) { // 将result中的结果作为q在graph中的新邻居
      if (i < result.size()) {
        graph.at(q, i) = result[i].id_;
      } else {
        graph.at(q, i) = Graph<DataType, IDType>::kEmptyId;
      }
    }
  }
  /**
   * @brief Add reverse links to the graph.
   *
   * This function ensures that the graph is bidirectional by adding reverse links.
   * For each node, it checks its neighbors and adds the current node as a neighbor
   * to those nodes if it is not already present. This process helps in maintaining
   * the bidirectional property of the graph.
   *
   * @param q The index of the current node.
   * @param locks A vector of mutexes for thread safety.
   * @param graph The graph to which reverse links are to be added.
   */
  void add_reverse_links(IDType q, std::vector<std::mutex> &locks, Graph<DataType, IDType> &graph) { 
    // prune for neighbors of q while adding reverse link
    for (int i = 0; i < max_nbrs_; i++) { // 跨越整个函数的大循环
      if (graph.at(q, i) == Graph<DataType, IDType>::kEmptyId) { // 可能没有max_nbrs那么多，检测完就退出
        break;
      }
      Node<IDType> sn(q, graph.at(q, i)); // 第二个参数是dist ,你传id几个意思呢？应该是space_.get_distance(q,des)
      IDType des = graph.at(q, i); // 此轮遍历的q的邻居
      
      std::vector<Node<IDType>> tmp_pool;
      int dup = 0;
      {
        std::scoped_lock guard(locks[des]);
        for (int j = 0; j < max_nbrs_; j++) { // 遍历q邻居的邻居
          if (graph.at(des, j) == Graph<DataType, IDType>::kEmptyId) {
            break;
          }
          if (q == graph.at(des, j)) { // q已经和des相互连接了,dup=1会跳过这个邻居des的遍历(为什么要这么做)
            dup = 1;
            break;
          }
          // 初始dist设为0? 应该是des和其邻居的距离
          tmp_pool.push_back(Node<IDType>(graph.at(des, j),
                                          0));  // Assuming Node constructor takes ID and distance
        }
      }

      if (dup != 0) {
        continue;
      }

      tmp_pool.push_back(sn); // 装的都是des的邻居，虽然q还不是
      // 使用三角不等式进行剪枝
      if (tmp_pool.size() > max_nbrs_) {
        std::vector<Node<IDType>> result;
        int start = 0; // 遍历指针
        std::sort(tmp_pool.begin(), tmp_pool.end()); // 只以与des(A)的dist从小到大排列，保证AB<AC
        result.push_back(tmp_pool[start]); // 装入一个"B"供后面的C比较

        while (result.size() < max_nbrs_ && (++start) < tmp_pool.size()) { // 直到tem_pool遍历完或result录满
          auto &p = tmp_pool[start];
          bool occlude = false;
          for (int t = 0; t < result.size(); t++) { //遍历result
            if (p.id_ == result[t].id_) { // duplicate
              occlude = true;
              break;
            }
            DistanceType djk = space_->get_distance(result[t].id_, p.id_); // 即 BC

            if (djk < p.distance_) { // BC<AC ，则C通过B可以很好地到达A，无需连接AC
              occlude = true;
              break;
            }
          }
          if (!occlude) {
            result.push_back(p);
          }
        } // while

        {
          std::scoped_lock guard(locks[des]);
          for (int t = 0; t < result.size(); t++) {
            graph.at(des, t) = result[t].id_; // 这一轮的结果作为des的新邻居
          }
        }

      } else { // tem_pool的size在限度内
        std::scoped_lock guard(locks[des]);
        for (int t = 0; t < max_nbrs_; t++) {
          if (graph.at(des, t) == Graph<DataType, IDType>::kEmptyId) { // des邻居末端插入q(reverse link)
            graph.at(des, t) = sn.id_; // sn.id_就是q
            break;
          }
        }
      }

    }
  }

  /**
   * @brief Grow a tree to ensure all nodes are connected in the graph.
   *
   * This function ensures that all nodes in the graph are connected by growing a tree.
   * It performs a depth-first search (DFS) to traverse the graph and attaches unlinked
   * nodes to the tree if they are not already connected.
   *
   * @param degrees A vector containing the degree of each node.
   * @return int The number of nodes that were attached to the tree.
   */
  auto tree_grow(std::vector<int> &degrees) -> int { // 返回新接入数
    int root = ep_;
    std::vector<bool> vis(vector_num_, false);
    int num_attached = 0;
    int cnt = 0;
    while (true) {
      cnt = dfs(vis, root, cnt);
      if (cnt >= vector_num_) {
        break;
      }
      std::vector<bool> vis2(vector_num_, false); // 单纯为了调用attach_unlinked而新建
      root = attach_unlinked(vis, vis2, degrees);
      num_attached += 1;
    }
    return num_attached;
  }

  /**
   * @brief Perform a depth-first search (DFS) to traverse the graph.
   *
   * This function performs a DFS to traverse the graph starting from the given root node.
   * It marks visited nodes and counts the number of nodes visited during the traversal.
   *
   * @param vis A vector indicating whether each node has been visited.
   * @param root The root node to start the DFS.
   * @param cnt The current count of visited nodes.
   * @return int The updated count of visited nodes.
   */
  auto dfs(std::vector<bool> &vis, IDType root, int cnt) const -> int {
    IDType node = root;
    std::stack<IDType> stack;
    stack.push(root);
    if (vis[root]) {
      cnt++;
    }
    vis[root] = true;
    while (!stack.empty()) {
      IDType next = Graph<DataType, IDType>::kEmptyId;
      for (int i = 0; i < max_nbrs_; i++) { // 找next,即下一个访问
        IDType id = final_graph_->at(node, i);
        if (id != Graph<DataType, IDType>::kEmptyId && !vis[id]) {
          next = id;
          break;
        }
      }
      if (next == Graph<DataType, IDType>::kEmptyId) { // 找不到next,就回溯(从stack中拿)
        stack.pop(); // pop掉当前node
        if (stack.empty()) { // dfs结束
          break;
        }
        node = stack.top(); 
        continue;
      }
      // 找到next,为了回溯，即使当前node visit后也要push stack
      node = next;
      vis[node] = true;
      stack.push(node);
      cnt++;
    }
    return cnt;
  }

  /**
   * @brief Attach unlinked nodes to the tree.
   *
   * This function attaches unlinked nodes to the tree by finding nodes that are not
   * connected and linking them to nodes with available degree slots. It ensures that
   * all nodes are eventually connected in the graph.
   *
   * @param vis A vector indicating whether each node has been visited.
   * @param vis2 A secondary vector indicating whether each node has been visited.
   * @param degrees A vector containing the degree of each node.
   * @return IDType The ID of the node that was attached.
   */
  auto attach_unlinked(std::vector<bool> &vis, std::vector<bool> &vis2,
                       std::vector<int> &degrees) -> IDType { // vis2完全没用
    IDType id = Graph<DataType, IDType>::kEmptyId;
    for (IDType i = 0; i < vector_num_; i++) {
      if (vis[i]) { // 找到一个dfs中已发现的node
        id = i;
        break;
      }
    }
    if (id == Graph<DataType, IDType>::kEmptyId) { // 找不到，直接返回
      return Graph<DataType, IDType>::kEmptyId;
    }
    std::vector<Neighbor<IDType>> tmp; //根本就没用，单纯为了可以调用search_on_graph
    std::vector<Node<IDType>> pool;
    // Searches for the nearest neighbors of a query vector (parameter 1) on the graph.
    search_on_graph<true>(space_->get_data_by_id(id), final_graph_, vis2, ep_, ef_construction_,
                          tmp, pool);
    std::sort(pool.begin(), pool.end());
    IDType node;
    bool found = false;
    for (int i = 0; i < pool.size(); i++) {
      node = pool[i].id_;
      if (degrees[node] < max_nbrs_ && node != id) { // 找一个degree还没满的node，距离id近优先
        found = true;
        break;
      }
    }
    if (!found) {
      do { // 一定能找到嘛
        node = rng_.rand_int(vector_num_);
        if (!vis[node] && degrees[node] < max_nbrs_ && node != id) { // 随机找一个符合条件的
          found = true;
        }
      } while (!found);
    }
    int pos = degrees[node];
    final_graph_->at(node, pos) = id;
    degrees[node] += 1;

    return node; // 返回接入处
  }

  /**
   * @brief Insert a node into the pool.
   *
   * @param pool The pool to insert the node into.
   * @param pool_size The size of the pool.
   * @param nn The node to insert.
   * @return int The position of the inserted node in the pool.
   */
  auto insert_into_pool(Neighbor<IDType> *pool, int pool_size,
                        const Neighbor<IDType> &nn) const -> int {
    for (int i = 0; i < pool_size; i++) { // check duplicate
      if (pool[i].id_ == nn.id_) {
        return pool_size;
      }
    }
    if (nn.distance_ >= pool[pool_size - 1].distance_) { // check if greater than the greatest in the pool
      return pool_size;
    }
    int pos = pool_size - 1;
    while (pos > 0 && nn.distance_ < pool[pos - 1].distance_) { // move and insert
      pool[pos] = pool[pos - 1];
      pos--;
    }
    pool[pos] = nn;
    return pos;
  }
};

}  // namespace alaya

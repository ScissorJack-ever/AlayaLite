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
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <queue>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>
#include "../../../space/space_concepts.hpp"
#include "../../../utils/log.hpp"
#include "visited_list_pool.hpp"

namespace alaya {

template <typename SpaceType, typename DataType = typename SpaceType::DataTypeAlias,
          typename DistanceType = typename SpaceType::DistanceTypeAlias,
          typename IDType = typename SpaceType::IDTypeAlias>
  requires Space<SpaceType, DataType, DistanceType, IDType>
class HNSWImpl {
  using InternalID = IDType;        ///< The internal id type of hnsw.
  using LinkListSizeType = IDType;  ///< The size of the linklist for graph.
  using ExternalID = IDType;        ///< The external id type of hnsw.

 public:
  static const InternalID kMaxLabelOperationLocks =
      65536;  ///< Maximum number of locks that can be held during operations that modify the graph
              ///< structure. i.e., 2^16
  size_t max_elements_{0};                            ///< Maximum number of elements for hnsw.
  mutable std::atomic<size_t> cur_element_count_{0};  ///< Current number of elements.

  size_t size_data_per_element_{0};  ///< The size of each element's data (includes internal id and
                                     ///< vector data) at upperlayer graph.
  size_t size_links_per_element_{0};  ///< The size of each element's data(only internal id) at overlayer graph
  
  size_t max_edge_num_{0};     ///< Maximum number of neighbors for node. 每个节点最大连接数（L0 层是两倍）
  size_t max_edge_num_l0_{0};  ///< Maximum number of neighbors for node at level 0.
  size_t ef_construction_{0};  ///< The maximum number of candidate neighbors for graph construction. 上面两个是实际Nei,这个只是candidate

  double mult_{0.0}, rev_size_{0.0};
  int maxlevel_{0};  ///< Currently max level for graph.

  VisitedListPool *visited_list_pool_{nullptr}; // 用于复用搜索过程中访问记录的对象池

  std::mutex global_;                        ///< Global lock for operating graph.
  std::vector<std::mutex> link_list_locks_;  ///< Lock the overlay graph for each.
  mutable std::vector<std::mutex>
      label_op_locks_;             ///< Locks operations with element by label value
  InternalID enterpoint_node_{0};  ///< The first enterpoint for hnsw.

  size_t offset_l0_{0}; // 就是0

  char *linklists_l0_memory_{nullptr};  ///< The vector data for level0 graph.
  char **link_lists_{nullptr};          ///< Store the overlay graph struction. 存储上层图结构（每层一个链接表）
  std::vector<int> element_levels_;     ///< keeps level of each element by internal id. 每个节点所在的层级

  std::shared_ptr<SpaceType> space_ =
      nullptr;                            ///< Unified manager vector data and distance computing.

  mutable std::mutex label_lookup_lock_;  ///< lock for label_lookup_
  std::unordered_map<ExternalID, InternalID>
      label_lookup_;                            ///< Mapping of external id and internal id
  std::vector<ExternalID> tableint_lookup_;     ///< Mapping of internal id and external id

  std::default_random_engine level_generator_;  ///< Generator a level for each node. 伪随机数生成器

  HNSWImpl(std::shared_ptr<SpaceType> &s, size_t max_elements, size_t max_edge_num = 16,
           size_t ef_construction = 200, size_t random_seed = 100)
      : link_list_locks_(max_elements),
        label_op_locks_(kMaxLabelOperationLocks),
        element_levels_(max_elements) {
    max_elements_ = max_elements;
    space_ = std::move(s);

    max_edge_num_ = max_edge_num;
    max_edge_num_l0_ = max_edge_num_ * 2;
    ef_construction_ = std::max(ef_construction, max_edge_num_);

    level_generator_.seed(random_seed);
    size_data_per_element_ = max_edge_num_l0_ * sizeof(InternalID) + sizeof(LinkListSizeType);
    offset_l0_ = 0;

    linklists_l0_memory_ = reinterpret_cast<char *>(malloc(max_elements_ * size_data_per_element_));
    memset(linklists_l0_memory_, 0, max_elements_ * size_data_per_element_);
    cur_element_count_ = 0;
    visited_list_pool_ = new VisitedListPool(1, max_elements);

    // initializations for special treatment of the first node
    enterpoint_node_ = -1;
    maxlevel_ = -1;

    link_lists_ = reinterpret_cast<char **>(malloc(sizeof(void *) * max_elements_)); // sizeof(void*) == sizeof(char*) 始终成立 
    size_links_per_element_ = max_edge_num_ * sizeof(InternalID) + sizeof(LinkListSizeType);
    mult_ = 1 / log(1.0 * max_edge_num_);
    rev_size_ = 1.0 / mult_;
    tableint_lookup_.resize(max_elements);
  }

  ~HNSWImpl() {
    delete visited_list_pool_;
    free(linklists_l0_memory_);
    for (InternalID i = 0; i < cur_element_count_; i++) {
      if (element_levels_[i] > 0) {
        free(link_lists_[i]);
      }
    }
    free(link_lists_);
  }

  /**
   * @brief A functor for comparing two pairs based on their first element.
   *
   */
  struct CompareByFirst { // 只比较dist
    constexpr auto operator()(std::pair<DistanceType, InternalID> const &a,
                              std::pair<DistanceType, InternalID> const &b) const noexcept -> bool {
      return a.first < b.first;
    }
  };

  /**
   * @brief Retrieve the node label of the required node in the HNSW graph.
   *
   * @param internal_id The internal id of required node in hnsw graph.
   * @return LabelType The label of node.
   */
  inline auto get_external_label(InternalID internal_id) const -> ExternalID { // 确定存在才会查找？
    auto label = tableint_lookup_[internal_id];
    return label;
  }

  /**
   * @brief Retrieves the mutex associated with the specified label for label operations.
   *
   * @param label The label value for which to retrieve the associated mutex.
   * @return Reference to the mutex associated with the specified label.
   */
  inline auto get_label_op_mutex(ExternalID label) const -> std::mutex & {
    size_t lock_id = label & (kMaxLabelOperationLocks - 1);
    return label_op_locks_[lock_id];
  }

  /**
   * @brief Calculate the maximum number of levels for the current insertion point.
   *
   * This function generates a random level based on the exponential distribution.
   * The level is determined by the formula:
   *   level = -log(U) * reverse_size,
   * where U is a uniformly distributed random number in the range [0, 1].
   * The `reverse_size` parameter controls the spread of the levels,
   * influencing how many levels can be assigned to the new node in the HNSW graph.
   *
   * @param reverse_size A scaling factor that affects the maximum number of levels.
   *                     A larger value increases the likelihood of generating higher levels.
   * @return int The randomly generated level for the current insertion point,
   *             which is a non-negative integer representing the level in the HNSW structure.
   */
  auto get_random_level(double reverse_size) -> size_t {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double r = -log(distribution(level_generator_)) * reverse_size; //reverse_size influences how many levels can be assigned to the new node in the HNSW graph.
    return static_cast<size_t>(r);
  }

  /**
   * @brief Retrieves a pointer to the link list at level 0 for a given internal ID.
   *
   * @param internal_id The internal ID used to identify the specific link list.
   */
  auto get_linklist0(InternalID internal_id) const -> LinkListSizeType * {
    return reinterpret_cast<LinkListSizeType *>(linklists_l0_memory_ +                 //首指针(指向第一个byte)
                                                internal_id * size_data_per_element_ + offset_l0_);
  }

  /**
   * @brief Sets the count of elements in the provided link list.
   *
   * This function writes the specified size value to the beginning of the link list
   * pointed to by the provided pointer. The size is stored as a `uint32_t` value.
   *
   * @param ptr Pointer to the link list where the size will be set.
   * @param size The number of elements to set in the link list.
   *
   * @note The function assumes that the pointer `ptr` points to a valid memory location
   *       where the size can be safely written.
   */
  auto set_list_count(LinkListSizeType *ptr, LinkListSizeType size) const -> void {
    *(reinterpret_cast<LinkListSizeType *>(ptr)) = size;
  }

  /**
   * @brief Retrieves the count of elements from the provided link list.
   *
   * This function reads and returns the count of elements stored at the beginning
   * of the link list pointed to by the provided pointer. The count is expected to
   * be stored as a `uint32_t` value.
   *
   * @param ptr Pointer to the link list from which the size will be retrieved.
   * @return The number of elements in the link list as a `uint32_t`.
   *
   * @note The function assumes that the pointer `ptr` points to a valid memory location
   *       where the size is stored.
   */
  auto get_list_count(const LinkListSizeType *ptr) const -> LinkListSizeType { return *(ptr); }

  /**
   * @brief Retrieves a pointer to the link list at a specified level for a given internal ID.
   *
   * This function calculates the memory address of the link list for the specified
   * internal ID and level. The result is cast to a pointer of type `LinkListSizeInt*`.
   *
   * @param internal_id The internal ID used to identify the specific link list.
   * @param level The level of the link list to retrieve (1-based index).
   * @return Pointer to the link list of type `LinkListSizeInt*`.
   *
   * @note The calculation involves accessing the link list array using the internal ID
   *       and adjusting for the specified level to retrieve the correct link list.
   */
  auto get_linklist(InternalID internal_id, int level) const -> LinkListSizeType * {
    return reinterpret_cast<LinkListSizeType *>(
        (link_lists_[internal_id] + (level - 1) * size_links_per_element_));
        //link_lists_[internal_id]指向节点internal_id在所有层级的连接信息起始地址,遍历它即是Internal_id在每一层的连接信息
        //(level - 1) * size_links_per_element_)即偏移
  }

  /**
   * @brief Retrieve the node ID of the required node in the HNSW graph.
   *
   * @param internal_id The internal id of required node in hnsw graph.
   * @return char* The raw vector data.
   */
  inline auto get_data_by_internal_id(InternalID internal_id) const -> char * {
    return reinterpret_cast<char *>(space_->get_data_by_id(get_external_label(internal_id)));
  }
  /**注释依托，看代码
   * @brief Selects the top M nearest neighbors from the given top candidates using a heuristic
   * approach.
   *
   * This function filters the top candidates based on their distances to ensure that only the most
   * relevant neighbors are retained. It uses a priority queue to manage the candidates and a vector
   * to store the selected neighbors.
   * The selection process is illustrated as follows:
   *       A (add point)
   *      / \
   *     /   \
   *    /     \
   *   B-------C
   *
   * In this diagram:
   * - A represents the query point.
   * - B and C are candidate points that are potential nearest neighbors.
   *
   * The distance selection process works as follows:
   * 1. Calculate the distances from the query point A to each candidate point:
   *    - Dist(A, B): Distance from A to B
   *    - Dist(A, C): Distance from A to C
   *    - Dist(B, C): Distance between candidates B and C
   *
   * 2. Compare these distances to determine which candidates to retain:
   *    - If Dist(A, B) < Dist(A, C), then B is closer to A than C.
   *
   * 3. To ensure diversity among selected neighbors, the function applies the following heuristic:
   *    - If Dist(A, B) < Dist(A, C) < Dist(B, C), we will discard node C.
   *      This means that if B is the closest to A （AB<AC） and also closer to A than C is to B (AB<CB), then C is
   * not a suitable neighbor.
   *    - Conversely, if Dist(A, C) < Dist(A, B) < Dist(B, C), we will discard node B.
   * @param top_candidates A priority queue containing the top candidates (nearest neighbors) found
   * during the search, sorted by distance.
   * @param m The maximum number of neighbors to retain.
   */
  void get_neighbors_by_heuristic2( // 确保邻居之间尽量分散, 利用上述pruning rule将top_candidates的size削减至m
      std::priority_queue<std::pair<DistanceType, InternalID>,
                          std::vector<std::pair<DistanceType, InternalID>>, CompareByFirst>
          &top_candidates,
      const size_t m) {
    // If the number of top candidates is less than m, no filtering is necessary.
    if (top_candidates.size() < m) {
      return;
    }

    // Create a temporary priority queue to hold the closest candidates.
    std::priority_queue<std::pair<DistanceType, InternalID>> queue_closest;
    // Vector to store the selected neighbors.
    std::vector<std::pair<DistanceType, InternalID>> return_list;

    // Transfer all top candidates to the queue_closest, negating the distances for max-heap
    // behavior.
    while (top_candidates.size() > 0) {
      queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
      top_candidates.pop();
    }

    // Iterate through the closest candidates to filter and select the best ones.
    while (queue_closest.size()) { //记B为selected nei,C为cur_nei, A为query
      // If we have already selected m neighbors, exit the loop.
      if (return_list.size() >= m) {
        break;
      }

      // Get the current candidate from the queue.
      std::pair<DistanceType, InternalID> current_pair = queue_closest.top();
      //dist_to_query是按从大到小取出的(因为取负了)，负数大则绝对值小，也即dist从小到大被取出，有AB<AC
      DistanceType dist_to_query = -current_pair.first;  // Get the distance to the query point.
      queue_closest.pop();  // Remove the current candidate from the queue.
      bool good = true;     // Flag to determine if the candidate is suitable.

      // Check the current candidate against the already selected neighbors.
      for (std::pair<DistanceType, InternalID> second_pair : return_list) {
        // Calculate the distance between the current candidate and the existing neighbor.
        // BC即curdist
        DistanceType curdist = space_->get_distance(
            get_external_label(second_pair.second),  // Get data for the existing neighbor.
            get_external_label(current_pair.second)  // Get data for the current candidate.
        );  // Additional parameters for the distance function.

        // If the current candidate is closer to an existing neighbor, mark it as not good.
        if (curdist < dist_to_query) { // BC<AC
          good = false;  // Current candidate is not suitable.
          break;         // Exit the loop since we found a closer neighbor.
        }
      }

      // If the candidate is good, add it to the return list.
      if (good) {
        return_list.push_back(current_pair);
      }
    }

    // Restore the selected neighbors back to the top_candidates priority queue.
    for (std::pair<DistanceType, InternalID> current_pair : return_list) { // 只restore 这轮的topm, 削减了一轮
      top_candidates.emplace(-current_pair.first,
                             current_pair.second);  // Negate distance for max-heap behavior.
    }
  }

  /**
   * @brief Search for the nearest neighbors in the specified layer of the HNSW graph.
   *
   * This function performs a search in the base layer of the HNSW graph to find the top candidates
   * (nearest neighbors) for a given data point. It uses a priority queue to maintain the best
   * candidates and a candidate set to explore potential neighbors. The search is performed using a
   * distance function to evaluate proximity.
   *
   * @param ep_id The internal ID of the entry point node from which the search begins.
   * @param data_point A pointer to the data point for which nearest neighbors are being searched.
   * @param layer The layer of the HNSW graph to search in. Typically, this will be 0 for the base
   * layer.
   * @return std::priority_queue<std::pair<DistType, TableInt>,
   *                             std::vector<std::pair<DistType, TableInt>>,
   *                             CompareByFirst> A priority queue containing the top candidates
   * found during the search, sorted by distance.
   */
   // 在layer中找对应data_label的top-k-nearest neighbors,其中 k = ef_construction
  auto search_base_layer(InternalID enterpoint_id, ExternalID data_label, uint32_t layer)
      -> std::priority_queue<std::pair<DistanceType, InternalID>,
                             std::vector<std::pair<DistanceType, InternalID>>, CompareByFirst> {
    // Obtain a free visited list from the pool to track visited nodes during the search.
    VisitedList *vl = visited_list_pool_->get_free_visited_list();
    vl_type *visited_array = vl->mass_;
    vl_type visited_array_tag = vl->cur_v_;
    // Priority queues to store top candidates and the current candidate set.
    std::priority_queue<std::pair<DistanceType, InternalID>,
                        std::vector<std::pair<DistanceType, InternalID>>, CompareByFirst>
        top_candidates;
    std::priority_queue<std::pair<DistanceType, InternalID>,
                        std::vector<std::pair<DistanceType, InternalID>>, CompareByFirst>
        candidate_set;
    
    // 以entry_point初始化
    DistanceType lower_bound;  // Variable to track the lower bound of distances.
    // Calculate the distance from the data point to the entry point node and initialize queues.
    DistanceType dist = space_->get_distance(data_label, get_external_label(enterpoint_id));
    top_candidates.emplace(dist, enterpoint_id);  // Add the entry point as the first candidate.
    lower_bound = dist;                           // Set the initial lower bound.
    candidate_set.emplace(
        -dist,
        enterpoint_id);  // Add to candidate set with negative distance for max-heap behavior.

    // Mark the entry point as visited.
    visited_array[enterpoint_id] = visited_array_tag;

    // Main search loop to explore candidates.
    while (!candidate_set.empty()) {
      std::pair<DistanceType, InternalID> curr_el_pair =
          candidate_set.top();  // Get the current candidate.

      // Break the loop if the current candidate's distance exceeds the lower bound and we have
      // enough candidates.
      if ((-curr_el_pair.first) > lower_bound && top_candidates.size() == ef_construction_) { 
        // curr_el_pair来自candidate_set(存储负数),所以此处取负
        // 后面的距离只会更大，但没招满还是要破底线录入，即使招满了也要找到破底线为止。 合理认为应该改为||
        break; 
      }
      candidate_set.pop();  // Remove the current candidate from the set.

      InternalID cur_node_num = curr_el_pair.second;  // Get the current node's ID.

      // Lock the link list for the current node to ensure thread safety.
      std::unique_lock<std::mutex> lock(link_list_locks_[cur_node_num]);

      LinkListSizeType *data;
      // Get the link list based on the specified layer.
      if (layer == 0) {
        data = get_linklist0(cur_node_num);
      } else {
        data = get_linklist(cur_node_num, layer);
      }

      // 可以推断data指向的首个IDType表示candidate size, 后面都是candid id
      LinkListSizeType size = get_list_count( 
          reinterpret_cast<LinkListSizeType *>(data));  // Get the size of the link list.
      auto *datal = reinterpret_cast<InternalID *>(
          data + 1);  // Pointer to the candidate IDs in the link list.

// prefetch里面是地址, 建议预取 1~2 个后续项即可
#ifdef USE_SSE
      // Prefetch data for performance optimization.
      _mm_prefetch(reinterpret_cast<char *>(visited_array + *(data + 1)), _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<char *>(visited_array + *(data + 1) + 64), _MM_HINT_T0);
      _mm_prefetch(get_data_by_internal_id(*datal), _MM_HINT_T0);
      _mm_prefetch(get_data_by_internal_id(*(datal + 1)), _MM_HINT_T0);
#endif

      // Iterate through the candidates in the link list.
      for (size_t j = 0; j < size; j++) {
        InternalID candidate_id = *(datal + j);  // Get the candidate ID.
#ifdef USE_SSE
        if (j < size - 1) { //还需要预取后面的元素
          _mm_prefetch(reinterpret_cast<char *>(visited_array + *(datal + j + 1)), _MM_HINT_T0);
          _mm_prefetch(get_data_by_internal_id(*(datal + j + 1)), _MM_HINT_T0);
        }
#endif

        // Skip if the candidate has already been visited.
        if (visited_array[candidate_id] == visited_array_tag) {
          continue;
        }

        visited_array[candidate_id] = visited_array_tag;  // Mark the candidate as visited.
        char *curr_obj1 = (get_data_by_internal_id(candidate_id));  // Get data for the candidate.

        // Calculate the distance to the current candidate.
        DistanceType dist1 = space_->get_distance(data_label, get_external_label(candidate_id));
        // If the candidate is a better match, update the candidate sets.
        if (top_candidates.size() < ef_construction_ || lower_bound > dist1) { //合理且还要加
          // 负数大根堆，即堆顶为最小距离
          candidate_set.emplace(-dist1, candidate_id);  // Add candidate to the candidate set.

#ifdef USE_SSE
          _mm_prefetch(get_data_by_internal_id(candidate_set.top().second), _MM_HINT_T0);
#endif
          top_candidates.emplace(dist1, candidate_id);  // Add candidate to the top candidates.

          // Maintain the size of the top candidates.
          if (top_candidates.size() > ef_construction_) { // 大根堆，即距离最大的candidate在堆顶，方便remove
            top_candidates.pop();  // Remove the worst candidate if exceeding size.
          }

          // Update the lower bound based on the top candidates.
          if (!top_candidates.empty()) { //top k min,
            lower_bound = top_candidates.top().first;
          }
        }
      } // for
    } // while

    // Release the visited list back to the pool.
    visited_list_pool_->release_visited_list(vl);

    // Return the top candidates found during the search.
    return top_candidates;
  }

  /**
   * @brief Mutually connects a new element to its selected neighbors in the HNSW graph.
   *
   * This function establishes connections between a newly added element and its nearest neighbors.
   * It updates the link lists of both the new element and its selected neighbors, ensuring that
   * all relevant connections are maintained in the graph structure. The function also handles
   * updates to existing connections if the element is being updated rather than added.
   *
   * @param unused A pointer to unused data; kept for compatibility with other signatures.
   * @param cur_c The internal ID of the current element being added or updated.
   * @param top_candidates A priority queue containing the top candidates (nearest neighbors)
   *                      found during the search. This queue is used to determine which neighbors
   *                      to connect to.
   * @param level The level in the HNSW graph where the connections are being made.
   *              Typically, level 0 represents the base layer.
   * @param isUpdate A boolean flag indicating whether the operation is an update (true)
   *                 or an addition (false). This affects how locks are managed during the
   * operation.
   * @return TableInt The internal ID of the next closest entry point in the graph, which
   *                  can be used for further operations or searches.
   */
  auto mutually_connect_new_element(
      InternalID cur_c,
      std::priority_queue<std::pair<DistanceType, InternalID>,
                          std::vector<std::pair<DistanceType, InternalID>>, CompareByFirst>
          &top_candidates,
      int level, bool isUpdate) -> InternalID {
    // Determine the maximum number of edges for the current level.
    size_t mcurmax = (level != 0) ? max_edge_num_ : max_edge_num_l0_;

    // prune and transfer
    // Retrieve neighbors based on heuristic to fill the top_candidates queue.
    get_neighbors_by_heuristic2(top_candidates, max_edge_num_);
    // Prepare a vector to hold the selected neighbors from the top candidates.
    std::vector<InternalID> selected_neighbors;
    selected_neighbors.reserve(max_edge_num_);  // Reserve space for efficiency.
    // Extract neighbors from the priority queue into the selected_neighbors vector.
    while (top_candidates.size() > 0) {
      // LOG_INFO("mutally first : u {} , v {} , dist {}", get_external_label(cur_c),
      //  top_candidates.top().second, top_candidates.top().first);
      selected_neighbors.push_back(top_candidates.top().second);
      top_candidates.pop();
    }

    // Get the closest entry point from the selected neighbors. 用于返回（中间并无改动）
    InternalID next_closest_entry_point = selected_neighbors.back(); //front处的距离最大，back处的距离最小

    {// Copy the selected neighbors into the link list.
      // Lock the link list for the current element during the update.
      // If adding a new element, the lock for cur_c is already acquired.
      std::unique_lock<std::mutex> lock(link_list_locks_[cur_c], std::defer_lock);
      if (isUpdate) {
        lock.lock();  // Lock only if this is an update.
      }

      // Get the link list for the current element based on the level.
      LinkListSizeType *ll_cur;
      if (level == 0) {
        ll_cur = get_linklist0(cur_c);
      } else {
        ll_cur = get_linklist(cur_c, level);
      }

      // Set the count of selected neighbors in the link list.
      set_list_count(ll_cur, selected_neighbors.size());
      auto *data = static_cast<InternalID *>(ll_cur + 1);  // Pointer to the data after the count.

      // Copy the selected neighbors into the link list.
      for (size_t idx = 0; idx < selected_neighbors.size(); idx++) {
        data[idx] = selected_neighbors[idx];
      }
    }

    // Iterate over each selected neighbor to establish mutual connections.
    // 对每个nei, 如果与cur_c没有连接，则想方设法把它与cur_c的连接加上
    for (auto &selected_neighbor : selected_neighbors) {
      std::unique_lock<std::mutex> lock(
          link_list_locks_[selected_neighbor]);  // Lock the neighbor's link list.

      // Get the link list for the selected neighbor based on the level.
      LinkListSizeType *ll_other;
      if (level == 0) {
        ll_other = get_linklist0(selected_neighbor);
      } else {
        ll_other = get_linklist(selected_neighbor, level);
      }

      size_t sz_link_list_other = get_list_count(ll_other);  // Get the size of the neighbor's link list.
      auto *data = static_cast<InternalID *>(ll_other + 1);  // Pointer to the neighbor's data.

      bool is_cur_c_present = false;  // Flag to check if cur_c is already connected.

      // Check if cur_c is already present in the neighbor's connections if this is an update.
      if (isUpdate) {
        for (size_t j = 0; j < sz_link_list_other; j++) {
          if (data[j] == cur_c) {
            is_cur_c_present = true;  // Found cur_c in the neighbor's connections.
            break;
          }
        }
      }

      // If cur_c is not already present, establish the connection.
      if (!is_cur_c_present) {
        if (sz_link_list_other < mcurmax) {
          // If the neighbor's link list is not full, add cur_c directly.
          data[sz_link_list_other] = cur_c;
          set_list_count(ll_other, sz_link_list_other + 1);
        } else {
          // If the neighbor's link list is full, find the weakest connection to replace.
          DistanceType d_max = space_->get_distance(get_external_label(cur_c),
                                                    get_external_label(selected_neighbor));
          // Use a priority queue to find the weakest connection.
          std::priority_queue<std::pair<DistanceType, InternalID>,
                              std::vector<std::pair<DistanceType, InternalID>>, CompareByFirst> candidates;
          candidates.emplace(d_max, cur_c);
          // Calculate distances to existing neighbors and add them to the candidates queue.
          for (size_t j = 0; j < sz_link_list_other; j++) {
            candidates.emplace(space_->get_distance(get_external_label(data[j]),
                                                    get_external_label(selected_neighbor)),
                               data[j]);
          }
          // Retrieve the best neighbors based on the heuristic.  不怕把cur_c给prune了嘛
          get_neighbors_by_heuristic2(candidates, mcurmax);

          // Replace the weakest connections in the neighbor's link list with the new connection.
          int index = 0;
          while (candidates.size() > 0) { // 一边插入一边计数，后面更新这个nei的link_list
            data[index] =
                candidates.top().second;  // Update the link list with the new connections.
            candidates.pop();
            index++;
          }

          set_list_count(ll_other, index);  // Update the count of connections.
        } //else
      }

    } //for

    // Return the next closest entry point for further processing.
    return next_closest_entry_point;
  }

  /**
   * @brief Adds a new point to the HNSW graph.
   *
   * This function inserts a data point into the graph structure, associating it with a specified
   * label. The label serves as a unique identifier for the node (or vector) within the graph,
   * allowing for efficient retrieval and management of the data points.
   *
   * @param data_point A pointer to the data of the point to be added. This data represents the
   *                   vector or feature set associated with the node in the graph.
   * @param label      The identifier or tag associated with the node. This label is used to
   * uniquely identify the point within the graph and may be used for further operations such as
   * searching or retrieval.
   * @return InternalID The internal ID of the newly added point in the graph, which can be used
   *                    for future references to this point.
   */
  auto add_point(ExternalID label) -> InternalID {
    std::unique_lock<std::mutex> lock_label(get_label_op_mutex(label));

    InternalID internal_id = 0;
    {
      // Checking if the element with the same label already exists
      // if so, updating it *instead* of creating a new element.
      std::unique_lock<std::mutex> lock_table(label_lookup_lock_);
      auto search = label_lookup_.find(label);
      if (search != label_lookup_.end()) { // alreadt exist, quit inserting 
        InternalID existing_internal_id = search->second;
        lock_table.unlock();
        return existing_internal_id;
      }
      // Mapping of external id and internal id.
      internal_id = cur_element_count_; 
      cur_element_count_++;
      label_lookup_[label] = internal_id;
      tableint_lookup_[internal_id] = label; // label就是external id
    }

    std::unique_lock<std::mutex> lock_el(link_list_locks_[internal_id]);
    // Calculate the maximum number of levels for the current insertion point.
    int cur_level = get_random_level(mult_);
    element_levels_[internal_id] = cur_level;

    std::unique_lock<std::mutex> templock(global_);
    int maxlevel_copy = maxlevel_; // 保留max_level_的原始值，后面可能会改变max_level_
    if (cur_level <= maxlevel_copy) {
      templock.unlock();
    }

    InternalID curr_node = enterpoint_node_;

    if (cur_level != 0) {
      // Allocate storage space for upper-level graphs.
      // size_links_per_element_ * cur_level表示该节点在所有层级（从 Level 1 到 Level cur_level）所需的空间总和。
      link_lists_[internal_id] =
          static_cast<char *>(malloc(size_links_per_element_ * cur_level + 1)); //+1是用来写size的
      memset(link_lists_[internal_id], 0, size_links_per_element_ * cur_level + 1); // initial value 0
    }

    if (curr_node != -1) { // 有顶层entry point
      if (cur_level < maxlevel_copy) {
        DistanceType curdist = space_->get_distance(label, get_external_label(curr_node));
        for (int level = maxlevel_copy; level > cur_level; level--) { // 顶层到cur_level遍历，主要是更新curr_node
          bool changed = true;
          while (changed) {
            changed = false;
            LinkListSizeType *data;
            std::unique_lock<std::mutex> lock(link_list_locks_[curr_node]);
            data = get_linklist(curr_node, level);
            LinkListSizeType size = get_list_count(data);

            auto *datal = static_cast<InternalID *>(reinterpret_cast<LinkListSizeType *>(data) + 1);
            for (int i = 0; i < size; i++) {
              InternalID cand = datal[i];
              DistanceType d = space_->get_distance(label, get_external_label(cand));
              if (d < curdist) { // 在该层不断找curr_node邻居中离插入点最近的点，将其作为新的curr_node继续找，直至止于此curr_node(就到下一层继续)
                curdist = d;
                curr_node = cand;
                changed = true;
              }
            }
          }
        } 
      } 

      // Update the structure of each layer of the graph
      for (int level = std::min(cur_level, maxlevel_copy); level >= 0; level--) { // 处理cur_level下层
        std::priority_queue<std::pair<DistanceType, InternalID>,
                            std::vector<std::pair<DistanceType, InternalID>>, CompareByFirst>
            top_candidates = search_base_layer(curr_node, label, level);

        // {
        //   auto candidata = search_base_layer(curr_obj, label, level);
        //   while (!candidata.empty()) {
        //     auto candidata_v = candidata.top();
        //     candidata.pop();

        //     LOG_INFO("the node id = {} , candidata id = {} , dist = {}", label,
        //              get_external_label(candidata_v.second), candidata_v.first);
        //   }
        // }

        // curr_node变为internal_id的closest neighbor（在level层）
        curr_node = mutually_connect_new_element(internal_id, top_candidates, level, false);
      } // for
    } else { // no entry point
      // Do nothing for the first element
      enterpoint_node_ = 0;
      maxlevel_ = cur_level;
    }

    // Releasing lock for the maximum level
    if (cur_level > maxlevel_copy) { // 新点在最高层(独此一个)，将其设为enterpoint_node
      enterpoint_node_ = internal_id;
      maxlevel_ = cur_level;
    }
    return internal_id;
  }
};

}  // namespace alaya

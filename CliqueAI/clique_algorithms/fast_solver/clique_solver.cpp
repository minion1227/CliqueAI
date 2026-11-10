// clique_solver.cpp - fast maximum-clique solver for CliqueAI (Bittensor SN83) miners.
//
// Reads a CliqueAI GraphCodec base92-encoded adjacency matrix on stdin and searches for maximum
// cliques until the time limit expires.
//
//   ls     (default) Phased Local Search (Pullan 2006): cycles random / penalty / greedy selection
//          phases of DLS-MC style "expand + plateau swap" iterations, on several threads.
//          Every distinct maximal clique of the best size is recorded together with how often it
//          was visited, so the caller can return a rarely-visited one (CliqueAI rewards unique
//          answers).
//   exact  Bitset branch and bound with greedy-colouring bounds (MCQ/BBMC style), warm-started by
//          local search. With --enumerate it collects every maximum clique. For offline
//          ground truth; too slow for big tiers inside the time limits.
//   dump   Print the decoded adjacency matrix (used by tests to check the decoder).
//
// Output protocol on stdout, one record per line:
//   N <n>                          number of vertices
//   B <size> <ms> <v1> ... <vk>    new best maximal clique, flushed at once (usable if killed)
//   R <hits> <v1> ... <vk>         final distinct best cliques (verified), rarest first
//   S key=value ...                statistics
//   E                              end of output
//   X <message>                    error (exit code != 0)

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

// ------------------------------------------------------------------------------------------ rng

struct Rng {  // xoshiro256**
  uint64_t s[4];
  explicit Rng(uint64_t seed) {
    for (auto& x : s) x = SplitMix(seed);
  }
  static uint64_t SplitMix(uint64_t& x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }
  static uint64_t Rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
  uint64_t operator()() {
    const uint64_t result = Rotl(s[1] * 5, 7) * 9;
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = Rotl(s[3], 45);
    return result;
  }
  // Uniform integer in [0, k), k > 0.
  uint32_t Below(uint32_t k) { return static_cast<uint32_t>(((*this)() >> 32) * k >> 32); }
};

// ---------------------------------------------------------------------------------------- graph

struct Graph {
  int n = 0;
  int W = 0;                   // 64-bit words per adjacency row
  std::vector<uint64_t> bits;  // n * W adjacency bitsets
  std::vector<int> non_start;  // CSR offsets into non_list, size n + 1
  std::vector<int> non_list;   // non-neighbours of each vertex (self excluded)

  const uint64_t* row(int v) const { return bits.data() + static_cast<size_t>(v) * W; }
  bool adj(int u, int v) const { return (row(u)[v >> 6] >> (v & 63)) & 1ULL; }
  void set_edge(int u, int v) {
    bits[static_cast<size_t>(u) * W + (v >> 6)] |= 1ULL << (v & 63);
    bits[static_cast<size_t>(v) * W + (u >> 6)] |= 1ULL << (u & 63);
  }
  void BuildNonNeighbours() {
    non_start.assign(n + 1, 0);
    non_list.clear();
    for (int v = 0; v < n; ++v) {
      non_start[v] = static_cast<int>(non_list.size());
      for (int u = 0; u < n; ++u)
        if (u != v && !adj(v, u)) non_list.push_back(u);
    }
    non_start[n] = static_cast<int>(non_list.size());
  }
  int64_t edges() const {
    int64_t twice = 0;
    for (uint64_t w : bits) twice += __builtin_popcountll(w);
    return twice / 2;
  }
};

// True iff `c` (sorted, distinct) is a clique that no vertex can extend: the validator's rule.
bool IsMaximalClique(const Graph& g, const std::vector<int>& c) {
  if (c.empty()) return false;
  for (size_t i = 0; i < c.size(); ++i) {
    if (c[i] < 0 || c[i] >= g.n || (i > 0 && c[i] <= c[i - 1])) return false;
    for (size_t j = i + 1; j < c.size(); ++j)
      if (!g.adj(c[i], c[j])) return false;
  }
  for (int w = 0; w < g.W; ++w) {
    uint64_t common = ~0ULL;
    for (int v : c) common &= g.row(v)[w];
    if (common) return false;  // some vertex is adjacent to every member
  }
  return true;
}

// ---------------------------------------------------------------------- base92 (GraphCodec)

constexpr int kHdrDigits = 5;
constexpr int kChunkBits = 1246;
constexpr int kChunkDigits = 191;

class Base92 {
 public:
  Base92() {
    std::fill(std::begin(index_), std::end(index_), -1);
    int k = 0;
    for (int c = 32; c < 127; ++c) {
      if (c == 34 || c == 39 || c == 92) continue;  // ", ', backslash
      index_[c] = k++;
    }
    std::vector<uint32_t> power{1};  // exact bit length of 92^m
    for (int m = 0; m <= kChunkDigits + 1; ++m) {
      pow_bitlen_.push_back(BitLen(power));
      MulAdd(power, 92, 0);
    }
  }
  int Digit(unsigned char c) const { return index_[c]; }
  // Smallest m with 92^m >= 2^r, as GraphCodec._min_digits_for_bits. 92^m is never a power of
  // two for m >= 1, so the condition is bitlen(92^m) >= r + 1.
  int MinDigitsForBits(int r) const {
    if (r <= 0) return 0;
    for (int m = 1; m < static_cast<int>(pow_bitlen_.size()); ++m)
      if (pow_bitlen_[m] >= r + 1) return m;
    return -1;
  }
  static void MulAdd(std::vector<uint32_t>& big, uint32_t mul, uint32_t add) {
    uint64_t carry = add;
    for (auto& limb : big) {
      const uint64_t x = static_cast<uint64_t>(limb) * mul + carry;
      limb = static_cast<uint32_t>(x);
      carry = x >> 32;
    }
    if (carry) big.push_back(static_cast<uint32_t>(carry));
  }
  static int BitLen(const std::vector<uint32_t>& big) {
    for (int i = static_cast<int>(big.size()) - 1; i >= 0; --i)
      if (big[i]) return i * 32 + (32 - __builtin_clz(big[i]));
    return 0;
  }

 private:
  int index_[256];
  std::vector<int> pow_bitlen_;
};

bool Decode(const std::string& s, Graph& g, std::string& err) {
  static const Base92 b92;
  if (s.size() < static_cast<size_t>(kHdrDigits)) return err = "truncated header", false;
  int64_t n64 = 0;
  for (int i = 0; i < kHdrDigits; ++i) {
    const int d = b92.Digit(static_cast<unsigned char>(s[i]));
    if (d < 0) return err = "invalid base92 char in header", false;
    n64 = n64 * 92 + d;
  }
  if (n64 > 50000) return err = "graph too large: n=" + std::to_string(n64), false;
  const int n = static_cast<int>(n64);
  const int64_t total_bits = n > 1 ? static_cast<int64_t>(n) * (n - 1) / 2 : 0;
  int64_t full_left = total_bits / kChunkBits;
  int rem_left = static_cast<int>(total_bits % kChunkBits);
  const int64_t expected =
      kHdrDigits + full_left * kChunkDigits + b92.MinDigitsForBits(rem_left);
  if (static_cast<int64_t>(s.size()) != expected)
    return err = "length mismatch: got " + std::to_string(s.size()) + ", expect " +
                 std::to_string(expected) + " for n=" + std::to_string(n),
           false;

  g.n = n;
  g.W = (n + 63) / 64;
  g.bits.assign(static_cast<size_t>(n) * g.W, 0);

  size_t pos = kHdrDigits;
  std::vector<uint32_t> chunk;
  int bits_left = 0;
  auto next_chunk = [&]() -> bool {
    int digits;
    if (full_left > 0) {
      digits = kChunkDigits;
      bits_left = kChunkBits;
      --full_left;
    } else if (rem_left > 0) {
      digits = b92.MinDigitsForBits(rem_left);
      bits_left = rem_left;
      rem_left = 0;
    } else {
      return false;
    }
    chunk.assign(1, 0);
    for (int k = 0; k < digits; ++k) {
      const int d = b92.Digit(static_cast<unsigned char>(s[pos + k]));
      if (d < 0) return false;
      Base92::MulAdd(chunk, 92, static_cast<uint32_t>(d));
    }
    pos += digits;
    return true;
  };
  // Upper-triangle bits in row-major order, most significant bit of each chunk first.
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      if (bits_left == 0 && !next_chunk()) return err = "invalid or truncated chunk", false;
      const int k = --bits_left;
      if (k / 32 < static_cast<int>(chunk.size()) && ((chunk[k / 32] >> (k % 32)) & 1u))
        g.set_edge(i, j);
    }
  }
  g.BuildNonNeighbours();
  return true;
}

// --------------------------------------------------------------------------- shared state

struct SharedState {
  Clock::time_point start;
  Clock::time_point deadline;
  std::atomic<int> best{0};
  std::atomic<bool> stop{false};
  int target = 0;  // stop once a clique this large is found (0 = run to the deadline)
  std::mutex out_mu;

  double ElapsedMs() const {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
  }
  bool Expired() const {
    return stop.load(std::memory_order_relaxed) || Clock::now() >= deadline;
  }
  // Raise the global best if `clique` beats it and print a B record.
  void Publish(const std::vector<int>& clique) {
    const int size = static_cast<int>(clique.size());
    int cur = best.load();
    bool raised = false;
    while (size > cur) {
      if (best.compare_exchange_weak(cur, size)) {
        raised = true;
        break;
      }
    }
    if (!raised) return;
    std::lock_guard<std::mutex> lock(out_mu);
    std::printf("B %d %.1f", size, ElapsedMs());
    for (int v : clique) std::printf(" %d", v);
    std::printf("\n");
    std::fflush(stdout);
    if (target > 0 && size >= target) stop = true;
  }
};

// ---------------------------------------------------------------------------- local search

enum class Phase { kRandom, kPenalty, kGreedy };

// Iterations per phase in one PLS cycle: random, penalty, greedy.
// Default is random selection only. On CliqueAI-like G(900, 0.735) graphs it reached the best
// clique with median 1.4 s / p90 3.3 s single-threaded, against 3.0 s / 7.8 s for the original
// PLS schedule (50,50,100), and it visited ~20% more distinct maximum cliques. Penalty phases
// help on structured graphs with hidden cliques (DIMACS brock), so the schedule is configurable.
struct Schedule {
  int random = 1;
  int penalty = 0;
  int greedy = 0;
};

class LocalSearch {
 public:
  LocalSearch(const Graph& g, SharedState& shared, uint64_t seed, int store_cap,
              int penalty_delay, Schedule schedule)
      : g_(g),
        sh_(shared),
        rng_(seed),
        n_(g.n),
        store_cap_(store_cap),
        pd_(std::max(1, penalty_delay)),
        schedule_(schedule),
        pos_c_(n_, -1),
        miss_(n_, 0),
        add_pos_(n_, -1),
        swap_pos_(n_, -1),
        pen_(n_, 0),
        tabu_(n_, 0),
        orig_(n_, 0),
        zob_(n_),
        add_bits_(g.W, 0) {
    for (auto& z : zob_) z = rng_();
    c_.reserve(n_);
    add_.reserve(n_);
    swap_.reserve(n_);
    for (int v = 0; v < n_; ++v) AddInsert(v);
  }

  void Run() {
    const struct {
      Phase phase;
      int iterations;
    } steps[] = {{Phase::kRandom, schedule_.random},
                 {Phase::kPenalty, schedule_.penalty},
                 {Phase::kGreedy, schedule_.greedy}};
    if (n_ == 0) return;
    if (schedule_.random + schedule_.penalty + schedule_.greedy <= 0) return;
    while (true) {
      for (const auto& step : steps) {
        for (int it = 0; it < step.iterations; ++it) {
          if (sh_.Expired() || best_ == n_) return;
          Iterate(step.phase);
          Perturb(step.phase);
        }
      }
    }
  }

  int best() const { return best_; }
  const std::vector<std::vector<int>>& cliques() const { return cliques_; }
  const std::vector<int>& hits() const { return hits_; }
  uint64_t iterations() const { return iterations_; }
  uint64_t dropped() const { return dropped_; }

 private:
  // --- indexed sets: add_ = {v not in C adjacent to all of C}, swap_ = {v missing exactly one}
  void AddInsert(int v) {
    add_pos_[v] = static_cast<int>(add_.size());
    add_.push_back(v);
    add_bits_[v >> 6] |= 1ULL << (v & 63);
  }
  void AddErase(int v) {
    const int i = add_pos_[v], last = add_.back();
    add_[i] = last;
    add_pos_[last] = i;
    add_.pop_back();
    add_pos_[v] = -1;
    add_bits_[v >> 6] &= ~(1ULL << (v & 63));
  }
  void SwapInsert(int v) {
    swap_pos_[v] = static_cast<int>(swap_.size());
    swap_.push_back(v);
  }
  void SwapErase(int v) {
    const int i = swap_pos_[v], last = swap_.back();
    swap_[i] = last;
    swap_pos_[last] = i;
    swap_.pop_back();
    swap_pos_[v] = -1;
  }

  // Precondition: v is in add_ (adjacent to every member of C).
  void AddVertex(int v) {
    AddErase(v);
    pos_c_[v] = static_cast<int>(c_.size());
    c_.push_back(v);
    hash_ ^= zob_[v];
    for (int k = g_.non_start[v]; k < g_.non_start[v + 1]; ++k) {
      const int w = g_.non_list[k];  // never in C: C is a clique and v is adjacent to all of it
      const int m = miss_[w]++;
      if (m == 0) {
        AddErase(w);
        SwapInsert(w);
      } else if (m == 1) {
        SwapErase(w);
      }
    }
  }

  void RemoveVertex(int v) {
    const int i = pos_c_[v], last = c_.back();
    c_[i] = last;
    pos_c_[last] = i;
    c_.pop_back();
    pos_c_[v] = -1;
    hash_ ^= zob_[v];
    for (int k = g_.non_start[v]; k < g_.non_start[v + 1]; ++k) {
      const int w = g_.non_list[k];
      const int m = --miss_[w];
      if (m == 1) {
        SwapInsert(w);
      } else if (m == 0) {
        SwapErase(w);
        AddInsert(w);
      }
    }
    AddInsert(v);  // v is adjacent to every remaining member
  }

  int Conflict(int v) const {  // the single member of C not adjacent to v (v in swap_)
    for (int u : c_)
      if (!g_.adj(v, u)) return u;
    return -1;
  }

  int SelectAdd(Phase p) {
    if (p == Phase::kRandom) return add_[rng_.Below(static_cast<uint32_t>(add_.size()))];
    int best_v = -1, best_key = INT_MIN;
    uint32_t ties = 0;
    for (int v : add_) {
      int key;
      if (p == Phase::kPenalty) {
        key = -pen_[v];
      } else {  // greedy: most neighbours inside the candidate set
        const uint64_t* r = g_.row(v);
        key = 0;
        for (int w = 0; w < g_.W; ++w) key += __builtin_popcountll(r[w] & add_bits_[w]);
      }
      if (key > best_key) {
        best_key = key;
        best_v = v;
        ties = 1;
      } else if (key == best_key && rng_.Below(++ties) == 0) {
        best_v = v;
      }
    }
    return best_v;
  }

  int SelectSwap(Phase p) {  // excludes vertices removed earlier in this plateau
    int best_v = -1, best_key = INT_MIN;
    uint32_t ties = 0;
    for (int v : swap_) {
      if (tabu_[v] == stamp_) continue;
      const int key = p == Phase::kPenalty ? -pen_[v] : 0;
      if (key > best_key) {
        best_key = key;
        best_v = v;
        ties = 1;
      } else if (key == best_key && rng_.Below(++ties) == 0) {
        best_v = v;
      }
    }
    return best_v;
  }

  // Store the current clique if it is maximal and at least as large as the best so far.
  void Record() {
    const int size = static_cast<int>(c_.size());
    if (size < best_) return;
    if (size > best_) {
      best_ = size;
      cliques_.clear();
      hits_.clear();
      seen_.clear();
    }
    const auto it = seen_.find(hash_);
    if (it != seen_.end()) {
      ++hits_[it->second];
      return;
    }
    if (static_cast<int>(cliques_.size()) >= store_cap_) {
      ++dropped_;
      return;
    }
    std::vector<int> sorted(c_.begin(), c_.end());
    std::sort(sorted.begin(), sorted.end());
    if (size > sh_.best.load(std::memory_order_relaxed)) sh_.Publish(sorted);
    seen_.emplace(hash_, static_cast<int>(cliques_.size()));
    cliques_.push_back(std::move(sorted));
    hits_.push_back(1);
  }

  // One DLS-MC iteration: expand until maximal, then plateau-swap while that keeps opening
  // new expansion moves.
  void Iterate(Phase p) {
    for (;;) {
      while (!add_.empty()) {
        const int v = SelectAdd(p);
        AddVertex(v);
        last_added_ = v;
      }
      Record();
      if (swap_.empty()) break;

      ++stamp_;
      for (int v : c_) orig_[v] = stamp_;
      int orig_left = static_cast<int>(c_.size());
      while (add_.empty() && orig_left > 0) {
        const int v = SelectSwap(p);
        if (v < 0) break;
        const int u = Conflict(v);
        if (orig_[u] == stamp_) --orig_left;
        RemoveVertex(u);
        tabu_[u] = stamp_;
        AddVertex(v);
        last_added_ = v;
        if (add_.empty()) Record();
      }
      if (add_.empty()) break;
    }
    ++iterations_;
  }

  void Perturb(Phase p) {
    if (static_cast<int>(c_.size()) >= n_) return;
    tmp_.assign(c_.begin(), c_.end());
    if (p == Phase::kPenalty) {
      for (int v : c_) ++pen_[v];
      if (++penalty_updates_ % pd_ == 0)
        for (int v = 0; v < n_; ++v)
          if (pen_[v] > 0) --pen_[v];
      // Restart from the most recently added vertex.
      const int keep = (last_added_ >= 0 && pos_c_[last_added_] >= 0) ? last_added_ : -1;
      for (int u : tmp_)
        if (u != keep) RemoveVertex(u);
      if (c_.empty()) AddVertex(add_[rng_.Below(static_cast<uint32_t>(add_.size()))]);
      return;
    }
    // Force in a random outside vertex and drop the members it is not adjacent to.
    int v;
    do {
      v = static_cast<int>(rng_.Below(static_cast<uint32_t>(n_)));
    } while (pos_c_[v] >= 0);
    for (int u : tmp_)
      if (!g_.adj(u, v)) RemoveVertex(u);
    AddVertex(v);
  }

  const Graph& g_;
  SharedState& sh_;
  Rng rng_;
  const int n_;
  const int store_cap_;
  const int pd_;
  const Schedule schedule_;

  std::vector<int> c_, pos_c_, miss_;
  std::vector<int> add_, add_pos_, swap_, swap_pos_;
  std::vector<int> pen_, tabu_, orig_, tmp_;
  std::vector<uint64_t> zob_, add_bits_;
  uint64_t hash_ = 0;
  int stamp_ = 0;
  int last_added_ = -1;
  uint64_t penalty_updates_ = 0;

  int best_ = 0;
  std::vector<std::vector<int>> cliques_;
  std::vector<int> hits_;
  std::unordered_map<uint64_t, int> seen_;
  uint64_t iterations_ = 0;
  uint64_t dropped_ = 0;
};

// ---------------------------------------------------------------------- exact branch & bound

class ExactSolver {
 public:
  ExactSolver(const Graph& g, SharedState& shared, bool enumerate, int store_cap)
      : g_(g), sh_(shared), enumerate_(enumerate), store_cap_(store_cap), W_(g.W) {
    // Renumber by non-increasing degree: colouring then starts from high-degree vertices.
    perm_.resize(g.n);
    for (int v = 0; v < g.n; ++v) perm_[v] = v;
    std::vector<int> degree(g.n);
    for (int v = 0; v < g.n; ++v) degree[v] = g.n - 1 - (g.non_start[v + 1] - g.non_start[v]);
    std::stable_sort(perm_.begin(), perm_.end(),
                     [&](int a, int b) { return degree[a] > degree[b]; });
    bits_.assign(static_cast<size_t>(g.n) * W_, 0);
    for (int i = 0; i < g.n; ++i)
      for (int j = 0; j < g.n; ++j)
        if (i != j && g.adj(perm_[i], perm_[j]))
          bits_[static_cast<size_t>(i) * W_ + (j >> 6)] |= 1ULL << (j & 63);
    levels_.assign(g.n + 2, std::vector<uint64_t>(W_, 0));
    order_.assign(g.n + 2, std::vector<int>(g.n));
    colour_.assign(g.n + 2, std::vector<int>(g.n));
    current_.assign(g.n + 1, 0);
  }

  // Search for cliques larger than `lower_bound` (or >= when enumerating).
  void Solve(int lower_bound) {
    best_ = lower_bound;
    if (g_.n == 0) return;
    auto& root = levels_[0];
    std::fill(root.begin(), root.end(), 0);
    for (int v = 0; v < g_.n; ++v) root[v >> 6] |= 1ULL << (v & 63);
    Expand(0);
    proved_ = !aborted_;
  }

  bool proved() const { return proved_; }
  int best() const { return best_; }
  uint64_t nodes() const { return nodes_; }
  const std::vector<std::vector<int>>& cliques() const { return cliques_; }

 private:
  const uint64_t* row(int v) const { return bits_.data() + static_cast<size_t>(v) * W_; }

  void Found(int size) {
    if (size > best_) {
      best_ = size;
      cliques_.clear();
    } else if (!(enumerate_ && size == best_)) {
      return;
    }
    if (static_cast<int>(cliques_.size()) >= store_cap_) return;
    std::vector<int> clique(size);
    for (int i = 0; i < size; ++i) clique[i] = perm_[current_[i]];
    std::sort(clique.begin(), clique.end());
    sh_.Publish(clique);
    cliques_.push_back(std::move(clique));
  }

  void Expand(int depth) {
    uint64_t* R = levels_[depth].data();
    int* order = order_[depth].data();
    int* colour = colour_[depth].data();

    // Greedy sequential colouring of R; vertices end up sorted by colour.
    std::vector<uint64_t>& U = scratch_u_;
    std::vector<uint64_t>& Q = scratch_q_;
    U.assign(R, R + W_);
    Q.resize(W_);
    int count = 0, k = 0;
    for (;;) {
      bool any = false;
      for (int w = 0; w < W_; ++w) any |= U[w] != 0;
      if (!any) break;
      ++k;
      std::copy(U.begin(), U.end(), Q.begin());
      for (int w = 0; w < W_; ++w) {
        while (Q[w]) {
          const int bit = __builtin_ctzll(Q[w]);
          const int v = (w << 6) | bit;
          Q[w] &= ~(1ULL << bit);
          U[w] &= ~(1ULL << bit);
          const uint64_t* r = row(v);
          for (int x = w; x < W_; ++x) Q[x] &= ~r[x];
          order[count] = v;
          colour[count] = k;
          ++count;
        }
      }
    }

    for (int i = count - 1; i >= 0; --i) {
      if ((++nodes_ & 1023) == 0 && sh_.Expired()) {
        aborted_ = true;
        return;
      }
      const int bound = depth + colour[i];
      if (enumerate_ ? bound < best_ : bound <= best_) return;
      const int v = order[i];
      current_[depth] = v;
      uint64_t* next = levels_[depth + 1].data();
      const uint64_t* r = row(v);
      bool empty = true;
      for (int w = 0; w < W_; ++w) {
        next[w] = R[w] & r[w];
        empty &= next[w] == 0;
      }
      if (empty) {
        Found(depth + 1);
      } else {
        Expand(depth + 1);
        if (aborted_) return;
      }
      R[v >> 6] &= ~(1ULL << (v & 63));
    }
  }

  const Graph& g_;
  SharedState& sh_;
  const bool enumerate_;
  const int store_cap_;
  const int W_;
  std::vector<int> perm_;
  std::vector<uint64_t> bits_;
  std::vector<std::vector<uint64_t>> levels_;
  std::vector<std::vector<int>> order_, colour_;
  std::vector<int> current_;
  std::vector<uint64_t> scratch_u_, scratch_q_;
  std::vector<std::vector<int>> cliques_;
  int best_ = 0;
  bool aborted_ = false;
  bool proved_ = false;
  uint64_t nodes_ = 0;
};

// ---------------------------------------------------------------------------------- driver

struct Options {
  double time_limit = 5.0;
  int threads = 1;
  uint64_t seed = 0;
  bool seed_set = false;
  int report = 256;
  int store_cap = 4096;
  int penalty_delay = 2;
  Schedule schedule;
  int target = 0;
  bool enumerate = false;
  std::string mode = "ls";
};

[[noreturn]] void Fail(const std::string& message, int code = 2) {
  std::printf("X %s\n", message.c_str());
  std::fflush(stdout);
  std::exit(code);
}

Options ParseArgs(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) Fail("missing value for " + a);
      return argv[++i];
    };
    try {
      if (a == "--time") o.time_limit = std::stod(value());
      else if (a == "--threads") o.threads = std::stoi(value());
      else if (a == "--seed") o.seed = std::stoull(value()), o.seed_set = true;
      else if (a == "--report") o.report = std::stoi(value());
      else if (a == "--store") o.store_cap = std::stoi(value());
      else if (a == "--penalty-delay") o.penalty_delay = std::stoi(value());
      else if (a == "--schedule") {  // "random,penalty,greedy" iterations per cycle
        const std::string v = value();
        const size_t c1 = v.find(','), c2 = v.find(',', c1 == std::string::npos ? c1 : c1 + 1);
        if (c1 == std::string::npos || c2 == std::string::npos) Fail("--schedule wants R,P,G");
        o.schedule.random = std::max(0, std::stoi(v.substr(0, c1)));
        o.schedule.penalty = std::max(0, std::stoi(v.substr(c1 + 1, c2 - c1 - 1)));
        o.schedule.greedy = std::max(0, std::stoi(v.substr(c2 + 1)));
      }
      else if (a == "--target") o.target = std::stoi(value());
      else if (a == "--mode") o.mode = value();
      else if (a == "--enumerate") o.enumerate = true;
      else Fail("unknown argument " + a);
    } catch (const std::exception&) {
      Fail("bad value for " + a);
    }
  }
  if (o.mode != "ls" && o.mode != "exact" && o.mode != "dump") Fail("unknown mode " + o.mode);
  o.threads = std::max(1, std::min(o.threads, 256));
  o.time_limit = std::max(0.0, o.time_limit);
  if (!o.seed_set) o.seed = static_cast<uint64_t>(Clock::now().time_since_epoch().count());
  return o;
}

struct Pool {  // distinct best cliques merged across workers
  int best = 0;
  std::vector<std::vector<int>> cliques;
  std::vector<int> hits;
  std::unordered_map<std::string, int> index;

  void Add(const std::vector<int>& clique, int count) {
    const int size = static_cast<int>(clique.size());
    if (size < best) return;
    if (size > best) {
      best = size;
      cliques.clear();
      hits.clear();
      index.clear();
    }
    std::string key(reinterpret_cast<const char*>(clique.data()), clique.size() * sizeof(int));
    const auto it = index.find(key);
    if (it != index.end()) {
      hits[it->second] += count;
      return;
    }
    index.emplace(std::move(key), static_cast<int>(cliques.size()));
    cliques.push_back(clique);
    hits.push_back(count);
  }
};

struct LsStats {
  uint64_t iterations = 0;
  uint64_t dropped = 0;
};

LsStats RunLocalSearch(const Graph& g, const Options& o, SharedState& sh, Pool& pool) {
  std::vector<std::unique_ptr<LocalSearch>> workers;
  for (int t = 0; t < o.threads; ++t)
    workers.emplace_back(new LocalSearch(g, sh, o.seed + 0x9e3779b97f4a7c15ULL * (t + 1),
                                         o.store_cap, o.penalty_delay, o.schedule));
  std::vector<std::thread> threads;
  for (int t = 1; t < o.threads; ++t) threads.emplace_back([&workers, t] { workers[t]->Run(); });
  workers[0]->Run();
  for (auto& th : threads) th.join();

  LsStats stats;
  for (const auto& w : workers) {
    stats.iterations += w->iterations();
    stats.dropped += w->dropped();
    for (size_t i = 0; i < w->cliques().size(); ++i) pool.Add(w->cliques()[i], w->hits()[i]);
  }
  return stats;
}

void PrintResults(const Graph& g, const Pool& pool, int report, int* verified_out) {
  std::vector<int> idx(pool.cliques.size());
  for (size_t i = 0; i < idx.size(); ++i) idx[i] = static_cast<int>(i);
  std::stable_sort(idx.begin(), idx.end(),
                   [&](int a, int b) { return pool.hits[a] < pool.hits[b]; });
  int printed = 0, verified = 0;
  for (int i : idx) {
    if (!IsMaximalClique(g, pool.cliques[i])) continue;  // never expected; guard the protocol
    ++verified;
    if (printed >= report) continue;
    std::printf("R %d", pool.hits[i]);
    for (int v : pool.cliques[i]) std::printf(" %d", v);
    std::printf("\n");
    ++printed;
  }
  *verified_out = verified;
}

}  // namespace

int main(int argc, char** argv) {
  const Options o = ParseArgs(argc, argv);
  SharedState sh;
  sh.start = Clock::now();
  sh.deadline = sh.start + std::chrono::duration_cast<Clock::duration>(
                               std::chrono::duration<double>(o.time_limit));
  sh.target = o.target;

  std::string input((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
  while (!input.empty() && (input.back() == '\n' || input.back() == '\r')) input.pop_back();

  Graph g;
  std::string err;
  if (!Decode(input, g, err)) Fail("decode failed: " + err);
  std::printf("N %d\n", g.n);
  std::fflush(stdout);
  const double decode_ms = sh.ElapsedMs();

  if (o.mode == "dump") {
    std::string line;
    for (int v = 0; v < g.n; ++v) {
      line.assign(g.n, '0');
      for (int u = 0; u < g.n; ++u)
        if (g.adj(v, u)) line[u] = '1';
      std::printf("A %s\n", line.c_str());
    }
    std::printf("S mode=dump edges=%lld\nE\n", static_cast<long long>(g.edges()));
    return 0;
  }

  Pool pool;
  LsStats ls_stats;
  int verified = 0;
  if (o.mode == "ls") {
    ls_stats = RunLocalSearch(g, o, sh, pool);
    PrintResults(g, pool, o.report, &verified);
    std::printf(
        "S mode=ls n=%d edges=%lld threads=%d best=%d distinct=%d iterations=%llu dropped=%llu "
        "decode_ms=%.1f time_ms=%.1f\n",
        g.n, static_cast<long long>(g.edges()), o.threads, pool.best, verified,
        static_cast<unsigned long long>(ls_stats.iterations),
        static_cast<unsigned long long>(ls_stats.dropped), decode_ms, sh.ElapsedMs());
  } else {  // exact
    const auto full_deadline = sh.deadline;
    const double warm = std::min(1.0, 0.1 * o.time_limit);
    sh.deadline = std::min(full_deadline, sh.start + std::chrono::duration_cast<Clock::duration>(
                                                         std::chrono::duration<double>(warm)));
    ls_stats = RunLocalSearch(g, o, sh, pool);
    sh.deadline = full_deadline;
    sh.stop = false;
    ExactSolver exact(g, sh, o.enumerate, o.store_cap);
    exact.Solve(pool.best);
    for (const auto& c : exact.cliques()) pool.Add(c, 1);
    PrintResults(g, pool, o.report, &verified);
    std::printf("S mode=exact n=%d best=%d proved=%d enumerate=%d distinct=%d nodes=%llu "
                "time_ms=%.1f\n",
                g.n, pool.best, exact.proved() ? 1 : 0, o.enumerate ? 1 : 0, verified,
                static_cast<unsigned long long>(exact.nodes()), sh.ElapsedMs());
  }
  std::printf("E\n");
  std::fflush(stdout);
  return 0;
}

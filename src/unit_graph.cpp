#include "Omega_h_graph.hpp"
#include "Omega_h_library.hpp"
#include "Omega_h_array_ops.hpp"

int main(int argc, char** argv) {
  using namespace Omega_h;
  auto lib = Library(&argc, &argv);

  // Test case: 4 nodes with various edges, filter out node 1
  {
    // Create a graph with 4 nodes:
    // Node 0: edges to [1, 2]     (2 edges)
    // Node 1: edges to [0, 3]     (2 edges)
    // Node 2: edges to [0, 3]     (2 edges)
    // Node 3: edges to [1, 2]     (2 edges)

    // Offset array: [0, 2, 4, 6, 8]
    LOs offsets({0, 2, 4, 6, 8});

    // Values array: [1, 2,  0, 3,  0, 3,  1, 2]
    LOs values({1, 2, 0, 3, 0, 3, 1, 2});

    Graph g(offsets, values);

    // Verify initial graph structure
    OMEGA_H_CHECK(g.nnodes() == 4);
    OMEGA_H_CHECK(g.nedges() == 8);

    // Filter: keep nodes 0, 2, and 3; remove node 1
    Read<I8> keep_node({1, 0, 1, 1});

    Graph filtered = filter_graph_nodes(g, keep_node);

    // Expected result: 3 nodes
    // New node 0 (old 0): edges to [1] (old node 2 only, since old node 1 is removed)
    // New node 1 (old 2): edges to [0, 2] (old nodes 0 and 3)
    // New node 2 (old 3): edges to [1] (old node 2 only, since old node 1 is removed)

    OMEGA_H_CHECK(filtered.nnodes() == 3);
    OMEGA_H_CHECK(filtered.nedges() == 4);

    // Check offset array: [0, 1, 3, 4]
    LOs expected_offsets({0, 1, 3, 4});
    OMEGA_H_CHECK(filtered.a2ab == expected_offsets);

    // Check values array: [1, 0, 2, 1]
    // New node 0 -> new node 1 (old 2)
    // New node 1 -> new nodes 0, 2 (old 0, 3)
    // New node 2 -> new node 1 (old 2)
    LOs expected_values({1, 0, 2, 1});
    OMEGA_H_CHECK(filtered.ab2b == expected_values);
  }

  // Test case: 3 nodes, keep all
  {
    // Node 0: edges to [1]
    // Node 1: edges to [0, 2]
    // Node 2: edges to [1, 0]

    LOs offsets({0, 1, 3, 5});
    LOs values({1, 0, 2, 1, 0});

    Graph g(offsets, values);
    OMEGA_H_CHECK(g.nnodes() == 3);
    OMEGA_H_CHECK(g.nedges() == 5);

    // Keep all nodes
    Read<I8> keep_node({1, 1, 1});

    Graph filtered = filter_graph_nodes(g, keep_node);

    // Should be identical to original
    OMEGA_H_CHECK(filtered.nnodes() == 3);
    OMEGA_H_CHECK(filtered.nedges() == 5);
    OMEGA_H_CHECK(filtered.a2ab == offsets);
    OMEGA_H_CHECK(filtered.ab2b == values);
  }

  // Test case: 4 nodes, keep only 2 non-adjacent nodes
  {
    // Node 0: edges to [1]
    // Node 1: edges to [0, 2]
    // Node 2: edges to [1, 3]
    // Node 3: edges to [2]

    LOs offsets({0, 1, 3, 5, 6});
    LOs values({1, 0, 2, 1, 3, 2});

    Graph g(offsets, values);

    // Keep only nodes 0 and 3 (not adjacent)
    Read<I8> keep_node({1, 0, 0, 1});

    Graph filtered = filter_graph_nodes(g, keep_node);

    // Result: 2 nodes with no edges (all edges connected to removed nodes)
    OMEGA_H_CHECK(filtered.nnodes() == 2);
    OMEGA_H_CHECK(filtered.nedges() == 0);

    LOs expected_offsets({0, 0, 0});
    OMEGA_H_CHECK(filtered.a2ab == expected_offsets);
  }

  return 0;
}

from __future__ import annotations

import networkx as nx


def main() -> None:
    graph = nx.Graph()
    graph.add_node("Alice", kind="person")
    graph.add_node("Project Atlas", kind="project")
    graph.add_node("Bob", kind="person")
    graph.add_edge("Alice", "Project Atlas", relation="worked_on")
    graph.add_edge("Bob", "Project Atlas", relation="managed")

    print("Nodes:")
    for node, attrs in graph.nodes(data=True):
        print(f"  - {node}: {attrs}")

    print("\n1-hop neighbors from Alice:", list(graph.neighbors("Alice")))
    print("Alice -> Project Atlas relation:", graph["Alice"]["Project Atlas"]["relation"])


if __name__ == "__main__":
    main()
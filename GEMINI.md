# SignBridge Agent Rules

This workspace is configured with **Ponytail** and **Graphify**.

- **Graphify**: Consult the knowledge graph in `graphify-out/` for architectural understanding. Use `graphify query "<topic>"` to retrieve targeted subgraphs. Update the graph with `graphify update .` upon substantial code changes.
- **Ponytail**: Always follow the lazy senior developer ladder: YAGNI -> reuse existing code -> stdlib -> native platform features -> installed dependencies -> one-liner -> minimal code. Avoid unrequested abstractions and bloated dependencies.

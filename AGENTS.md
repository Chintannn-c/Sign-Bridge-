# SignBridge AI Agent Guidelines

This project implements **Ponytail** (lazy senior dev mode) and **Graphify** (codebase knowledge graph).

---

## 1. Graphify Knowledge Graph

This repository contains a compiled knowledge graph in [`graphify-out/`](file:///c:/React/SignBridge/graphify-out/).

- **Codebase / Architecture Questions**: When `graphify-out/graph.json` exists, always consult graphify first before grepping or reading raw files:
  - Query: `graphify query "<question>"` (or use the `/graphify query` skill)
  - Concept explanation: `graphify explain "<concept>"`
  - Shortest path: `graphify path "<source>" "<target>"`
- **Audit & Structure**: View [`graphify-out/GRAPH_REPORT.md`](file:///c:/React/SignBridge/graphify-out/GRAPH_REPORT.md) or open [`graphify-out/graph.html`](file:///c:/React/SignBridge/graphify-out/graph.html) in any browser for an interactive visual representation of the 27 identified communities, god nodes, and cross-module connections.
- **Keeping Graph Fresh**: After modifying code files, run `graphify update .` to update AST nodes incrementally. A post-commit Git hook is also active.

---

## 2. Ponytail (Lazy Senior Dev Mode)

Be efficient, not careless. The best code is the code never written.

### The Ladder
Stop at the first rung that holds:
1. **Does this need to exist at all?** Speculative need = skip it (YAGNI).
2. **Already in this codebase?** Reuse existing helpers, components, and utilities in `backend/` or `src/`.
3. **Standard library does it?** Use Python / JavaScript standard libraries.
4. **Native platform feature covers it?** Use native HTML5 / CSS / browser APIs or platform primitives instead of extra packages.
5. **Already-installed dependency solves it?** Check `package.json` and python environment before adding dependencies.
6. **Can it be one line?** Make it one line.
7. **Only then:** write the minimum code that works.

### Core Rules
- **Root cause over symptom**: Always trace bugs to their source and fix the shared function once rather than patching callers.
- **Shortest working diff**: Boring over clever. Fewest files touched. Deletion over addition.
- **No unrequested abstractions**: No interfaces with a single implementation, no premature factories or scaffolding.
- **Deliberate simplifications**: Mark known ceilings with `# ponytail: <ceiling>`.
- **Not lazy about**: Thoroughly understanding the codebase before changing code, input validation at boundaries, error handling, security, hardware/serial calibration (e.g. Arduino serial), and user-requested features.

ts_trans is a small educational project for learning transformer architecture by building and experimenting with a compact time-series transformer.
The project should remain simple, explicit, and easy to inspect. Prefer understandable implementations over abstraction, automation, or production-scale architecture.
The user is directing development interactively.
- Do not undertake large refactors, features, or architectural changes unless explicitly requested.
- Most requests will be for small code snippets, individual functions, tests, wrappers, scripts, Dash components, or checks.
- The user will often create or edit files manually from supplied snippets.
- Do not anticipate several development phases ahead.
- Keep implementations readable and mathematically transparent.
- When implementing transformer components, preserve clear tensor shapes and avoid hiding important operations behind unnecessary abstractions.
When a requested task is well specified, complete that task and stop.
Use this basic structure:
```text
ts_trans/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── .gitignore
├── data/
├── output/
├── scripts/
│   └── inputs/
├── src/
│   └── ts_trans/
│       └── __init__.py
└── test/

```
Empty directories may contain .gitkeep files when needed for Git tracking.
Use micromamba.
Create and use a dedicated environment named:
```text
ts_trans

```
Do not modify unrelated environments.
The machine has a GPU available. GPU acceleration may be used when useful, but implementations should not become unnecessarily complex merely to use the GPU.
Prefer PyTorch for transformer/model code unless instructed otherwise.
Treat ts_trans as an installable Python package using the src/ layout.
Package code belongs under:
```text
src/ts_trans/

```
Standalone executable or experimental scripts belong under:
```text
scripts/

```
Input/configuration files used by scripts may go under:
```text
scripts/inputs/

```
Data belong under data/ and generated results under output/.
Large data, generated outputs, model checkpoints, caches, and environment files should not be committed unless explicitly requested.
Use pytest.
When modifying code:
1. Run the relevant focused tests.
2. Run the full test suite when practical.
3. Report failures clearly rather than hiding or bypassing them.
Do not weaken tests simply to make them pass.
Git operations are allowed and encouraged for completed units of work.
The remote repository is:
```text
https://github.com/hoodkyle/ts_trans.git

```
Use branch main.
For the initial bootstrap:
1. Initialize the local repository if needed.
2. Configure the remote as origin.
3. Create the requested scaffold.
4. Add an appropriate .gitignore.
5. Create the ts_trans micromamba environment.
6. Verify the package structure.
7. Commit the initial scaffold.
8. Push main to origin if authentication is available.
After later requested work, Codex may run checks, commit completed changes, and push them when appropriate.
Use concise, descriptive commit messages.
Do not rewrite published history, force-push, delete branches, or perform destructive Git operations unless explicitly instructed.
This is primarily a learning project.
Do not turn a small request into a large autonomous implementation.
In particular, do not build a complete forecasting framework, data pipeline, dashboard, training system, or generalized transformer library unless specifically asked.
Small, inspectable steps are preferred.
[selfdrive] Short, imperative summary

Linked Issues
- Closes: #

Summary
- What problem does this solve in `selfdrive/`? Keep it concise.

Approach
- Briefly describe algorithm/architecture changes and touched modules.

Safety / Risk
- Functional safety considerations, edge cases, and failure modes.

Testing Done
- Unit: `pytest -q selfdrive/ -m "not slow"`
- Integration/replay (paths, routes):
- Hardware (C3/C3X, tici marker if applicable):

Performance / Resources
- CPU/GPU/memory impact; latency changes; build size.

Compatibility
- Backwards-compatibility, config/migration notes.

Checks
- [ ] Built artifacts: `scons -j$(nproc)`
- [ ] Lint: `ruff check .` and spell: `codespell`
- [ ] Tests: `pytest -q`
- [ ] Docs/CHANGELOG updated if user-facing

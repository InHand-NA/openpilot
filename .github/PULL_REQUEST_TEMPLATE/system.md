[system] Short, imperative summary

Linked Issues
- Closes: #

Summary
- What subsystem in `system/` is affected (e.g., managerd, camerad, loggerd)?

Approach
- Key changes to processes, IPC, startup/shutdown, permissions.

Risk / Ops Notes
- Boot/upgrade implications, logging, persistence, watchdogs.

Testing Done
- Unit: `pytest -q system/ -m "not slow"`
- Process bring-up, logs attached where relevant
- Hardware-on-target (C3/C3X) if needed

Performance / Resources
- CPU, I/O, power, and startup time changes.

Checks
- [ ] `scons -j$(nproc)`
- [ ] `ruff check .` / `codespell`
- [ ] `pytest -q`

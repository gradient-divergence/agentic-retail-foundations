# Upstream provenance

This directory vendors the companion repository for the book:

- Upstream: https://github.com/gradient-divergence/agentic-retail-foundations
- Vendored commit: b188034a75338047ab8fb9f8cdb4f2a6415d8ad7
- Vendored on: 2025-12-18T08:28:15Z

## Why this exists

We maintain the book manuscript and the runnable companion code together (monorepo-style)
so code snippets, APIs, and runnable demos do not drift from the book.

## Sync policy

- Make changes here (inside this monorepo) as the source of truth during V2 development.
- When ready to publish code updates, sync changes back to the upstream repo.

## Recommended sync-back workflow (one-time setup)

Option A (recommended): `git subtree` (monorepo → upstream-friendly)

This keeps the book repo as the working monorepo while still publishing the code to the upstream repository.

1. Add the upstream remote (once):

   ```bash
   git remote add companion-upstream https://github.com/gradient-divergence/agentic-retail-foundations.git
   ```

2. Split the companion directory into its own branch:

   ```bash
   git subtree split --prefix=companion/agentic-retail-foundations -b companion-sync
   ```

3. Push the split branch to a fork or upstream (depending on permissions) and open a PR:

   ```bash
   # Push to upstream (if you have access):
   git push companion-upstream companion-sync:main

   # Or push to your fork and open a PR:
   # git push <your-fork-remote> companion-sync:main
   ```

Option B (simple, manual):
- Clone the upstream repo separately and copy changes across, then open a PR.

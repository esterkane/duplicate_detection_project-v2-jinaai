"""Read-only MCP server exposing the duplicate-detection core as agent tools.

Thin adapters over existing functions in :mod:`src` (hybrid search, the
``detect_duplicates`` similarity helper, and a single-article fetch). The MCP
layer holds no business logic of its own.
"""

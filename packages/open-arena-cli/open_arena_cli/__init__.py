"""open-arena-cli: Standalone CLI for Open Arena.

Thin commands (env/verifier/leaderboard/run/discover/request) import only
``open_arena_core`` and work without the heavy evaluation engine.

Heavy commands (sweep, serve, --local run mode) lazy-import the engine from
the ``open-arena`` package and raise a friendly error if it is not installed.
"""

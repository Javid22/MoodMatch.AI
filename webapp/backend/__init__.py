"""Game backend package for MoodMatch.Ai.

This package is purely a thin web layer around the existing ML pipeline in
`training/` and `src/model/`. It does not re-implement or modify the model —
it only loads the trained checkpoint once and exposes it over HTTP so the
game UI can call it without spawning a new Python process per request.
"""

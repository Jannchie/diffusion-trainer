"""Process-wide settings that must be in place before torch allocates on CUDA.

Deliberately free of heavy imports: every entry point (run_train, run_prepare,
the scripts/ tools, the tests) reaches this module just by importing anything
from the package, and the allocator only reads its config at the first
allocation — so an import-time setdefault is early enough.
"""

import os

# Bucketed training and VAE encoding both walk many resolutions, which fragments
# the caching allocator; expandable segments grow in place instead of leaving
# unusable holes. setdefault so an explicit env var from the caller still wins.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

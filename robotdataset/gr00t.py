"""GR00T export of dataset statistics (kept for compatibility).

The implementation lives in :mod:`robotdataset.adapters.gr00t`, next to the adapter that uses it.
"""

from robotdataset.adapters.gr00t import GR00T_STAT_KEYS, check_against, to_gr00t_statistics

__all__ = ["GR00T_STAT_KEYS", "check_against", "to_gr00t_statistics"]

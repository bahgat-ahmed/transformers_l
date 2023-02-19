"""
Utilities for working with package versions
"""

import operator
import re
import sys
from typing import Optional

from packaging import version

# The package importlib_metadata is in a different place, depending on the python version.
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata


ops = {
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    ">": operator.gt,
}


def _compare_versions(op, got_ver, want_ver, requirement, pkg, hint):
    if got_ver is None or want_ver is None:
        raise ValueError(
            f"Unable to compare versions for {requirement}: need={want_ver} found={got_ver}. This"
            f"is unusual. Consider reinstalling {pkg}"
        )
    if not ops[op](version.parse(got_ver), version.parse(want_ver)):
        raise ImportError(
            f"{requirement} is required for a normal functioning of this module, but found {pkg}=={got_ver}.{hint}"
        )

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class CleaningStats:
    total: int = 0
    passed: int = 0
    removed_garbled: int = 0
    removed_quality: int = 0
    removed_short: int = 0
    sensitive_counts: Dict[str, int] = field(default_factory=dict)

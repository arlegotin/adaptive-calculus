"""Scoped regression for attempt-helper handling of pytest's IntEnum status."""
from enum import IntEnum
class ModuleExit(IntEnum):
    OK=0
print('IntEnum successful module exit regression')
raise SystemExit(ModuleExit.OK)

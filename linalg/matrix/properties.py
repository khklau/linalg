from enum import Enum, auto


class Singularity(Enum):
    NON_SINGULAR = auto()
    SINGULAR = auto()

    def __str__(self) -> str:
        return self.name

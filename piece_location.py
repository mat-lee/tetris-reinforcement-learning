from dataclasses import dataclass, replace

@dataclass
class PieceLocation:
    x: int
    y: int
    rotation: int = 0
    rotation_just_occurred: bool = False
    rotation_just_occurred_and_used_last_tspin_kick: bool = False

    def copy(self) -> 'PieceLocation':
        return replace(self)
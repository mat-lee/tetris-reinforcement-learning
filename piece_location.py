from dataclasses import dataclass, replace

@dataclass(slots=True)
class PieceLocation:
    x: int
    y: int
    rotation: int = 0
    rotation_just_occurred: bool = False
    rotation_just_occurred_and_used_last_tspin_kick: bool = False

    def copy(self) -> 'PieceLocation':
        return PieceLocation(self.x, self.y, self.rotation,
                             self.rotation_just_occurred,
                             self.rotation_just_occurred_and_used_last_tspin_kick)
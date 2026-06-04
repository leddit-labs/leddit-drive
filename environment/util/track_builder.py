TRACK_DEFINITION = {
    "walls": [
        ((100, 100), (700, 100)),
        ((700, 100), (700, 500)),
        ((700, 500), (100, 500)),
        ((100, 500), (100, 100)),

        ((250, 250), (500, 250)),
        ((500, 350), (250, 350)),
        ((500, 250), (500, 350)),
        ((250, 250), (250, 350)),
    ],

    "checkpoint_mode": "midpoint" # to where the checkpoints are located on the given line
}

class TrackBuilder:
    def __init__(self, definition):
        self.definition = definition

    def build(self):
        walls = self.definition["walls"]

        segments = []
        checkpoints = []

        for i, (a, b) in enumerate(walls):
            segment = {
                "a": a,
                "b": b,
                "mid": ((a[0]+b[0])/2, (a[1]+b[1])/2),
                "id": i
            }

            segments.append(segment)

            # checkpoint logic
            if self.definition["checkpoint_mode"] == "midpoint":
                checkpoints.append(segment["mid"])

        return {
            "segments": segments,
            "checkpoints": checkpoints
        }
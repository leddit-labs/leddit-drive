TRACK_DEFINITION = {
    "outer": [
        (54, 77),
        (145, 377),
        (63, 549),
        (185, 698),
        (617, 565),
        (930, 661),
        (1096, 623),
        (1126, 505),
        (1002, 383),
        (987, 254),
        (1095, 140),
        (1034, 17),
        (768, 51),
        (614, 169),
        (367, 169),
        (207, 27),
        (54, 77),
    ],
    "inner": [
        (197, 161),
        (276, 374),
        (193, 519),
        (242, 578),
        (591, 444),
        (900, 537),
        (945, 505),
        (772, 332),
        (915, 148),
        (877, 133),
        (592, 305),
        (382, 307),
        (217, 154),
        (197, 161),
    ],
}


class TrackBuilder:
    def __init__(self, definition):
        self.definition = definition
        self._validate_walls(TRACK_DEFINITION["outer"], "outer")
        self._validate_walls(TRACK_DEFINITION["inner"], "inner")

    def build(self):
        outer = self.definition["outer"]
        inner = self.definition["inner"]

        def build_segments(points):
            segs = []
            checkpoints = []
            n = len(points)

            for i in range(n):
                a = points[i]
                b = points[(i + 1) % n]

                mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)

                segs.append((a, b))
                checkpoints.append(mid)

            return segs, checkpoints

        outer_segments, outer_checkpoints = build_segments(outer)
        inner_segments, inner_checkpoints = build_segments(inner) # not currently used

        return {
            "outer": outer,
            "inner": inner,
            "segments": outer_segments + inner_segments,
            "outer_checkpoints": outer_checkpoints,
        }

    def _validate_walls(self, points, name):
        if len(points) <= 0:
            return
        if points[0] != points[-1]:
            print(f"ERROR: '{name}' is not closed")

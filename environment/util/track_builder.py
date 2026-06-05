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
    "checkpoints": [
        ((205, 165), (87, 203)),
        ((249, 275), (120, 313)),
        ((280, 371), (137, 377)),
        ((227, 464), (114, 425)),
        ((199, 516), (57, 551)),
        ((244, 571), (184, 704)),
        ((281, 554), (327, 665)),
        ((425, 498), (470, 617)),
        ((650, 454), (644, 593)),
        ((871, 520), (868, 650)),
        ((923, 509), (1051, 643)),
        ((1057, 418), (900, 477)),
        ((1002, 318), (831, 244)),
        ((898, 155), (1099, 122)),
        ((849, 29), (877, 148)),
        ((676, 109), (738, 227)),
        ((517, 159), (509, 317)),
        ((384, 159), (330, 263)),
        ((288, 86), (236, 179)),
        ((167, 34), (211, 165)),
    ],
}


class TrackBuilder:
    def __init__(self, definition):
        self.definition = definition

    def build(self):
        outer = self.definition["outer"]
        inner = self.definition["inner"]
        checkpoints = self.definition.get("checkpoints", [])

        def build_segments(points):
            segs = []
            n = len(points)

            for i in range(n):
                a = points[i]
                b = points[(i + 1) % n]
                segs.append((a, b))

            return segs

        return {
            "outer": outer,
            "inner": inner,
            "segments": build_segments(outer) + build_segments(inner),
            "checkpoints": checkpoints,
        }

    def _validate_walls(self, points, name):
        if len(points) <= 0:
            return
        if points[0] != points[-1]:
            print(f"ERROR: '{name}' is not closed")

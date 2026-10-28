SOURCES_LABELS = {
    "Amnesty POI": 0,
    "ASMSpotter": 1,
    "Landcover": 2,
    "UNHCR": 3,
}

# Settlement: 1700, Forest: 733, Agriculture: 699, Other: 489 (incl. 71 missing),
# Grassland: 205, Water: 58, Wetland: 41.
IPCC_LABELS = {
    "Settlement": 0,
    "Forest": 1,
    "Agriculture": 2,
    "Other": 3,
    "Grassland": 4,
    "Water": 5,
    "Wetland": 6,
}

SIMPLIFIED_IPCC_LABELS = {
    "Settlement": 0,
    "Forest": 1,
    "Agriculture": 2,
    "Other": 3,
    "Grassland": 3,
    "Water": 3,
    "Wetland": 3,
}


class LabelMappings:
    """Map dataset label names to integer class indices."""

    def __init__(self, label_type: str):
        if label_type == "sources":
            self.label = SOURCES_LABELS
        elif label_type == "ipcc":
            self.label = IPCC_LABELS
        elif label_type == "simplified_ipcc":
            self.label = SIMPLIFIED_IPCC_LABELS
        else:
            raise ValueError(f"Unknown label type: {label_type}")

    def get_label(self, name: str) -> int:
        try:
            return self.label[name]
        except KeyError:
            return self.label["Other"]

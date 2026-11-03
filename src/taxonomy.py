"""Explicit class definitions for the COCO-to-KITTI comparison."""

SHARED_CATEGORIES = [{"id": 1, "name": "vehicle"}, {"id": 2, "name": "person"}]
COCO_TO_SHARED = {
    "car": "vehicle", "truck": "vehicle", "bus": "vehicle", "train": "vehicle",
    "person": "person",
}
KITTI_TO_SHARED = {
    "Car": "vehicle", "Van": "vehicle", "Truck": "vehicle", "Tram": "vehicle",
    "Pedestrian": "person", "Person_sitting": "person",
}
KITTI_IGNORED = ("Cyclist", "Misc", "DontCare")


def shared_name(name):
    # Already-converted annotations also pass through this mapping unchanged.
    if name in {"vehicle", "person"}:
        return name
    return COCO_TO_SHARED.get(name)


def remap_coco(dataset):
    """Copy mapped annotations; retain all images, including negative images."""
    name_to_id = {category["name"]: category["id"] for category in SHARED_CATEGORIES}
    id_map = {}
    for category in dataset["categories"]:
        name = shared_name(category["name"])
        if name is not None:
            id_map[category["id"]] = name_to_id[name]
    annotations = [
        {**annotation, "category_id": id_map[annotation["category_id"]]}
        for annotation in dataset["annotations"]
        if annotation["category_id"] in id_map
    ]
    return {**dataset, "categories": SHARED_CATEGORIES, "annotations": annotations}

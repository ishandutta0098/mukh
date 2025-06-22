import csv


def json_to_csv(json_data: list[dict], csv_path: str) -> None:
    """
    Convert list of face detections in JSON to CSV.

    Args:
        json_data: List of face detections in JSON format.
        csv_path: Path to save the CSV file.

    Example:
        json_data = [
            {
                "image_name": "img1.jpg",
                "x1": 71,
                "y1": 161,
                "x2": 440,
                "y2": 637,
                "confidence": 1.0
            }
        ]
        json_to_csv(json_data, "output.csv")
    """
    if not json_data:
        return

    keys = json_data[0].keys()
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(json_data)

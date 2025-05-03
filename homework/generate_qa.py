import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0  # Default values if parsing fails


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    # Read the image using PIL
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    img_width, img_height = pil_image.size

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Extract frame ID and view index from image filename
    _, view_index = extract_frame_info(image_path)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Draw each detection
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Get color for this object type
        if track_id == 0:
            color = (255, 0, 0)
        else:
            color = COLORS.get(class_id, (255, 255, 255))

        # Draw bounding box using PIL
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

    # Convert PIL image to numpy array for matplotlib
    return np.array(pil_image)


def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list:
    """
    Extract kart objects from the info.json file, including their center points and identify the center kart.
    Filters out karts that are out of sight (outside the image boundaries).

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 100)
        img_height: Height of the image (default: 150)

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center
        - is_center_kart: Boolean indicating if this is the kart closest to image center
    """
    with open(info_path) as f:
        info = json.load(f)

    karts = []
    if view_index < len(info["detections"]):
        detections = info["detections"][view_index]
        process_karts(detections,
                      img_width / 2,
                      img_height / 2,
                      img_height,
                      img_width,
                      info["karts"],
                      karts,
                      min_box_size,
                      img_width / ORIGINAL_WIDTH,
                      img_height / ORIGINAL_HEIGHT)
        process_karts_center(karts)

    return karts


def process_karts_center(karts):
    if karts:
        closest_kart = min(karts, key=lambda k: k["distance_to_center"])
        for kart in karts:
            if kart is closest_kart:
                kart["is_center_kart"] = True


def process_karts(detections, image_center_x, image_center_y, img_height, img_width, kart_names, karts, min_box_size,
                  scale_width, scale_height):
    for detection in detections:
        class_id, track_id, x1, y1, x2, y2 = detection

        class_id = int(class_id)
        track_id = int(track_id)
        scaled_width_1 = int(x1 * scale_width)
        scaled_height_1 = int(y1 * scale_height)
        scaled_width_2 = int(x2 * scale_width)
        scaled_height_2 = int(y2 * scale_height)

        if class_id != 1 or (scaled_height_2 - scaled_height_1) < min_box_size or (scaled_width_2 - scaled_width_1) < min_box_size or scaled_width_2 < 0 or scaled_width_1 > img_width or scaled_height_2 < 0 or scaled_height_1 > img_height:
            continue

        center_width = (scaled_width_1 + scaled_width_2) / 2
        center_height = (scaled_height_1 + scaled_height_2) / 2

        karts.append({
            "instance_id": track_id,
            "kart_name": kart_names[track_id] if track_id < len(kart_names) else f"Unknown_{track_id}",
            "center": (center_width, center_height),
            "distance_to_center": ((center_width - image_center_x) ** 2 + (center_height - image_center_y) ** 2) ** 0.5,
            "is_center_kart": False
        })


def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """
    with open(info_path) as f:
        info = json.load(f)
    return info.get("track", "unknown_track")


def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 100)
        img_height: Height of the image (default: 150)

    Returns:
        List of dictionaries, each containing a question and answer
    """
    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)

    qa_pairs = []

    c_kart = get_center_kart(karts)
    c_name = c_kart["kart_name"]
    c_center_x, c_center_y = c_kart["center"]

    qa_pairs.append({"question": "What kart is the ego car?", "answer": c_name})
    qa_pairs.append({"question": "How many karts are there in the scenario?", "answer": str(len(karts))})
    qa_pairs.append({"question": "What track is this?", "answer": track_name})

    process_karts_and_qa_pairs(c_center_x, c_center_y, karts, qa_pairs)
    link_image_file(info_path, qa_pairs, view_index)
    return qa_pairs


def process_karts_and_qa_pairs(c_center_x, c_center_y, karts, qa_pairs):
    karts_left = []
    karts_right = []
    karts_front = []
    karts_back = []
    for kart in karts:
        if not kart["is_center_kart"]:
            process_non_center_kart(c_center_x,
                                    c_center_y,
                                    kart,
                                    karts_back,
                                    karts_front,
                                    karts_left,
                                    karts_right,
                                    qa_pairs)
    qa_pairs.append({"question": "How many karts are to the left of the ego car?", "answer": str(len(karts_left))})
    qa_pairs.append({"question": "How many karts are to the right of the ego car?", "answer": str(len(karts_right))})
    qa_pairs.append({"question": "How many karts are in front of the ego car?", "answer": str(len(karts_front))})
    qa_pairs.append({"question": "How many karts are behind the ego car?", "answer": str(len(karts_back))})


def link_image_file(info_path, qa_pairs, view_index):
    info_file_path = Path(info_path)
    base_name = info_file_path.stem.replace("_info", "")
    image_file = f"{info_file_path.parent.name}/{base_name}_{view_index:02d}_im.jpg"
    for qa_pair in qa_pairs:
        qa_pair["image_file"] = image_file


def process_non_center_kart(c_center_x, c_center_y, kart, karts_back, karts_front, karts_left, karts_right, qa_pairs):
    kart_center_x, kart_center_y = kart["center"]
    is_left = kart_center_x <= c_center_x
    horizontal_pos = "left" if is_left else "right"
    is_front = kart_center_y <= c_center_y
    vertical_pos = "front" if is_front else "back"
    if is_left:
        karts_left.append(kart)
    else:
        karts_right.append(kart)
    if is_front:
        karts_front.append(kart)
    else:
        karts_back.append(kart)
    kart_name = kart["kart_name"]
    qa_pairs.append({"question": f"Is {kart_name} to the left or right of the ego car?", "answer": horizontal_pos})
    qa_pairs.append({"question": f"Is {kart_name} in front of or behind the ego car?", "answer": "in front of" if is_front else "behind"})
    qa_pairs.append({"question": f"Where is {kart_name} relative to the ego car?", "answer": f"{vertical_pos} and {horizontal_pos}"})


def get_center_kart(karts):
    return next((kart for kart in karts if kart["is_center_kart"]), None)


def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    # Find corresponding image file
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    # Visualize detections
    annotated_image = draw_detections(str(image_file), info_file)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(info_file, view_index)

    # Print QA pairs
    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)


def generate_all(output_file: str = "data/train/balanced_qa_pairs.json"):
    generated = []
    populate(generated, list(Path("data/train").glob("*_info.json")))
    write_to_file(generated, output_file)
    return generated


def populate(generated, info):
    for info_file in info:
        with open(info_file) as f:
            info = json.load(f)

        num_views = len(info.get("detections", []))

        for view_index in range(num_views):
            qa_pairs = generate_qa_pairs(str(info_file), view_index)
            generated.extend(qa_pairs)


def write_to_file(all_qa_pairs, output_file):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_qa_pairs, f, indent=2)


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_qa.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({
        "check": check_qa_pairs,
        "generate_all": generate_all
    })


if __name__ == "__main__":
    main()

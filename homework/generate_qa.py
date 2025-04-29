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
    #read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    #get the kart names
    kart_names = info["karts"]

    #get the detections for the specified view
    if view_index < len(info["detections"]):
        detections = info["detections"][view_index]
    else:
        return []  #No detections for this view

    #calculate scaling factors to adjust the coordinates
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    #calcualte the center of the image
    image_center_x = img_width / 2
    image_center_y = img_height / 2

    #a list to store kart objects
    karts = []

    # Process each detection
    for detection in detections:
        #parse detection data
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        #only consider kart objects (class_id is 1)
        if class_id != 1:
            continue

        #scale the coordinates
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        #skip if the bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        # Skip if kart is out of sight
        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        #Calculate center of the kart
        center_x = (x1_scaled + x2_scaled) / 2
        center_y = (y1_scaled + y2_scaled) / 2

        #Get kart name (if track_id is valid)
        kart_name = kart_names[track_id] if track_id < len(kart_names) else f"Unknown_{track_id}"

        #Calculate distance to image center
        distance_to_center = ((center_x - image_center_x) ** 2 + (center_y - image_center_y) ** 2) ** 0.5

        #Add kart to list
        kart = {
            "instance_id": track_id,
            "kart_name": kart_name,
            "center": (center_x, center_y),
            "distance_to_center": distance_to_center,
            "is_center_kart": False  # Will be set later
        }
        karts.append(kart)

    #find the kart closest to the image center
    if karts:
        closest_kart = min(karts, key=lambda k: k["distance_to_center"])
        #mark the closest kart as the center kart
        for kart in karts:
            if kart is closest_kart:
                kart["is_center_kart"] = True

    return karts

def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """
    #read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    #extract track name from the info
    track_name = info.get("track", "unknown_track")

    return track_name


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
    # Get kart objects
    karts = extract_kart_objects(info_path, view_index, img_width, img_height)

    # Get track info
    track_name = extract_track_info(info_path)

    # Prepare empty list for QA pairs
    qa_pairs = []

    #skip if there are no karts found
    if not karts:
        return qa_pairs

    #find the ego (center) kart
    ego_kart = next((kart for kart in karts if kart["is_center_kart"]), None)
    if not ego_kart:
        return qa_pairs

    # Get ego kart info
    ego_name = ego_kart["kart_name"]
    ego_center_x, ego_center_y = ego_kart["center"]

    # 1. Ego car question
    # What kart is the ego car?
    qa_pairs.append({
        "question": "What kart is the ego car?",
        "answer": ego_name
    })

    # 2. Total karts question
    # How many karts are there in the scenario?
    qa_pairs.append({
        "question": "How many karts are there in the scenario?",
        "answer": str(len(karts))
    })

    # 3. Track information questions
    # What track is this?
    qa_pairs.append({
        "question": "What track is this?",
        "answer": track_name
    })

    # Create lists for counting karts in different positions
    karts_left = []
    karts_right = []
    karts_front = []
    karts_back = []

    # 4. Relative position questions for each kart
    # Is {kart_name} to the left or right of the ego car?
    # Is {kart_name} in front of or behind the ego car?
    for kart in karts:
        # Skip the ego kart itself
        if kart["is_center_kart"]:
            continue

        kart_name = kart["kart_name"]
        kart_center_x, kart_center_y = kart["center"]

        # Determine horizontal position (left or right)
        is_left = kart_center_x <= ego_center_x
        horizontal_pos = "left" if is_left else "right"

        # Determine vertical position (front or back)
        # In images, lower y values are at the top, so front is above (lower y)
        is_front = kart_center_y <= ego_center_y
        vertical_pos = "front" if is_front else "back"

        # Add to position lists for counting
        if is_left:
            karts_left.append(kart)
        else:
            karts_right.append(kart)

        if is_front:
            karts_front.append(kart)
        else:
            karts_back.append(kart)

        # Individual position questions
        qa_pairs.append({
            "question": f"Is {kart_name} to the left or right of the ego car?",
            "answer": horizontal_pos
        })

        qa_pairs.append({
            "question": f"Is {kart_name} in front of or behind the ego car?",
            "answer": "in front of" if is_front else "behind"
        })

        # Combined position question (as mentioned in the student guide)
        qa_pairs.append({
            "question": f"Where is {kart_name} relative to the ego car?",
            "answer": f"{vertical_pos} and {horizontal_pos}"
        })

    # 5. Counting questions
    # How many karts are to the left of the ego car?
    # How many karts are to the right of the ego car?
    # How many karts are in front of the ego car?
    # How many karts are behind the ego car?
    qa_pairs.append({
        "question": "How many karts are to the left of the ego car?",
        "answer": str(len(karts_left))
    })

    qa_pairs.append({
        "question": "How many karts are to the right of the ego car?",
        "answer": str(len(karts_right))
    })

    qa_pairs.append({
        "question": "How many karts are in front of the ego car?",
        "answer": str(len(karts_front))
    })

    qa_pairs.append({
        "question": "How many karts are behind the ego car?",
        "answer": str(len(karts_back))
    })

    # Add image file path for reference
    info_file_path = Path(info_path)
    base_name = info_file_path.stem.replace("_info", "")
    image_file = f"{info_file_path.parent.name}/{base_name}_{view_index:02d}_im.jpg"

    # Add image file to each QA pair
    for qa_pair in qa_pairs:
        qa_pair["image_file"] = image_file

    return qa_pairs


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


def generate_all(output_file: str = None):
    """
    Generate QA pairs for all images in the training set.

    Args:
        output_file: Path to save the combined QA pairs (default: data/train/balanced_qa_pairs.json)
    """
    # Default output file
    if output_file is None:
        output_file = "data/train/balanced_qa_pairs.json"

    #find all the info files in the training set
    data_dir = Path("data/train")
    info_files = list(data_dir.glob("*_info.json"))

    print(f"Found {len(info_files)} info files in the training set")

    #generate QA pairs for each info file and view
    all_qa_pairs = []

    for info_file in info_files:
        print(f"Processing {info_file}")

        #read the info file to find out how many views it has
        with open(info_file) as f:
            info = json.load(f)

        num_views = len(info.get("detections", []))

        #process each view
        for view_index in range(num_views):
            qa_pairs = generate_qa_pairs(str(info_file), view_index)
            all_qa_pairs.extend(qa_pairs)

            print(f"  View {view_index}: Generated {len(qa_pairs)} QA pairs")

    print(f"Total QA pairs generated: {len(all_qa_pairs)}")

    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(all_qa_pairs, f, indent=2)

    print(f"Saved QA pairs to {output_path}")

    return all_qa_pairs


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

# utils/image_processing.py
import cv2
import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk
from typing import Dict, Any, List


def _is_black_pixel(bw_orig: np.ndarray, x: float, y: float, r: int = 1) -> bool:
    """
    bw_orig: binary image where 'black' pixels are 0 and white are 255 (uint8).
    x,y: centroid coordinates in image coordinates (x horizontal, y vertical).
    Returns True if majority in (2r+1)x(2r+1) patch around (x,y) are black (0).
    """
    h, w = bw_orig.shape
    xi = int(round(x))
    yi = int(round(y))
    if xi < 0 or yi < 0 or xi >= w or yi >= h:
        return False
    x0 = max(0, xi - r)
    x1 = min(w - 1, xi + r)
    y0 = max(0, yi - r)
    y1 = min(h - 1, yi + r)
    patch = bw_orig[y0:y1 + 1, x0:x1 + 1]
    # black pixels are 0
    black_ratio = np.sum(patch == 0) / float(patch.size)
    return black_ratio > 0.5


def analyze_sheet(image_path: str) -> Dict[str, Any]:
    """
    Main pipeline that mimics the MATLAB script.
    Returns JSON-serializable dict with staff spacing, num_staff and notes.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # 1. Adaptive thresholding
    # bw_inv: notes + staff will be WHITE (255). (used for morphology)
    bw_inv = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 35, 10)
    # bw_orig: original polar (black notes => 0)
    bw_orig = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 35, 10)

    # 2. Detect and remove staff lines
    se_line = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    staff_lines = cv2.morphologyEx(bw_inv, cv2.MORPH_OPEN, se_line)
    bw_no_staff = cv2.bitwise_and(bw_inv, cv2.bitwise_not(staff_lines))

    # get projection and yLines similar to MATLAB
    proj = np.sum(staff_lines > 0, axis=1)
    if proj.size == 0:
        y_lines = np.array([], dtype=int)
    else:
        thresh = proj.max() * 0.5
        y_lines = np.where(proj > thresh)[0]

    # group nearby lines
    y_groups = []
    if y_lines.size > 0:
        diffs = np.diff(y_lines)
        # split at gaps > 2
        splits = np.where(diffs > 2)[0]
        start = 0
        for sp in splits:
            segment = y_lines[start:sp + 1]
            y_groups.append(int(np.round(segment.mean())))
            start = sp + 1
        # final segment
        segment = y_lines[start:]
        if segment.size > 0:
            y_groups.append(int(np.round(segment.mean())))

    num_staff = max(0, len(y_groups) // 5)
    staff_spacing = []
    for s in range(num_staff):
        idxStart = s * 5
        yStaff = y_groups[idxStart: idxStart + 5]
        if len(yStaff) == 5:
            spacing = float(np.mean(np.diff(yStaff)))
            staff_spacing.append(spacing)
        else:
            staff_spacing.append(None)

    # 3. Clean (close + remove small objects)
    # closing to connect parts, then remove tiny blobs using area threshold
    # convert to boolean for skimage operations where needed
    # already bw_no_staff has white foreground (255). Prepare binary boolean (True where foreground)
    bin_no_staff = (bw_no_staff > 0).astype(np.uint8) * 255
    # morphological close with disk radius 2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bin_no_staff = cv2.morphologyEx(bin_no_staff, cv2.MORPH_CLOSE, kernel)
    # fill holes: use flood fill on inverse
    inv = cv2.bitwise_not(bin_no_staff)
    h, w = inv.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    filled = bin_no_staff.copy()
    cv2.floodFill(filled, mask, (0, 0), 255)
    filled_inv = cv2.bitwise_not(filled)
    bin_no_staff = cv2.bitwise_or(bin_no_staff, filled_inv)
    # remove small objects using connected components area threshold (30 px)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((bin_no_staff > 0).astype(np.uint8))
    cleaned = np.zeros_like(bin_no_staff)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 30:
            cleaned[labels == i] = 255

    # label and regionprops
    lbl = label(cleaned > 0)
    props = regionprops(lbl)

    # compute mean staff spacing if available
    mean_staff_spacing = np.mean([s for s in staff_spacing if (s is not None)]) if any(s is not None for s in staff_spacing) else 10.0

    # 4. Filter ellipse-like regions similar to MATLAB
    selected_props = []
    for p in props:
        # regionprops gives major_axis_length, minor_axis_length, orientation (radians)
        if p.minor_axis_length == 0:
            continue
        ratio = p.major_axis_length / (p.minor_axis_length + 1e-9)
        orientation_deg = np.degrees(abs(p.orientation))  # use magnitude
        bb_height = p.bbox[2] - p.bbox[0]  # rows height
        if (orientation_deg < 70) and (0.5 < ratio < 2.2) and (bb_height > (mean_staff_spacing - 1)):
            selected_props.append(p)

    stats = selected_props

    # 4.5 classify note type (đen / trắng / tròn)
    note_types = []
    for p in stats:
        cx = p.centroid[1]  # x coordinate (col)
        cy = p.centroid[0]  # y coordinate (row)
        # use original binarized (bw_orig) to check black pixel
        if _is_black_pixel(bw_orig, cx, cy, r=1):
            t = "đen"
        else:
            # in MATLAB they used orientation < 10 for "tròn" else "trắng"
            orientation_deg = np.degrees(abs(p.orientation))
            t = "tròn" if orientation_deg < 10 else "trắng"
        note_types.append(t)

    # 4.6 detect stems
    stem_types = []
    for p in stats:
        x0 = int(round(p.centroid[1]))
        y0 = int(round(p.centroid[0]))
        radius = p.minor_axis_length / 2.0
        x_left = max(0, x0 - 3)
        x_right = min(cleaned.shape[1] - 1, x0 + 3)
        y_top = max(0, int(round(y0 - 4 * radius)))
        y_bottom = min(cleaned.shape[0] - 1, int(round(y0 + 4 * radius)))
        patch = cleaned[y_top:y_bottom + 1, x_left:x_right + 1] > 0
        if patch.size == 0:
            stem_types.append("không đuôi")
            continue
        vert_proj = np.sum(patch, axis=1)
        if vert_proj.size == 0 or vert_proj.max() == 0:
            stem_types.append("không đuôi")
            continue
        peak_count = np.sum(vert_proj > 0.5 * vert_proj.max())
        if peak_count > (3.5 * radius):
            stem_types.append("có đuôi")
        else:
            stem_types.append("không đuôi")

    # 4.7 detect hooks (single/double) using connected components in small patch above
    hook_types = []
    for idx, p in enumerate(stats):
        if stem_types[idx] == "có đuôi":
            x0 = int(round(p.centroid[1]))
            y0 = int(round(p.centroid[0]))
            height = int(round(p.major_axis_length))
            yTop = max(0, y0 - 3 * height)
            x_left = max(0, x0 - 5)
            x_right = min(cleaned.shape[1] - 1, x0 + 5)
            patch = cleaned[yTop:y0 + 1, x_left:x_right + 1]
            if patch.size == 0:
                hook_types.append("không móc")
            else:
                num_cc, labels, stats_cc, centroids_cc = cv2.connectedComponentsWithStats((patch > 0).astype(np.uint8))
                num_objects = num_cc - 1
                if num_objects > 1:
                    hook_types.append("móc kép")
                elif num_objects == 1:
                    hook_types.append("móc đơn")
                else:
                    hook_types.append("không móc")
        else:
            hook_types.append("không móc")

    # 5. Map to staff and name notes (Treble clef mapping used same noteOrder)
    noteOrder = ['A3','B3','C4','D4','E4','F4','G4','A4','B4','C5','D5','E5','F5','G5','A5']
    StaffData = [{"notes": []} for _ in range(max(1, num_staff))]

    y_groups_arr = np.array(y_groups) if len(y_groups) > 0 else np.array([])

    notes_out = []
    for k, p in enumerate(stats):
        x0 = float(p.centroid[1])
        y0 = float(p.centroid[0])
        # find nearest line index
        if y_groups_arr.size == 0:
            staff_idx = 0
        else:
            staffIdx = int(np.argmin(np.abs(y0 - y_groups_arr)))
            groupIdx = int(np.ceil((staffIdx + 1) / 5.0))  # MATLAB used ceil(staffIdx/5)
            groupIdx = max(1, groupIdx)
            groupIdx = min(groupIdx, max(1, num_staff))
            staff_idx = groupIdx - 1

        # determine yStaff lines for that staff block
        baseIdx = staff_idx * 5
        if len(y_groups) >= baseIdx + 5:
            yStaff = y_groups[baseIdx:baseIdx+5]
            spacing = float(np.mean(np.diff(yStaff)))
            yRef = yStaff[-1]
            pos = int(round((yRef - y0) / (spacing / 2.0))) if spacing != 0 else 0
            idxNote = pos + 5
            if idxNote >= 1 and idxNote <= len(noteOrder):
                noteName = noteOrder[idxNote - 1]  # index shift
            else:
                noteName = '?'
        else:
            noteName = '?'

        # assemble
        note_rec = {
            "staff": staff_idx + 1,
            "name": noteName,
            "type": note_types[k],
            "stem": stem_types[k],
            "hook": hook_types[k],
            "position": [round(x0, 1), round(y0, 1)]
        }
        notes_out.append(note_rec)
        # also push into StaffData
        StaffData[staff_idx]["notes"].append(note_rec)

    result = {
        "num_staff": num_staff,
        "staff_spacing": staff_spacing,
        "notes": notes_out,
        "staffs": StaffData,
        "meta": {
            "detected_ellipses": len(stats)
        }
    }

    return result

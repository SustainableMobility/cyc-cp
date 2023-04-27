import csv

file = open("gt.txt")
for row in csv.reader(file):
    frame_id = row[0]
    object_id = row[1]
    object_type = row[2]    # Supported object types are "vehicle", "pedestrian" or "DontCare"
    is_truncated = row[3]   # Boolean 0 (non-truncated), 1 (truncated),  truncated refers to the object leaving image
    # boundaries
    is_occluded = row[4]    # This is set to "unknown" and is not calculated currently
    alpha = row[5]          # Observation angle of object
    bbox_2d = [row[6], row[7], row[8], row[9]]  # 2D bounding box pixel coordinates
    dimensions = [row[10], row[11], row[12]]    # 3D object dimensions: height, width, length (in meters)
    location = [row[13], row[14], row[15]]      # 3D object location x,y,z in camera coordinates (in meters)
    relative_y_rotation = row[16]               # Rotation ry around Y-axis in camera coordinates [-pi..pi]
    velocity = [row[17], row[18], row[19]]      # object velocity in X, Y and Z directions
    camera_extrinsic_matrix = [
        [row[20], row[21], row[22], row[23]],
        [row[24], row[25], row[26], row[27]],
        [row[28], row[29], row[30], row[31]],
        [row[32], row[33], row[34], row[35]]
    ]   # Extrinsic matrix of the RGB camera
    is_near_miss = row[36]  # Boolean 0 (not involved in a near-miss), 1 (involved in a near-miss)
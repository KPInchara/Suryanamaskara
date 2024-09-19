import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load an image or video frame of the asana in correct form
image_path = r"C:\Users\Inchara K P\Downloads\as5.jpeg"
# Read the image
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image
results = pose.process(image_rgb)

# Extract landmarks
landmarks = results.pose_landmarks.landmark

# Define landmarks
left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Calculate angles
left_hip_angle = calculate_angle(left_knee, left_hip, left_shoulder)
right_hip_angle = calculate_angle(right_knee, right_hip, right_shoulder)
left_knee_angle = calculate_angle(left_ankle, left_knee, left_hip)
right_knee_angle = calculate_angle(right_ankle, right_knee, right_hip)
left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_ankle)
right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_ankle)
left_shoulder_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
right_shoulder_angle = calculate_angle(right_hip, right_shoulder, right_elbow)

# Print calculated angles
print(f"Left hip angle: {left_hip_angle}, Right hip angle: {right_hip_angle}")
print(f"Left knee angle: {left_knee_angle}, Right knee angle: {right_knee_angle}")
print(f"Left elbow angle: {left_elbow_angle}, Right elbow angle: {right_elbow_angle}")
print(f"Left shoulder angle: {left_shoulder_angle}, Right shoulder angle: {right_shoulder_angle}")

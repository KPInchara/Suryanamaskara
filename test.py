import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    
    return angle

# Reference angles for the poses
reference_angles = {
    "Tadasana": {"left_shoulder_angle": 180, "right_shoulder_angle": 180, "left_hip_angle": 180, "right_hip_angle": 180},
    "Uttanasana": {"left_hip_angle": 90, "right_hip_angle": 90, "left_knee_angle": 180, "right_knee_angle": 180},
    "Ashwa Sanchalanasana": {"left_knee_angle": 170, "right_knee_angle": 180, "left_hip_angle": 170, "right_hip_angle": 175, "left_elbow_angle": 180, "right_elbow_angle": 180, "Left shoulder angle": 40, "right_shoulder angle": 40},
   "Adho Mukha Svanasana": { "left_hip_angle": 80, "right_hip_angle": 70, "left_knee_angle": 160, "right_knee_angle": 160, "left_elbow_angle": 180, "right_elbow_angle": 180, "left_shoulder_angle": 150, "right_shoulder_angle": 160},
    "Chaturanga Dandasana": {"left_elbow_angle": 90, "right_elbow_angle": 90, "left_hip_angle": 180, "right_hip_angle": 180},
    "Bhujangasana": {"left_shoulder_angle": 90, "right_shoulder_angle": 90, "left_hip_angle": 180, "right_hip_angle": 180},
    "Phalakasana": {"left_hip_angle": 180, "right_hip_angle": 180, "left_knee_angle": 180, "right_knee_angle": 180},
#     "Pranamasana": {"left_hip_angle": 180, "right_hip_angle": 180, "left_knee_angle": 180, "right_knee_angle": 180, "left_elbow_angle": 160,
#   "right_elbow_angle": 160},
}

# Corrections for different angles
correction_guidelines = {
    "left_shoulder_angle": ["Raise your left arm higher by {:.2f}%", "Lower your left arm by {:.2f}%"],
    "right_shoulder_angle": ["Raise your right arm higher by {:.2f}%", "Lower your right arm by {:.2f}%"],
    "left_elbow_angle": ["Bend your left elbow more by {:.2f}%", "Straighten your left elbow by {:.2f}%"],
    "right_elbow_angle": ["Bend your right elbow more by {:.2f}%", "Straighten your right elbow by {:.2f}%"],
    "left_hip_angle": ["Raise your left hip by {:.2f}%", "Lower your left hip by {:.2f}%"],
    "right_hip_angle": ["Raise your right hip by {:.2f}%", "Lower your right hip by {:.2f}%"],
    "left_knee_angle": ["Bend your left knee more by {:.2f}%", "Straighten your left knee by {:.2f}%"],
    "right_knee_angle": ["Bend your right knee more by {:.2f}%", "Straighten your right knee by {:.2f}%"],
}

# Define function to classify poses and provide corrections
def classify_and_correct_pose(landmarks):
    # Extract required landmarks
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    
    # Calculate angles
    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    left_shoulder_angle = calculate_angle(left_elbow, left_shoulder, left_hip)
    right_shoulder_angle = calculate_angle(right_elbow, right_shoulder, right_hip)
    left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

    pose_name = "Unknown Pose"
    corrections = []
    accuracy = 0

    # Example thresholds for pose classification
    if left_shoulder_angle > 160 and right_shoulder_angle > 160 and left_hip_angle > 160 and right_hip_angle > 160:
        pose_name = "Tadasana"
        reference = reference_angles[pose_name]
    elif left_hip_angle < 90 and right_hip_angle < 90 and left_knee_angle > 160 and right_knee_angle > 160:
        pose_name = "Uttanasana"
        reference = reference_angles[pose_name]
    elif left_elbow_angle < 90 and right_elbow_angle < 90 and left_hip_angle > 160 and right_hip_angle > 160:
        pose_name = "Chaturanga Dandasana"
        reference = reference_angles[pose_name]
    elif left_shoulder_angle < 30 and right_shoulder_angle < 30 and left_hip_angle < 160 and right_hip_angle < 160:
        pose_name = "Bhujangasana"
        reference = reference_angles[pose_name]
    elif left_hip_angle > 160 and left_hip_angle <= 180 and right_hip_angle > 160 and right_hip_angle <= 180 and left_knee_angle > 160 and left_knee_angle <= 180 and right_knee_angle > 160 and right_knee_angle <= 180:
        pose_name = "Phalakasana"
        reference = reference_angles[pose_name]
    elif (left_elbow_angle and right_elbow_angle >170) and (left_shoulder_angle and right_shoulder_angle >32)and left_knee_angle > 140 and right_knee_angle > 150 and left_hip_angle > 160 and right_hip_angle >155:
        pose_name = "Ashwa Sanchalanasana"
        # print(left_elbow_angle,right_elbow_angle,left_shoulder_angle,right_shoulder_angle,left_hip_angle,right_hip_angle,left_knee_angle,right_knee_angle)
        reference = reference_angles[pose_name]    
    elif left_elbow_angle>170 and left_elbow_angle<180 and right_elbow_angle >175 and right_elbow_angle <185 and left_shoulder_angle > 135 and right_shoulder_angle > 150 and left_hip_angle and right_hip_angle >60 and left_hip_angle and right_hip_angle <75 and left_knee_angle >125 and right_knee_angle >145:
        pose_name = "Adho Mukha Svanasana"
        reference = reference_angles[pose_name]       
    # elif left_elbow_angle < 160 and right_elbow_angle < 160 and left_hip_angle < 180 and right_hip_angle < 180 and left_knee_angle < 180 and right_knee_angle < 180:
    #     pose_name = "Pranamasana"
    #     reference = reference_angles[pose_name]

    else:
        reference = {}
        #print(left_elbow_angle,right_elbow_angle,left_shoulder_angle,right_shoulder_angle,left_hip_angle,right_hip_angle,left_knee_angle,right_knee_angle)
    

    if pose_name != "Unknown Pose":
        if(pose_name== "Adho Mukha Svanasana"):{
            print(left_elbow_angle,right_elbow_angle,left_shoulder_angle,right_shoulder_angle,left_hip_angle,right_hip_angle,left_knee_angle,right_knee_angle)  
        
        }
        # Compare detected angles with reference angles and provide corrections
        for angle_name, detected_angle in {
            "left_shoulder_angle": left_shoulder_angle,
            "right_shoulder_angle": right_shoulder_angle,
            "left_elbow_angle": left_elbow_angle,
            "right_elbow_angle": right_elbow_angle,
            "left_hip_angle": left_hip_angle,
            "right_hip_angle": right_hip_angle,
            "left_knee_angle": left_knee_angle,
            "right_knee_angle": right_knee_angle
        }.items():
            if angle_name in reference:
                reference_angle = reference[angle_name]
                deviation = abs(detected_angle - reference_angle)
                percentage_deviation = (deviation / reference_angle) * 100

                if percentage_deviation > 10:
                    if detected_angle > reference_angle:
                        corrections.append(correction_guidelines[angle_name][0].format(percentage_deviation))
                    else:
                        corrections.append(correction_guidelines[angle_name][1].format(percentage_deviation))

        # Calculate accuracy as (1 - average deviation / max deviation) * 100
        max_deviation = 180  # Max angle deviation considered
        average_deviation = np.mean([
            abs(left_shoulder_angle - reference.get("left_shoulder_angle", 180)),
            abs(right_shoulder_angle - reference.get("right_shoulder_angle", 180)),
            abs(left_elbow_angle - reference.get("left_elbow_angle", 180)),
            abs(right_elbow_angle - reference.get("right_elbow_angle", 180)),
            abs(left_hip_angle - reference.get("left_hip_angle", 180)),
            abs(right_hip_angle - reference.get("right_hip_angle", 180)),
            abs(left_knee_angle - reference.get("left_knee_angle", 180)),
            abs(right_knee_angle - reference.get("right_knee_angle", 180))
        ])
        accuracy = max(0, (1 - average_deviation / max_deviation) * 100)

    # Determine color based on accuracy
    accuracy_color = (0, 255, 0) if accuracy > 80 else (0, 0, 255)
    print(pose_name,accuracy)
    return pose_name, corrections, accuracy, accuracy_color

# Example usage with a captured frame and detected landmarks
cap = cv2.VideoCapture(0)  # Capture from the default camera

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and find landmarks
    results = pose.process(rgb_frame)

    # Check if landmarks are detected
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Draw landmarks on the frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Classify pose and get corrections and accuracy
        pose_name, corrections, accuracy, accuracy_color = classify_and_correct_pose(landmarks)

        # Display the pose name and accuracy on the frame
        cv2.putText(frame, f"Pose: {pose_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Accuracy: {accuracy:.2f}%", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, accuracy_color, 2, cv2.LINE_AA)

        # Display the corrections
        y_offset = 110
        for correction in corrections:
            cv2.putText(frame, correction, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            y_offset += 30

    # Display the frame
    cv2.imshow('Pose Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()



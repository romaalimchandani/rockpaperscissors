import cv2
import mediapipe as mp
import time
import random

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Determine the winner
def get_winner(player, computer):
    if player == computer:
        return "Draw"
    if (player == "Rock" and computer == "Scissors") or \
       (player == "Paper" and computer == "Rock") or \
       (player == "Scissors" and computer == "Paper"):
        return "Player"
    return "Computer"

# Detect hand gesture
def detect_gesture(landmarks):
    fingers = []

    # Finger tip ids: index, middle, ring, pinky
    tips_ids = [8, 12, 16, 20]

    for i in range(0, 4):
        tip = landmarks.landmark[tips_ids[i]].y
        pip = landmarks.landmark[tips_ids[i] - 2].y
        fingers.append(tip < pip)

    if fingers == [False, False, False, False]:
        return "Rock"
    elif fingers == [True, True, True, True]:
        return "Paper"
    elif fingers == [True, True, False, False]:
        return "Scissors"
    return "Unknown"

# Main game loop
def play_game():
    cap = cv2.VideoCapture(0)
    result = ""
    move = ""
    label = "Press SPACE to play"
    result_color = (255, 255, 255)

    with mp_hands.Hands(min_detection_confidence=0.75, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result_hands = hands.process(rgb)

            if result_hands.multi_hand_landmarks:
                for hand_landmarks in result_hands.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    move = detect_gesture(hand_landmarks)

            # Display result
            if result:
                cv2.putText(frame, result, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, result_color, 2)

            # Display label
            cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            cv2.imshow("Rock Paper Scissors", frame)

            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE
                if move in ["Rock", "Paper", "Scissors"]:
                    comp_move = random.choice(["Rock", "Paper", "Scissors"])
                    winner = get_winner(move, comp_move)

                    if winner == "Draw":
                        result = f"Draw! You: {move}, Computer: {comp_move}"
                        result_color = (0, 255, 255)
                    elif winner == "Player":
                        result = f"You win! You: {move}, Computer: {comp_move}"
                        result_color = (0, 255, 0)
                    else:
                        result = f"You lose! You: {move}, Computer: {comp_move}"
                        result_color = (0, 0, 255)
                else:
                    result = "Couldn't read your move!"
                    result_color = (200, 200, 200)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    play_game()

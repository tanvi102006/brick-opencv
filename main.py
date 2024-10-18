import cv2
import numpy as np
import mediapipe as mp

# Initialize Mediapipe for hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Game variables
width, height = 800, 600
paddle_width, paddle_height = 100, 20
ball_radius = 10
ball_speed = [15, -15]
paddle_x = (width - paddle_width) // 2
ball_x, ball_y = width // 2, height - 50
bricks = [(x, y) for x in range(0, width, 80) for y in range(0, 200, 40)]
game_over = False

# Function to draw the game elements
def draw_game(frame):
    global paddle_x, ball_x, ball_y
    frame.fill(0)
    
    # Draw bricks
    for brick in bricks:
        cv2.rectangle(frame, brick, (brick[0] + 80, brick[1] + 40), (0, 255, 0), -1)
    
    # Draw paddle
    cv2.rectangle(frame, (paddle_x, height - paddle_height), (paddle_x + paddle_width, height), (255, 0, 0), -1)
    
    # Draw ball
    cv2.circle(frame, (ball_x, ball_y), ball_radius, (0, 0, 255), -1)
    
# Main game loop
cap = cv2.VideoCapture(0)

while not game_over:
    ret, frame = cap.read()
    if not ret:
        break

    # Hand tracking
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        # Get the x position of the wrist
        wrist_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * width)
        # Invert the paddle control
        paddle_x = max(0, min(width - paddle_width, width - wrist_x - paddle_width // 2))

    # Update ball position
    ball_x += ball_speed[0]
    ball_y += ball_speed[1]

    # Ball collision with walls
    if ball_x <= ball_radius or ball_x >= width - ball_radius:
        ball_speed[0] = -ball_speed[0]
    if ball_y <= ball_radius:
        ball_speed[1] = -ball_speed[1]

    # Ball collision with paddle
    if (height - paddle_height <= ball_y + ball_radius <= height) and (paddle_x <= ball_x <= paddle_x + paddle_width):
        ball_speed[1] = -ball_speed[1]

    # Ball collision with bricks
    for brick in bricks[:]:
        if (brick[0] <= ball_x <= brick[0] + 80) and (brick[1] <= ball_y <= brick[1] + 40):
            bricks.remove(brick)
            ball_speed[1] = -ball_speed[1]
            break

    # Check if ball falls
    if ball_y > height:
        game_over = True

    # Draw the game
    draw_game(frame)
    cv2.imshow('Brick Breaker', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

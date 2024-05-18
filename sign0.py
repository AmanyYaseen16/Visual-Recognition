import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands  # track hands when camera is on (hands module from mediapipe.solutions package for hand tracking)
hands = mp_hands.Hands()  # instance of the Hands class to use its methods such as .process() to analyze frames and obtain hand tracking results.
mp_draw = mp.solutions.drawing_utils  # to draw lines and dots of detected hands

tips = [4, 8, 12, 16, 20]  # array with the fingertips landmarks
ily_gest = [1, 1, 0, 0, 1]  # define the gesture "I love you" in sign language.
ily_gest_l = [1, 0, 0, 1, 1]
ThumpUp_gest = [1, 0, 0, 0, 0]
Ok_gest = [0, 0, 1, 1, 1]
peace_gest = [0, 1, 1, 0, 0]
pointup_guest = [0, 1, 0, 0, 0]

vid = cv2.VideoCapture(0)  # to use the camera of laptop so we write 0

# to start the camera and takes frames from video
while True:
    st, Frame = vid.read()  # st boolean , read the video as frames

    # we make the camera to read real dimensions by flip that transfer from right to left
    # 1 means to flip on y-axis
    Frame = cv2.flip(Frame, 1)  # so left side of the frame corresponds to the user's left side

    image = cv2.cvtColor(Frame, cv2.COLOR_BGR2RGB)  # transfer it to RGB
    results = hands.process(image)  # to apply algorithm and detect hands in the frame
    list_lm = []  # empty list will contain landmark's position

    # if he detects hand
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:  # for loop on the detected hands
            mp_draw.draw_landmarks(Frame, hand, mp_hands.HAND_CONNECTIONS)

            # from the return values on hands.process() function is the handedness of detected hand
            # so, we want to access that return and use if-conditions.
            hand_dir = " "
            handedness = results.multi_handedness[0].classification[0].label
            if handedness == "Right":
                hand_dir = "Right Hand"
            elif handedness == "Left":
                hand_dir = "Left Hand"

            for ID, lm in enumerate(hand.landmark):  # id for every landmark in hand that is detected
                h, w, c = Frame.shape  # to get the dimensions of frame
                cx, cy = int(lm.x * w), int(
                    lm.y * h)  # to get the position of points in frame (x and y coordinates of each landmark)
                list_lm.append([ID, cx, cy])  # append in the list id and position for each landmark

                if len(list_lm) == 21:  # when the length reach 21 the number of landmarks
                    # print(list_lm)
                    # print(tips)
                    fingers = []  # the number that will be generated

                    # because the thump can't be detected closed from y-axis, so we will detect it by x-axis for each
                    # hand
                    if hand_dir == "Right Hand":
                        if list_lm[tips[0]][1] < list_lm[tips[0] - 2][
                            1]:  # tips[0] -> index of the first element in the tips array (4)
                            fingers.append(1)  # put number 1 if the tip is open
                            # list_lm[tips[0]][1] -> access the x coordinate of the first fingertip landmark
                            # tips[0]-2 -> the index of the second landmark in the same finger.

                            # print(tips[0])
                            # print(list_lm[tips[0]][1])
                            # print(list_lm[tips[0] - 2][1])
                            # print(tips[0]-2)
                        else:
                            # put number 0 if the tip is closed
                            fingers.append(0)
                    else:
                        if list_lm[tips[0]][1] > list_lm[tips[0] - 2][1]:
                            fingers.append(1)  # put number 1 if the tip is open
                        else:
                            # put number 0 if the tip is closed
                            fingers.append(0)

                    # detect rest of the fingers
                    for t in range(1, 5):  # for loop around the tips after the thumb
                        # if the height of tips is smaller than the height of the second landmark in the hand,
                        # so it is number 1 else 0
                        if list_lm[tips[t]][2] < list_lm[tips[t] - 2][
                            2]:  # it means that the fingertip is higher than the base of the finger, indicating that the finger is open.
                            fingers.append(1)  # put number 1 if the tip is open
                        else:
                            fingers.append(0)  # put number 0 if the tip is closed

                    total = fingers.count(1)  # to sum number of 1 if it opens to know the number

                    # to print the number we will put the frame, text , position, text font , size , color , thickness
                    # open/closed hand
                    if total == 0:
                        cv2.putText(Frame, f'Hand is closed', (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    elif total == 5:
                        cv2.putText(Frame, f'Hand is opened', (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                    # number of fingers opened
                    cv2.putText(Frame, f'Numbers held : {total}', (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                                2)

                    # left/right hand
                    cv2.putText(Frame, f'Hand 1: {hand_dir}', (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Compare the finger positions with the "I love you" gesture
                    if fingers == ily_gest:
                        cv2.putText(Frame, "I love you", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # Compare the finger positions with the "thumbs up" gesture
                    if fingers == ThumpUp_gest:
                        cv2.putText(Frame, "Thumbs Up", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # Compare the finger positions with the "ok" gesture
                    if fingers == Ok_gest:
                        cv2.putText(Frame, "Ok", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # Compare the finger positions with the "peace" gesture
                    if fingers == peace_gest:
                        cv2.putText(Frame, "Peace", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    if fingers == pointup_guest:
                        cv2.putText(Frame, "Point Up", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # to show frames
    cv2.imshow("Hand Tracking", Frame)
    # to close the camera
    if cv2.waitKey(33) & 0xFF == ord('l'):
        break
vid.release()

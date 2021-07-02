# GET NUMBER OF KEYPOINTS IN acquireData
            # cap = cv2.VideoCapture(0)
            # with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            #     while cap.isOpened():
            #         _, frame = cap.read()

            #         # Recolor Feed. We need this bc mp works with RGB but we have BGR
            #         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            #         # Make Detections (find keypoints). Results are on: results.face_landmarks, pose_landmarks, left_hand_landmarks and right_hand_landmarks
            #         image.flags.writeable = False
            #         results = holistic.process(image)
            #         image.flags.writeable = True
            #         break
            #     cap.release()
            # num_coords = len(results.pose_landmarks.landmark) + len(results.face_landmarks.landmark)
            # face_landmarks+left_hand+right_hand+pose_landmarks

# DRAW UNUSED LANDMARKS

            # DRAW LANDMARKS

            # # Draw face landmarks. 468 landmarks
            # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS,
            #                           mp_drawing.DrawingSpec(
            #                               color=(80, 110, 10), thickness=1, circle_radius=1),
            #                           mp_drawing.DrawingSpec(
            #                               color=(80, 256, 121), thickness=1, circle_radius=1)
            #                           )

            # # Draw right hand landmarks
            # mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            #                           mp_drawing.DrawingSpec(
            #                               color=(80, 22, 10), thickness=2, circle_radius=4),
            #                           mp_drawing.DrawingSpec(
            #                               color=(80, 44, 121), thickness=2, circle_radius=2)
            #                           )

            # # Draw left hand landmarks
            # mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            #                           mp_drawing.DrawingSpec(
            #                               color=(121, 22, 76), thickness=2, circle_radius=4),
            #                           mp_drawing.DrawingSpec(
            #                               color=(121, 44, 250), thickness=2, circle_radius=2)
            #                           )

# GET UNUSED KEYPOINTS
                # face = results.face_landmarks.landmark
                # face_row = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in face]).flatten())     

                # r_hand = results.right_hand_landmarks.landmark
                # r_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in r_hand]).flatten())

                # l_hand = results.left_hand_landmarks.landmark
                # l_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in l_hand]).flatten())
                
                #row = pose_row + face_row + r_hand_row + l_hand_row
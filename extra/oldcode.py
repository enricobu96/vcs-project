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


# IMAGE FILTERING KINECT
                #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # image = np.concatenate((image, image, image), axis=0)
                # image = np.swapaxes(image, 0, 2)
                # image = np.swapaxes(image, 0, 1)

# DEPTH CAMERA KINECT
                # """
                # DEPTH CAMERA
                # Retrieve depth camera frame and do image filtering on that
                # """
                # frame_dep = depth_camera.read_frame()
                # frame_dep_data = frame_dep.get_buffer_as_uint16()
                # image_dep = np.frombuffer(frame_dep_data, dtype=np.uint8)
                # image_dep.shape = (480,640,2)
                # firstChannel, secondChannel = cv2.split(image_dep)


# TRY TO SAVE SOME FRAMES
        # save_for_later = []
        # frame_count = 0
        # i = 0
        # # SAVE DATA FOR LATER
        #     if frame_count == 0:
        #         if image is not None:
        #             save_for_later.append(image)
        #             frame_count += 1
        #             i += 1
        #     elif frame_count == 15:
        #         frame_count = 0
        #     else:
        #         frame_count += 1
        # for image in save_for_later:
        #        try:
        #        print(image)
        #        sleep(1)
        #        image_flip, dark_image, bright_image, resized_image = self.__data_augmentation(image)
        #        ifresults = holistic.process(image_flip)
        #        diresults = holistic.process(dark_image)
        #        biresults = holistic.process(bright_image)
        #        riresults = holistic.process(resized_image)
        #        row_pose_if = ifresults.pose_landmarks.landmark
        #        row_pose_di = diresults.pose_landmarks.landmark
        #        row_pose_bi = biresults.pose_landmarks.landmark
        #        row_pose_ri = riresults.pose_landmarks.landmark
        #        row_pose_if.insert(0, gesture)
        #        row_pose_di.insert(0, gesture)
        #        row_pose_bi.insert(0, gesture)
        #        row_pose_ri.insert(0, gesture)
        #        with open('./dataset/keypoints/coords_mediapipe.csv', mode='a', newline='') as f:
        #            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #            csv_writer.writerow(row_pose_if)
        #            csv_writer.writerow(row_pose_di)
        #            csv_writer.writerow(row_pose_bi)
        #            csv_writer.writerow(row_pose_ri)
        #    except Exception as e:
        #        print('Error in post processing (do not worry: ' + str(e))
        #        pass
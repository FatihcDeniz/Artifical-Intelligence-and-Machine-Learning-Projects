import tkinter as tk
from PIL import Image, ImageTk
import cv2
from tkinter import DISABLED, ACTIVE, Button
import mediapipe as mp
import numpy as np

mpfacemesh = mp.solutions.face_mesh
facemesh = mpfacemesh.FaceMesh(static_image_mode=False, max_num_faces=2, refine_landmarks=True)
mpdraw = mp.solutions.drawing_utils
drawing_spec = mpdraw.DrawingSpec(thickness=1, circle_radius=1)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

params = cv2.SimpleBlobDetector_Params()
# Filter by Area.
## Add blob parameters to takinter
params.filterByArea = True
params.maxArea = 1500

# params.minThreshold = 10
# params.maxThreshold = 200

params.filterByCircularity = True
# params.minCircularity = 0.5

params.filterByConvexity = True
# params.minConvexity = 0.87

params.filterByInertia = True
# params.minInertiaRatio = 0.01

headpos = True
font = cv2.FONT_HERSHEY_SIMPLEX


def initialize_blob():
    ver = cv2.__version__.split('.')
    if int(ver[0]) < 3:
        detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params)
    return detector


detector = initialize_blob()


class Window:
    def __init__(self, window, cap):
        self.window = window
        self.cap = cap
        self.face_height = 400
        self.face_width = 400
        self.eye_height = 100
        self.eye_width = 100
        self.interval = 20

        self.raw_right = True
        self.raw_left = True
        self.contour_left = False
        self.contour_right = False
        self.mesh = False
        self.head_position = False
        self.show_face = False
        self.eye_position = False

        self.canvas = tk.Canvas(self.window, width=self.face_height, height=self.face_width)
        self.canvas.grid(row=0, column=0)

        self.canvas2 = tk.Canvas(self.window, width=self.eye_height, height=self.eye_width)
        self.canvas2.grid(row=0, column=1)

        self.canvas3 = tk.Canvas(self.window, width=self.eye_height, height=self.eye_width)
        self.canvas3.grid(row=0, column=2)

        self.var = tk.DoubleVar()
        self.scale = tk.Scale(self.window, variable=self.var, orient=tk.HORIZONTAL, from_=1, to=100, label='Threshold')
        self.scale.grid(row=1, column=0)

        self.erode = tk.DoubleVar()
        self.scale = tk.Scale(self.window, variable=self.erode, orient=tk.HORIZONTAL, from_=1, to=10, label='Erosion')
        self.scale.grid(row=2, column=0)

        self.dilate = tk.DoubleVar()
        self.scale = tk.Scale(self.window, variable=self.dilate, orient=tk.HORIZONTAL, from_=1, to=10, label='Dilation')
        self.scale.grid(row=3, column=0)

        self.blur = tk.DoubleVar()
        self.scale = tk.Scale(self.window, variable=self.blur, orient=tk.HORIZONTAL, from_=1, to=10, resolution=2,
                              label='Blur')
        self.scale.grid(row=4, column=0)

        self.circularity = tk.DoubleVar()
        self.circularity_scale = tk.Scale(self.window, variable=self.circularity, orient=tk.HORIZONTAL, from_=0, to=1,
                                          resolution=0.05, label="Min Circularity")
        self.circularity_scale.grid(row=1, column=3)

        self.area = tk.DoubleVar()
        self.area_scale = tk.Scale(self.window, variable=self.area, orient=tk.HORIZONTAL, from_=0, to=2000,
                                   resolution=20,
                                   label="Min Area")
        self.area_scale.grid(row=2, column=3)

        self.convexity = tk.DoubleVar()
        self.convexity_scale = tk.Scale(self.window, variable=self.convexity, orient=tk.HORIZONTAL, from_=0, to=1,
                                        resolution=0.05, label="Convexity")
        self.convexity_scale.grid(row=3, column=3)

        self.inertia = tk.DoubleVar()
        self.inertia_scale = tk.Scale(self.window, variable=self.inertia, orient=tk.HORIZONTAL, from_=0, to=1,
                                      resolution=0.05, label="Inertia")
        self.inertia_scale.grid(row=4, column=3)

        self.button_right = tk.Button(self.window, text='Raw version right eye', height=1, width=15,
                                      command=self.press_right)
        self.button_right.grid(row=1, column=2)

        self.button_left = tk.Button(self.window, text='Raw version left eye', height=1, width=15,
                                     command=self.press_left)
        self.button_left.grid(row=1, column=1)

        self.button_contour_left = tk.Button(self.window, text="Contour Right", height=1, width=15,
                                             command=self.contour_r)
        self.button_contour_left.grid(row=2, column=2)

        self.button_contour_right = tk.Button(self.window, text="Contour Left", height=1, width=15,
                                              command=self.contour_l)
        self.button_contour_right.grid(row=2, column=1)

        self.inertia_state = True
        self.button_inertia = tk.Button(self.window, text="Inertia", height=1, width=10,
                                        command=lambda: self.switch_state(self.inertia_state, self.inertia_scale))
        self.button_inertia.grid(row=4, column=4)

        self.convexity_state = True
        self.button_convexity = tk.Button(self.window, text="Convexity", height=1, width=10,
                                          command=lambda: self.switch_state(self.convexity_state, self.convexity_scale))
        self.button_convexity.grid(row=3, column=4)

        self.circularity_state = True
        self.button_circularity = tk.Button(self.window, text="Circularity", height=1, width=10,
                                            command=lambda: self.switch_state(self.circularity_state,
                                                                              self.circularity_scale))
        self.button_circularity.grid(row=1, column=4)

        self.area_state = True
        self.button_area = tk.Button(self.window, text="Min Area", height=1, width=10,
                                     command=lambda: self.switch_state(self.area_state, self.area_scale))
        self.button_area.grid(row=2, column=4)

        self.mesh_button = tk.Button(self.window,text="Face Mesh", height=1,width=10,command=self.face_mesh)
        self.mesh_button.grid(row=3,column=1)

        self.head_position_button = tk.Button(self.window, text="Face Position", height=1, width=10,command=self.head_pos)
        self.head_position_button.grid(row=3,column=2)

        self.face_button = tk.Button(self.window,text="Show Face",width=10,height=1,command=self.show_face_func)
        self.face_button.grid(row=4,column=2)

        self.eye_button = tk.Button(self.window,text="Eye Direction",width=10,height=1,command=self.show_eye_direction)
        self.eye_button.grid(row=4,column=1)

        self.update_screen()
        self.detect_faces()

    def show_eye_direction(self):
        if not self.eye_position:
            self.eye_position = True
        else:
            self.eye_position = False

    def head_pos(self):
        if not self.head_position:
            self.head_position = True
        else:
            self.head_position = False

    def show_face_func(self):
        if not self.show_face:
            self.show_face = True
        else:
            self.show_face = False


    def face_mesh(self):
        if not self.mesh:
            self.mesh = True
        else:
            self.mesh = False

    def switch_state(self, state, scale):
        if not state:
            scale.config(state=ACTIVE)
            state = True
        else:
            scale.config(state=DISABLED)
            state = False

    def detect_faces(self):
        self.gray = cv2.cvtColor(self.cap.read()[1], cv2.COLOR_BGR2GRAY)
        self.face = face_cascade.detectMultiScale(self.gray, 1.3, 5)

    def press_right(self):
        if not self.raw_right:
            self.raw_right = True
            self.contour_right = False
        else:
            self.raw_right = False
            self.contour_right = False

    def press_left(self):
        if not self.raw_left:
            self.raw_left = True
            self.contour_left = False
        else:
            self.raw_left = False
            self.contour_left = False

    def contour_l(self):
        if not self.contour_left:
            self.contour_left = True
            self.raw_left = False
        else:
            self.contour_left = False
            self.raw_left = False

    def contour_r(self):
        if not self.contour_right:
            self.contour_right = True
            self.raw_right = False
        else:
            self.contour_right = False
            self.raw_right = False

    def update_screen(self):
        params.minCircularity = float(self.circularity.get())
        params.minArea = int(self.area.get())
        params.minConvexity = float(self.convexity.get())
        params.minInertiaRatio = float(self.inertia.get())
        detector = initialize_blob()

        self.image = cv2.cvtColor(self.cap.read()[1], cv2.COLOR_BGR2RGB)
        self.results = facemesh.process(self.image)
        self.image_rgb = Image.fromarray(self.image)
        self.image_rgb = ImageTk.PhotoImage(self.image_rgb)
        self.detect_faces()
        self.right_eye_shape = None

        if headpos:
            img_h, img_w, img_c = self.image.shape
            self.face_3d = []
            self.face_2d = []
            self.left_eye_2d = []
            self.left_eye_3d = []
            self.right_eye_2d = []
            self.right_eye_3d = []
            if self.results.multi_face_landmarks:
                for face_landmarks in self.results.multi_face_landmarks:
                    for id, lm in enumerate(face_landmarks.landmark):
                        if id == 33 or id == 263 or id == 1 or id == 61 or id == 291 or id == 199:
                            if id == 1:
                                self.nose_2d = (lm.x * img_w, lm.y * img_h)
                                self.nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                            self.face_2d.append([int(lm.x * img_w), int(lm.y * img_h)])
                            self.face_3d.append([int(lm.x * img_w), int(lm.y * img_h), lm.z])

                        if id == 474 or id == 475 or id == 476 or id == 477:
                            if id == 474:
                                self.left_iris_2d = (lm.x * img_w, lm.y * img_h)
                                self.left_iris_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                            if id == 475:
                                self.left_iris_2d_475 = (lm.x * img_w, lm.y * img_h)
                                self.left_iris_3d_475 = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                            if id == 476:
                                self.left_iris_2d_476 = (lm.x * img_w, lm.y * img_h)
                                self.left_iris_3d_476 = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                            if id == 477:
                                self.left_iris_2d_477 = (lm.x * img_w, lm.y * img_h)
                                self.left_iris_3d_477 = (lm.x * img_w, lm.y * img_h, lm.z * 3000)


                            self.left_eye_2d.append([int(lm.x * img_w), int(lm.y * img_h)])
                            self.left_eye_3d.append([int(lm.x * img_w), int(lm.y * img_h), lm.z])

                        if id == 469 or id == 470 or id == 471 or id ==472:
                            if id == 469:
                                self.right_iris_2d_469 = (lm.x * img_w, lm.y * img_h)
                                self.right_iris_3d_469 = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                            if id == 470:
                                self.right_iris_2d_470 = (lm.x * img_w, lm.y * img_h)
                                self.right_iris_3d_470 = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                            if id == 471:
                                self.right_iris_2d_471 = (lm.x * img_w, lm.y * img_h)
                                self.right_iris_3d_471 = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                            if id == 472:
                                self.right_iris_2d_472 = (lm.x * img_w, lm.y * img_h)
                                self.right_iris_3d_472 = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                            self.right_eye_2d.append([int(lm.x * img_w), int(lm.y * img_h)])
                            self.right_eye_3d.append([int(lm.x * img_w), int(lm.y * img_h), lm.z])


                    self.face_2d = np.array(self.face_2d, dtype=np.float64)
                    self.face_3d = np.array(self.face_3d, dtype=np.float64)

                    self.left_eye_2d = np.array(self.left_eye_2d, dtype=np.float64)
                    self.left_eye_3d = np.array(self.left_eye_3d, dtype=np.float64)

                    self.right_eye_2d = np.array(self.right_eye_2d, dtype=np.float64)
                    self.right_eye_3d = np.array(self.right_eye_3d, dtype=np.float64)

                    self.camera = 1 * img_w
                    self.eye_camera = 1 * int(50)

                    cam_matrix = np.array([[self.camera, 0, img_h / 2],
                                           [0, self.camera, img_w / 2],
                                           [0, 0, 1]])


                    eye_matrix = np.array([[self.eye_camera, 0, int(self.right_eye_2d.shape[1]) / 2],
                                           [0, self.eye_camera, int(self.right_eye_2d.shape[0]) / 2],
                                           [0, 0, 1]])

                    dist_matrix = np.zeros((4, 1), dtype=np.float64)

                    success, rot_vec, trans_vec = cv2.solvePnP(self.face_3d, self.face_2d, cam_matrix, dist_matrix)

                    _, left_rot_vec, left_trans_vec = cv2.solvePnP(self.left_eye_3d, self.left_eye_2d, cam_matrix,
                                                                   dist_matrix)

                    _, right_rot_vec, right_trans_vec = cv2.solvePnP(self.right_eye_3d, self.left_eye_2d, cam_matrix,
                                                                   dist_matrix)
                    # Get rotational matrix
                    rmat, jac = cv2.Rodrigues(rot_vec)
                    left_rmat, _ = cv2.Rodrigues(left_rot_vec)
                    right_rmat , _ = cv2.Rodrigues(right_rot_vec)

                    # Get angles
                    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                    left_angles, _, _, _, _, _ = cv2.RQDecomp3x3(left_rmat)

                    right_angles, _, _, _, _, _ = cv2.RQDecomp3x3(right_rmat)

                    # Get the y rotation degree
                    x = angles[0] * 360
                    y = angles[1] * 360
                    z = angles[2] * 360

                    left_x = left_angles[0] * 360
                    left_y = left_angles[1] * 360
                    left_z = left_angles[2] * 360

                    right_x = right_angles[0] * 360
                    right_y = right_angles[1] * 360
                    right_z = right_angles[2] * 360

                    nose_3d_projection, _ = cv2.projectPoints(self.nose_3d, rot_vec, trans_vec, cam_matrix,
                                                              dist_matrix)

                    left_eye_3d_projection, _ = cv2.projectPoints(self.left_eye_3d, left_rot_vec, left_trans_vec,
                                                                  cam_matrix,
                                                                  dist_matrix)

                    right_eye_3d_projection, _ = cv2.projectPoints(self.right_eye_3d, left_rot_vec, left_trans_vec,
                                                                  cam_matrix,
                                                                  dist_matrix)

                    p1 = (int(self.nose_2d[0]), int(self.nose_2d[1]))
                    p2 = (int(self.nose_2d[0] + y * 10), int(self.nose_2d[1] - x * 10))

                    left_p1 = (int(self.left_iris_2d[0]), int(self.left_iris_2d[1]))
                    left_p2 = (int(self.left_iris_2d[0] + left_y), int(self.left_iris_2d[1] - left_x ))
                    left_p1_475 = (int(self.left_iris_2d_475[0]), int(self.left_iris_2d_475[1]))
                    left_p2_475 = (int(self.left_iris_2d_475[0] + left_y), int(self.left_iris_2d_475[1] - left_x))
                    left_p1_476 = (int(self.left_iris_2d_476[0]), int(self.left_iris_2d_476[1]))
                    left_p2_476 = (int(self.left_iris_2d_476[0] + left_y), int(self.left_iris_2d_476[1] - left_x))
                    left_p1_477 = (int(self.left_iris_2d_477[0]), int(self.left_iris_2d_477[1]))
                    left_p2_477 = (int(self.left_iris_2d_477[0] + left_y ), int(self.left_iris_2d_477[1] - left_x))

                    avg_left_p1 = (int((left_p1_475[0] + left_p1_477[0])/2), int((left_p1_475[1] + left_p1_477[1])/2))
                    avg_left_p2 = (int((left_p2_475[0] + left_p2_477[0])/2), int((left_p2_475[1] + left_p2_477[1])/2))

                    right_p1_469 = (int(self.right_iris_2d_469[0]), int(self.right_iris_2d_469[1]))
                    right_p2_469 = (int(self.right_iris_2d_469[0] + right_y), int(self.right_iris_2d_469[1] - right_x))
                    right_p1_470 = (int(self.right_iris_2d_470[0]), int(self.right_iris_2d_470[1]))
                    right_p2_470 = (int(self.right_iris_2d_470[0] + right_y), int(self.right_iris_2d_470[1] - right_x))
                    right_p1_471 = (int(self.right_iris_2d_471[0]), int(self.right_iris_2d_471[1]))
                    right_p2_471 = (int(self.right_iris_2d_471[0] + right_y), int(self.right_iris_2d_471[1] - right_x))
                    right_p1_472 = (int(self.right_iris_2d_472[0]), int(self.right_iris_2d_472[1]))
                    right_p2_472 = (int(self.right_iris_2d_472[0] + right_y), int(self.right_iris_2d_472[1] - right_x))

                    avg_right_p1 = (
                    int((right_p1_469[0] + right_p1_471[0]) / 2), int((right_p1_469[1] + right_p1_471[1]) / 2))
                    avg_right_p2 = (
                    int((right_p2_469[0] + right_p2_471[0]) / 2), int((right_p2_469[1] + right_p2_471[1]) / 2))

                    if self.head_position:
                        cv2.arrowedLine(self.image, p1, p2, (0, 0, 255), 2, tipLength=0.5)

                    if self.eye_position:
                        cv2.arrowedLine(self.image, avg_left_p1, p2, (0, 255, 0), 3, tipLength=0.5)
                        cv2.arrowedLine(self.image, avg_right_p1, p2, (0, 255, 0), 3, tipLength=0.5)


                    if self.mesh:
                        mpdraw.draw_landmarks(
                            image=self.image,
                            landmark_list=face_landmarks,
                            connections=mpfacemesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=drawing_spec,
                            connection_drawing_spec=drawing_spec)

        for (x, y, w, h) in self.face:
            # Cropping Image by using face location given by face_cascade.
            self.face_gray = self.gray[y:y + w, x:x + w]
            self.face_img = self.image[y:y + w, x:x + w]
            self.face_image = Image.fromarray(self.face_img)
            self.face_image = ImageTk.PhotoImage(self.face_image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.face_image)
            if self.show_face:
                cv2.imshow("Face",cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))
            if not self.show_face:
                cv2.destroyAllWindows()
            self.eyes = eye_cascade.detectMultiScale(self.face_gray, 1.3, 5)

            for (e_x, e_y, e_w, e_h) in self.eyes:
                # Cropping Image by using eye locations given by eye_cascade. Calculating eye centers.
                self.eye_center = int(float(e_x) + (float(e_w) / float(2)))
                self.eye_center_y = int(float(e_y) + (float(e_h) / float(2)))
                if int(self.face_img.shape[0] * 0.1) < self.eye_center < int(self.face_img.shape[1] * 0.4):
                    current_loc = np.linalg.norm(np.array((self.eye_center, self.eye_center_y)) - np.array((e_x, e_y)))

                    self.right_eye = self.face_img[e_y: e_y + e_h, e_x: e_x + e_w]
                    self.right_eye_shape = self.right_eye.shape
                    cv2.putText(self.right_eye, "idk", (self.eye_center, self.eye_center_y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.3, (255, 0, 0), 1)

                    rows, cols, _ = self.right_eye.shape
                    self.right_eye_gray = self.face_gray[e_y: e_y + e_h, e_x: e_x + e_w]

                    _, self.right_eye_gray = cv2.threshold(self.right_eye_gray, int(self.var.get()), 200,
                                                           cv2.THRESH_BINARY)
                    self.right_eye_gray = cv2.erode(self.right_eye_gray, None, iterations=int(self.erode.get()))
                    self.right_eye_gray = cv2.dilate(self.right_eye_gray, None, iterations=int(self.dilate.get()))
                    self.right_eye_gray = cv2.medianBlur(self.right_eye_gray, int(self.blur.get()) - 1)
                    self.keypoints = detector.detect(self.right_eye_gray)

                    if self.raw_right and not self.contour_right:
                        self.right_eye_keypoints = cv2.drawKeypoints(self.right_eye, self.keypoints,
                                                                     self.right_eye_gray, (0, 0, 255),
                                                                     cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                        self.right_eye_image = Image.fromarray(self.right_eye_keypoints)
                        self.right_eye_image = ImageTk.PhotoImage(self.right_eye_image)

                        self.canvas3.create_image(0, 0, anchor=tk.NW, image=self.right_eye_image)

                    elif self.contour_right and not self.raw_right:
                        self.contours, _ = cv2.findContours(self.right_eye_gray, cv2.RETR_TREE,
                                                            cv2.CHAIN_APPROX_SIMPLE)
                        self.hull = []
                        # calculate points for each contour
                        for i in range(len(self.contours)):
                            # creating convex hull object for each contour
                            self.hull.append(cv2.convexHull(self.contours[i], False))

                        drawing = np.zeros((self.right_eye_gray.shape[0], self.right_eye_gray.shape[1], 3), np.uint8)
                        # draw contours and hull points
                        for i in range(len(self.contours)):
                            color_contours = (0, 255, 0)  # green - color for contours
                            color = (255, 0, 0)  # blue - color for convex hull
                            # draw ith contour
                            cv2.drawContours(drawing, self.contours, i, color_contours)
                            # draw ith convex hull object
                            cv2.drawContours(drawing, self.hull, i, color)

                        self.right_eye_image = Image.fromarray(drawing)
                        self.right_eye_image = ImageTk.PhotoImage(self.right_eye_image)

                        self.canvas3.create_image(0, 0, anchor=tk.NW, image=self.right_eye_image)

                    elif not self.raw_right or self.contour_right:

                        self.right_eye_keypoints = cv2.drawKeypoints(self.right_eye_gray, self.keypoints,
                                                                     self.right_eye_gray, (0, 0, 255),
                                                                     cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                        self.right_eye_image = Image.fromarray(self.right_eye_keypoints)
                        self.right_eye_image = ImageTk.PhotoImage(self.right_eye_image)

                        self.canvas3.create_image(0, 0, anchor=tk.NW, image=self.right_eye_image)

                elif int(self.face_img.shape[0] * 0.5) < self.eye_center < int(self.face_img.shape[1] * 0.9):
                    self.left_eye = self.face_img[e_y: e_y + e_h, e_x: e_x + e_w]
                    rows, cols, _ = self.left_eye.shape
                    self.left_eye_gray = self.face_gray[e_y: e_y + e_h, e_x: e_x + e_w]
                    _, self.eye_threshold = cv2.threshold(self.left_eye_gray, 50, 255, cv2.THRESH_BINARY)

                    _, self.left_eye_gray = cv2.threshold(self.left_eye_gray, int(self.var.get()), 200,
                                                          cv2.THRESH_BINARY)
                    self.left_eye_gray = cv2.erode(self.left_eye_gray, None, iterations=int(self.erode.get()))
                    self.left_eye_gray = cv2.dilate(self.left_eye_gray, None, iterations=int(self.dilate.get()))
                    self.left_eye_gray = cv2.medianBlur(self.left_eye_gray, 5)

                    if self.raw_left and not self.contour_left:
                        self.keypoints = detector.detect(self.left_eye_gray)
                        self.left_eye_keypoints = cv2.drawKeypoints(self.left_eye, self.keypoints, self.left_eye_gray,
                                                                    (0, 0, 255),
                                                                    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                        self.left_eye_image = Image.fromarray(self.left_eye_keypoints)
                        self.left_eye_image = ImageTk.PhotoImage(self.left_eye_image)

                        self.canvas2.create_image(0, 0, anchor=tk.NW, image=self.left_eye_image)

                    elif self.contour_left and not self.raw_left:
                        self.contours, _ = cv2.findContours(self.left_eye_gray, cv2.RETR_TREE,
                                                            cv2.CHAIN_APPROX_SIMPLE)
                        self.hull = []
                        # calculate points for each contour
                        for i in range(len(self.contours)):
                            # creating convex hull object for each contour
                            self.hull.append(cv2.convexHull(self.contours[i], False))

                        drawing_left = np.zeros((self.left_eye_gray.shape[0], self.left_eye_gray.shape[1], 3), np.uint8)
                        # draw contours and hull points
                        for i in range(len(self.contours)):
                            color_contours = (0, 255, 0)  # green - color for contours
                            color = (255, 0, 0)  # blue - color for convex hull
                            # draw ith contour
                            cv2.drawContours(drawing_left, self.contours, i, color_contours)
                            # draw ith convex hull object
                            cv2.drawContours(drawing_left, self.hull, i, color)

                        self.left_eye_image = Image.fromarray(drawing_left)
                        self.left_eye_image = ImageTk.PhotoImage(self.left_eye_image)

                        self.canvas2.create_image(0, 0, anchor=tk.NW, image=self.left_eye_image)

                    elif not self.raw_left and not self.contour_left:
                        self.keypoints = detector.detect(self.left_eye_gray)
                        self.left_eye_keypoints = cv2.drawKeypoints(self.left_eye_gray, self.keypoints,
                                                                    self.left_eye_gray,
                                                                    (0, 0, 255),
                                                                    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                        self.left_eye_image = Image.fromarray(self.left_eye_keypoints)
                        self.left_eye_image = ImageTk.PhotoImage(self.left_eye_image)

                        self.canvas2.create_image(0, 0, anchor=tk.NW, image=self.left_eye_image)

        self.window.after(self.interval, self.update_screen)


if __name__ == "__main__":
    window = tk.Tk()
    Window(window, cv2.VideoCapture(0))
    window.mainloop()

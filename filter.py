import cv2
import mediapipe as mp
import pyvirtualcam

cap = cv2.VideoCapture(2) # depends on input device, usually 0

def get_coords(point, image):
  x = int(image.shape[1] * point.x)
  y = int(image.shape[0] * point.y)
  return (x, y)

def add_halfstash(minX, maxX, minY, maxY, stash, image):
  stash = cv2.resize(stash, (maxX - minX, maxY - minY))
  alpha = stash[:, :, 3] / 255.0
  alpha_inv = 1 - alpha
  for c in range(0, 3):
    image[minY:maxY, minX:maxX, c] = (
      alpha * stash[:,:,c] + alpha_inv * image[minY:maxY, minX:maxX, c])

def draw(face, image):
  lip_mid = face[164]
  lip_left = face[322]
  lip_right = face[92]
  lip_upper = face[2]
  lip_lower = face[0]

  x_mid, _ = get_coords(lip_mid, image)
  x_left, _ = get_coords(lip_left, image)
  x_right, _ = get_coords(lip_right, image)

  _, y_upper = get_coords(lip_upper, image)
  _, y_lower = get_coords(lip_lower, image)

  stash = cv2.imread('halfstash.png', cv2.IMREAD_UNCHANGED)
  add_halfstash(x_mid, x_left, y_upper, y_lower, stash, image)

  stash = cv2.flip(stash, 1)
  add_halfstash(x_right, x_mid, y_upper, y_lower, stash, image)


with mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True) as face_mesh:

  _, frame1 = cap.read()
  with pyvirtualcam.Camera(
    width=frame1.shape[1], height=frame1.shape[0], fps=20) as cam:

    while cap.isOpened():
      success, image = cap.read()
      if not success: break

      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = face_mesh.process(image)

      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
          face = face_landmarks.landmark
          draw(face, image)

          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
          cam.send(image)
          cam.sleep_until_next_frame()

      cv2.imshow('Face', cv2.flip(image, 1)) # selfie flip
      if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(2) # depends on input device, usually 0

def get_coords(point, image):
  x = int(image.shape[1] * point.x)
  y = int(image.shape[0] * point.y)
  return (x, y)

def draw(face, image):
  lip_mid = face[164]
  x, y = get_coords(lip_mid, image)
  width, height = 200, 50

  stash = cv2.imread('mustache.png', cv2.IMREAD_UNCHANGED)
  stash = cv2.resize(stash, (width, height))

  minY = int(y - height/2)
  maxY = int(y + height/2)
  minX = int(x - width/2)
  maxX = int(x + width/2)

  alpha = stash[:, :, 3] / 255.0
  alpha_inv = 1 - alpha
  for c in range(0, 3):
    image[minY:maxY, minX:maxX, c] = (
      alpha * stash[:,:,c] + alpha_inv * image[minY:maxY, minX:maxX, c])


with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True) as face_mesh:
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

    cv2.imshow('Face', cv2.flip(image, 1)) # selfie flip
    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()
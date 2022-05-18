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
  cv2.circle(image, (x,y), 10, (0,0,255), 2)

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
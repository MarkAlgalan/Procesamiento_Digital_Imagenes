import cv2
import mediapipe as mp
import math
import tkinter as tk
from tkinter import filedialog, messagebox

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.7)

def calculate_angle(a, b, c):
    """Calcula el ángulo (en grados) entre los puntos a, b, c (b es el vértice)."""
    ba = [a.x - b.x, a.y - b.y]
    bc = [c.x - b.x, c.y - b.y]
    dot_product = ba[0]*bc[0] + ba[1]*bc[1]
    mag_ba = math.hypot(ba[0], ba[1])
    mag_bc = math.hypot(bc[0], bc[1])
    if mag_ba * mag_bc == 0:
        return 0
    angle = math.acos(dot_product / (mag_ba * mag_bc))
    return math.degrees(angle)

def process_image(image):
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.7)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Invertir etiqueta
            label = handedness.classification[0].label
            label = "Right" if label == "Left" else "Left"
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            lm = hand_landmarks.landmark
            fingers_up = 0
            h, w, _ = image.shape
            thumb_angle = calculate_angle(lm[2], lm[3], lm[4])
            thumb_points = [(2, (0,0,255)), (3, (0,255,0)), (4, (255,0,0))]
            for idx, color in thumb_points:
                cx, cy = int(lm[idx].x * w), int(lm[idx].y * h)
                cv2.circle(image, (cx, cy), 10, color, -1)
            if thumb_angle > 150:
                fingers_up += 1
            finger_defs = [
                (5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)
            ]
            for mcp, pip, tip in finger_defs:
                angle = calculate_angle(lm[mcp], lm[pip], lm[tip])
                points = [(mcp, (0,0,255)), (pip, (0,255,0)), (tip, (255,0,0))]
                for idx, color in points:
                    cx, cy = int(lm[idx].x * w), int(lm[idx].y * h)
                    cv2.circle(image, (cx, cy), 10, color, -1)
                if angle > 160:
                    fingers_up += 1
            cx = int(lm[8].x * w)
            cy = int(lm[8].y * h)
            cv2.putText(image, f"{label}: {fingers_up}", (cx, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            print(f"{label}: {fingers_up} dedos levantados")
    cv2.imshow("Dedos de ambas manos", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_camera():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
    cap = cv2.VideoCapture(0)
    while True:
        ret, image = cap.read()
        if not ret:
            break
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Invertir etiqueta
                label = handedness.classification[0].label
                label = "Right" if label == "Left" else "Left"
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                lm = hand_landmarks.landmark
                fingers_up = 0
                h, w, _ = image.shape
                thumb_angle = calculate_angle(lm[2], lm[3], lm[4])
                thumb_points = [(2, (0,0,255)), (3, (0,255,0)), (4, (255,0,0))]
                for idx, color in thumb_points:
                    cx, cy = int(lm[idx].x * w), int(lm[idx].y * h)
                    cv2.circle(image, (cx, cy), 10, color, -1)
                if thumb_angle > 160:
                    fingers_up += 1
                finger_defs = [
                    (5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)
                ]
                for mcp, pip, tip in finger_defs:
                    angle = calculate_angle(lm[mcp], lm[pip], lm[tip])
                    points = [(mcp, (0,0,255)), (pip, (0,255,0)), (tip, (255,0,0))]
                    for idx, color in points:
                        cx, cy = int(lm[idx].x * w), int(lm[idx].y * h)
                        cv2.circle(image, (cx, cy), 10, color, -1)
                    if angle > 160:
                        fingers_up += 1
                cx = int(lm[8].x * w)
                cy = int(lm[8].y * h)
                cv2.putText(image, f"{label}: {fingers_up}", (cx, cy - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                print(f"{label}: {fingers_up} dedos levantados")
        cv2.imshow("Dedos de ambas manos", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        image = cv2.imread(file_path)
        if image is None:
            messagebox.showerror("Error", "No se pudo cargar la imagen.")
            return
        process_image(image)

def main_interface():
    root = tk.Tk()
    root.title("Detección de Manos")
    root.geometry("300x150")
    btn_img = tk.Button(root, text="Subir Imagen", command=select_image, width=20, height=2)
    btn_img.pack(pady=10)
    btn_cam = tk.Button(root, text="Usar Cámara", command=process_camera, width=20, height=2)
    btn_cam.pack(pady=10)
    lbl = tk.Label(root, text="Presiona 'q' para salir de la cámara.")
    lbl.pack(pady=5)
    root.mainloop()

if __name__ == "__main__":
    main_interface()
import face_recognition
import cv2 as cv
import numpy as np

video_capture = cv.VideoCapture("./Videos/V5.mp4")

ibrahim_image = face_recognition.load_image_file("Hassan.jpeg")
ibrahim_image_encoding = face_recognition.face_encodings(ibrahim_image)[0]

mohasin_image = face_recognition.load_image_file("Mohasin.jpeg")
mohasin_image_encoding = face_recognition.face_encodings(mohasin_image)[0]

known_face_encodings = [
    ibrahim_image_encoding,
    mohasin_image_encoding
]
known_face_names = [
    "Hassan",
    "Mohasin"
]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    ret, frame = video_capture.read()
    small_frame = cv.resize(frame, (0, 0), fx=0.25, fy=0.25)
    cv.imshow('check', small_frame)
    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(
                known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv.rectangle(frame, (left, bottom - 35),
                     (right, bottom), (0, 0, 255), cv.FILLED)
        font = cv.FONT_HERSHEY_DUPLEX
        cv.putText(frame, name, (left + 6, bottom - 6),
                   font, 1.0, (255, 255, 255), 1)

    cv.imshow('Student Face Recognizer', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv.destroyAllWindows()

from matplotlib import pyplot as plt
import cv2

video = cv2.VideoCapture(0)

classificadorFace = cv2.CascadeClassifier('cascades\haarcascade_frontalface_default.xml')
classificadorOlho = cv2.CascadeClassifier('cascades\haarcascade_eye.xml')

while True:
    conectado, frame = video.read()

    frameCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    facesDetectadas = classificadorFace.detectMultiScale(frame, minSize=(20, 20), scaleFactor=1.3, minNeighbors=5)
    for (x, y, l, a) in facesDetectadas:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "People", (x, y - 5), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 0, 255), 2)

        regiao = frame[y:y + a, x:x + l]
        regiaoCinzaOlho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
        olhosDetecatados = classificadorOlho.detectMultiScale(regiaoCinzaOlho, minSize=(20, 20), scaleFactor=1.3, minNeighbors=5)
        for (ox, oy, ol, oa) in olhosDetecatados:
            cv2.rectangle(regiao, (ox, oy), (ox + ol, oy + oa), (255, 0, 0), 1)

    cv2.imshow('image', frame)

    if cv2.waitKey(1) == ord('q'):
        break
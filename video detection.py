import cv2

video = cv2.VideoCapture(0)

classificadorFace = cv2.CascadeClassifier('cascades\haarcascade_frontalface_default.xml')
classificadorOlho = cv2.CascadeClassifier('cascades\haarcascade_eye.xml')
classificadorSorriso = cv2.CascadeClassifier('cascades\haarcascade_smile.xml')

while True:
    i = 1
    conectado, frame = video.read()

    frameCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    facesDetectadas = classificadorFace.detectMultiScale(frame, minSize=(20, 20), scaleFactor=1.3, minNeighbors=5)
    for (x, y, l, a) in facesDetectadas:
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(frame, 'People [{}]'.format(i), (x, y - 15), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 0, 255), 2)

        regiao = frame[y:y + a, x:x + l]

        olhosDetecatados = classificadorOlho.detectMultiScale(regiao, minSize=(20, 20), scaleFactor=1.3, minNeighbors=15)

        i += 1

        for (ox, oy, ol, oa) in olhosDetecatados:
            cv2.putText(frame, 'Eyes [{}]'.format(len(olhosDetecatados)), (x, y - 35), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.rectangle(regiao, (ox, oy), (ox + ol, oy + oa), (255, 0, 0), 1)

        sorrisosDetectados = classificadorSorriso.detectMultiScale(regiao, minSize=(40, 40), scaleFactor=1.7, minNeighbors=25)

        for (ex, ey, el, ea) in sorrisosDetectados:
            cv2.putText(frame, 'Smile [{}]'.format(len(sorrisosDetectados)), (x, y - 55), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.rectangle(regiao, (ex, ey), (ex + el, ey + ea), (0, 255, 0), 1)

            cv2.imwrite(str(i)+'.jpg', frame)

    cv2.imshow('image', frame)

    if cv2.waitKey(1) == ord('q'):
        break
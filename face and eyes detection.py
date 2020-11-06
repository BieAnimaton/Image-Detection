import cv2
qntd_pessoas = 0
qntd_olhos = 0

classificadorFace = cv2.CascadeClassifier('cascades\haarcascade_frontalface_default.xml')
classificadorOlhos = cv2.CascadeClassifier('cascades\haarcascade_eye.xml')

imagem = cv2.imread('pessoas\\pessoas2.jpg')
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

facesDetectadas = classificadorFace.detectMultiScale(imagemCinza, minSize=(32, 32), scaleFactor=1.3, minNeighbors=5)
qntd_pessoas = len(facesDetectadas)
print(qntd_pessoas)

for (x, y, l, a) in facesDetectadas:
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(imagem, "People", (x, y - 5), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    imagem = cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 1)

    regiao = imagem[y:y + a, x:x + l]
    regiaoCinzaOlho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
    olhosDetectados = classificadorOlhos.detectMultiScale(regiaoCinzaOlho, scaleFactor=1.2, minNeighbors=3)

    qntd_olhos = len(olhosDetectados)
    print(qntd_olhos)

    for (ox, oy, ol, oa) in olhosDetectados:
        cv2.rectangle(regiao, (ox, oy), (ox + ol, oy + oa), (255, 0, 0), 1)

def line():
    print("-"*20)

line()
print("Faces detected: {}".format(len(facesDetectadas)))
line()
print("Eyes detected: {}".format(len(olhosDetectados)))
line()

cv2.imshow('Identificacao de olhos e rostos',imagem)
cv2.waitKey(0)
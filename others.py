import cv2
qntd_pessoas = 0
qntd_olhos = 0

classificadorBird = cv2.CascadeClassifier('cascades\\cars.xml')

imagem = cv2.imread('others\\car1.jpg')
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

passarosDetectadas = classificadorBird.detectMultiScale(imagemCinza, minSize=(20, 20))
qntd_pessoas = len(passarosDetectadas)
print(qntd_pessoas)

for (x, y, l, a) in passarosDetectadas:
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(imagem, "Car", (x, y - 5), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    imagem = cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)


cv2.imshow('Detection of others thins or objects',imagem)
cv2.waitKey(0)
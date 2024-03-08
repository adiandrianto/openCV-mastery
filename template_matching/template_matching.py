import cv2

#load input and template
img = cv2.imread('template_matching\pepsi1.png')
template = cv2.imread('template_matching\pepsi_template.png')

#resized
img_resized = cv2.resize(img, (500,500))
template_resized = cv2.resize(template, (100,100))

#turn grayscale
imgGray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
templateGray = cv2.cvtColor(template_resized, cv2.COLOR_BGR2GRAY)

cv2.imshow("image", imgGray)
cv2.imshow("template", templateGray)

#perform template matching
result = cv2.matchTemplate(imgGray, templateGray, cv2.TM_CCOEFF_NORMED)
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)

(startX, startY) = maxLoc
endX = startX + template_resized.shape[1]
endY = startY + template_resized.shape[0]

#draw bbox
cv2.rectangle(img_resized, (startX, startY), (endX, endY), (255,0,0), 2)
cv2.imshow("output", img_resized)
cv2.waitKey(0)


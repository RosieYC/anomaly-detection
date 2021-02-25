def change_color(img_path):
  img = cv2.imread(img_path)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  return img, gray
def FFT2(img_path):
  img, gray = change_color(img_path)
  rows, cols = gray.shape
  mask = np.ones(gray.shape, np.uint8)
  mask[int(rows/2-10):int(rows/2+10), int(((cols-1)/2)-10):int(((cols-1)/2)+10)] = 0
  f1 = np.fft.fft2(gray)
  f1s = np.fft.fftshift(f1)
  f1s = f1s*mask
  f2s = np.fft.ifftshift(f1s)
  img_new = np.abs(np.fft.ifft2(f2s))
  img_new = (img_new-np.amin(img_new))/(np.amax(img_new)-np.amin(img_new))
  return img_new
def compare(img1, img2):
  conf=0.8
  diff = np.abs(img1-img2)
  result = np.array(diff<(1-conf)*np.max(diff))+0
  similar = result.reshape(1, -1)[0].tolist()
  similar = sum(similar)/len(similar)
  return diff, similar
def resize_img(img1, img2):
  img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
  return img2
def M5(img1_path, img2_path):
  img1 = FFT2(img1_path)
  img2 = FFT2(img2_path)
  img2 = resize_img(img1, img2)
  diff, similar = compare(img1, img2)
  reutrn diff, similar
def threshold_method(img, gate, max_value, method):
	## methods 
	## - cv2.THRESH_BINARY
	## - cv2.THRESH_BINARY_INV
  ## - cv2.THRESH_TRUNC
  ## - cv2.THRESH_TOZERO
  ## - cv2.THRESH_TOZERO_INV
  _, threshold = cv2.threshold(img, gate, max_value, method)
  return threshold
def find_contours_method(img, mode, method):
  ## modes
  ## - cv2.RETR_EXTERNAL
  ## - cv2.RETR_LIST
  ## - cv2.RETR_CCOMP
  ## - cv2.RETR_TREE
  ## methods
  ## - cv2.CHAIN_APPROX_NONE
  ## - cv2.CHAIN_APPROX_SIMPLE
  ## - cv2.CHAIN_APPROX_TC89_L1
  ## - cv2.CHAIN_APPROX_TC89_KCOS
  _, contours, hierarchy = cv2.findContours(img, mode, method)
  
  if len(contours)<=0:
    return 1
  contour_count = []
  for contour in contours:
    areas = cv2.contourArea(contour)
    contour_count.append(areas)
  N = 1
  res = sorted(range(len(contour_count)), key = lambda sub: contour_count[sub])[-N:] 
  area = cv2.contourArea(contours[res[0]])
  x,y,w,h = cv2.boundingRect(contours[res[0]])
  img = img[y:y+h, x:x+w]
  return img
def img_resize(img, x, y, method):
  ## method 
  ## - cv2.INTER_NEAREST
  ## - cv2.INTER_LINEAR(default)
  ## - cv2.INTER_AREA
  ## - cv2.INTER_CUBIC
  ## - cv2.INTER_LANCZOS4
  img = cv2.resize(img, (x, y), interpolation=method)
  return img
def sobel(img, depth, xdx, xdy, ydx, ydy, ksize=3, scale=1, delta=0, borderType=BORDER_DEFAULT, alpha, beta, gamma):
  ## depth
  ## - cv2.CV_8U
  ## - CV_16S
  ## - CV_16U
  ## - CV_32F
  ## - CV_64F
  
  sobel_x = cv2.Sobel(img, depth, xdx, xdy, ksize, scale, delta, borderType) # xdx = 1, xdy = 0
  sobel_y = cv2.Sobel(img, depth, ydx, ydy, ksize, scale, delta, borderType) # ydx = 0, ydy = 1

  sobel_absX = cv2.convertScaleAbs(sobel_x)# convert to uint8
  sobel_absY = cv2.convertScaleAbs(sobel_y)# convert to uint8
  
  result = cv2.addWeighted(sobel_absX, alpha, sobel_absY, beta, gamma)
  return result
def scharr(img, depth, xdx, xdy, ydx, ydy, scale=1, delta=0, borderType=BORDER_DEFAULT, alpha, beta, gamma):
  scharr_x = cv2.Sobel(img, depth, xdx, xdy, -1, scale, delta, borderType) # xdx = 1, xdy = 0
  scharr_y = cv2.Sobel(img, depth, ydx, ydy, -1, scale, delta, borderType) # ydx = 0, ydy = 1
  
  scharr_absX = cv2.convertScaleAbs(scharr_x)# convert to uint8
  scharr_absY = cv2.convertScaleAbs(scharr_y)# convert to uint8
  
  result = cv2.addWeighted(scharr_absX, alpha, scharr_absY, beta, gamma)
  return result
def Laplacian(img, depth):
  ## depth
  ## - cv2.CV_8U
  ## - CV_16S
  ## - CV_16U
  ## - CV_32F
  ## - CV_64F
  laplacian = cv2.Laplacian(img, depth, ksize=3)
  result = cv2.convertScaleAbs(laplacian)
  return result
def eroded(img, method, size=(3,3)):
  ## method
  ## - cv2.MORPH_ELLIPSE
  ## - cv2.MORPH_CROSS
  ## - cv2.MORPH_RECT
  kernel = cv2.getStructuringElement(method, ksize=size)
  erode = cv2.erode(img, kernel)
  return erode
def dilated(img, method, size=(3,3)):
  ## method
  ## - cv2.MORPH_ELLIPSE
  ## - cv2.MORPH_CROSS
  ## - cv2.MORPH_RECT
  kernel = cv2.getStructuringElement(method, ksize=size)
  dilat = cv2.dilate(img, kernel)
  return dilat
def closed(img, kernel_method, size=(3,3)):
  ## dilated -> erode
  ## method
  ## - cv2.MORPH_ELLIPSE
  ## - cv2.MORPH_CROSS
  ## - cv2.MORPH_RECT
  kernel = cv2.getStructuringElement(kernel_method, ksize=size)
  result = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
  return result
def opened(img, kernel_method, size=(3,3)):
  ## erode -> dilated
  ## method
  ## - cv2.MORPH_ELLIPSE
  ## - cv2.MORPH_CROSS
  ## - cv2.MORPH_RECT
  kernel = cv2.getStructuringElement(kernel_method, ksize=size)
  result = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
  return result
def equlizehist_method(img):
  equ = cv2.equalizeHist(img)
  return equ

  
  

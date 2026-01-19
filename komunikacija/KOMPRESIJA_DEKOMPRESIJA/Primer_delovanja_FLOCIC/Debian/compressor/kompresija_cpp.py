import cv2
import numpy as np
import compressor

img = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)
assert img is not None

compressed = compressor.compress(img)

img2 = compressor.decompress(bytes(compressed))
img2 = np.asarray(img2, dtype=np.uint8)

print("Decompressed:", img2.shape, img2.dtype)

cv2.imshow("orig", img)
cv2.imshow("decompressed", img2)
cv2.waitKey(0)

import cv2


class Filter():

    def __init__(self) -> None:
        pass

    def blur(self, data, ksize=(3, 3)):
        """
        均值滤波
        """
        return cv2.blur(data, ksize)

    def gassianBlur(self, data, ksize=(3, 3), sigma=1):
        """
        高斯滤波
        """
        return cv2.GaussianBlur(data, ksize, sigma)

    def medianBlur(self, data, ksize=(3, 3)):
        """
        中值滤波
        """
        return cv2.medianBlur(data, ksize)

    def bilateralFilter(self, data, d=9, sigmaColor=100, sigmaSpace=100):
        """
        双边滤波
        """
        return cv2.bilateralFilter(data, d, sigmaColor, sigmaSpace)

import numpy as np
from PIL import Image
from sobel import Sobel
from convolve import Convolve

class Canny:
    def __init__(self, image, low_threshold=0.1, high_threshold=0.7):
        img = image.convert("L")  # Ensure grayscale
        img = self.gauss_convolve(img, sigma=1.0)  # Apply Gaussian filter
        
        sobel_obj = Sobel(img)
        sobel_obj.Sobel(img)
        
        

        self.image = np.array(img, dtype=np.float32)
        self.width, self.height = img.size
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

        self.edge_strength = np.array(sobel_obj.get_edge_strength(), dtype=np.float32)
        self.edge_direction = np.array(sobel_obj.get_edge_orientation(), dtype=np.float32)
        
        
        # edge_strength_image = Image.open("edge_strength.png")
        # self.edge_strength = np.array(edge_strength_image, dtype=np.float32)

        # # Convert the edge direction image to a numpy array and extract the hue channel
        # edge_direction_image = Image.open("edge_orientation.png").convert("HSV")
        # edge_direction_array = np.array(edge_direction_image, dtype=np.float32)
        # self.edge_direction = edge_direction_array[:, :, 0]

        #self.edge_strength.show()
        self.nms_image = self.non_max_suppression()
        self.double_threshold_array, self.weak, self.strong = self.double_threshold(self.nms_image, self.low_threshold, self.high_threshold)
        self.canny_image = self.hysteresis(self.double_threshold_array, self.weak, self.strong)

    def gauss_convolve(self, image, sigma=1.0):
        convolve_obj = Convolve()

        #img.show()
        gauss_filter = convolve_obj.gauss_1d(sigma)  # Apply Gaussian filter
        image = convolve_obj.convolve_horizontal(image, gauss_filter)
        image = convolve_obj.convolve_vertical(image, gauss_filter)
        return image

    def non_max_suppression(self):
        # Initializing variables D is the gradient direction image, Z is the non-max suppression array
        # and img is the edge strength image
        N, M = self.edge_strength.shape
        D = self.edge_direction.copy()
        Z = np.zeros((N,M), dtype=np.float32)
        strength = self.edge_strength.copy()

        # Calculate the gradient direction from the edge strength image into degrees
        # For each pixel in the image, calculate the gradient direction and store it in D
        D = D * 180 / np.pi
        D = np.where(D < 0, D + 180, D)  # Ensure all angles are positive

        # for j in range(N):
        #     for i in range(M):
        #         # Calculate the gradient direction in degrees
        #         a = self.edge_strength.getpixel((i,j)) * 180 / np.pi
        #         if a < 0:
        #             D.putpixel((i,j), a + 180)
        #         else:
        #             D.putpixel((i, j), a)
        
        # Loop through the image and apply non-max suppression by checking the gradient direction and grouping
        # the pixels into 4 groups: 0, 45, 90, and 135 degrees
        # The pixels are then compared to their neighbors in the gradient direction and the pixel with the highest value is kept
        for j in range(1,M-1):
            for i in range(1,N-1):
                try:
                    q = 255
                    r = 255
                    angle = D[i,j]  # Get the gradient direction
                    magnitude = strength[i,j]  # Get the edge strength
                    #angle 0
                    if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                        q = strength[i,j+1]
                        r = strength[i,j-1]
                    #angle 45
                    elif (22.5 <= angle < 67.5):
                        q = strength[i+1,j-1]
                        r = strength[i-1,j+1]
                    #angle 90
                    elif (67.5 <= angle < 112.5):
                        q = strength[i+1,j]
                        r = strength[i-1,j]
                    #angle 135
                    elif (112.5 <= angle < 157.5):
                        q = strength[i-1,j-1]
                        r = strength[i+1,j+1]

                    if (magnitude >= q) and (magnitude >= r):
                        Z[i,j] = magnitude
                    else:
                        Z[i,j] = 0

                except IndexError as e:
                    pass
        
        return Z

    def double_threshold(self, nms_array, lowThresholdRatio=0.1, highThresholdRatio=0.5):
        # Initialize the thresholds, nms_array is the non-max suppression array
        # lowThresholdRatio and highThresholdRatio are the ratios of the maximum value in the nms_array
        # res is the double threshold array
        highThreshold = nms_array.max() * highThresholdRatio
        #print("highThreshold: ", highThreshold)
        lowThreshold = highThreshold * lowThresholdRatio
        
        # M, N = nms_array.shape
        # res = np.zeros((M,N), dtype=np.int32)
        
        # Set the strong and weak pixel values
        weak = 25
        strong = 255
        
        res = np.where(nms_array >= highThreshold, strong, 0)  # Set strong pixels
        res = np.where((nms_array < highThreshold) & (nms_array >= lowThreshold), weak, res)  # Set weak pixels

        # Set the pixel values in the double threshold array based on the thresholds
        # for j in range(M):
        #     for i in range(N):
        #         if nms_array[j,i] >= highThreshold:
        #             res[j,i] = strong
        #         elif nms_array[j,i] < lowThreshold:
        #             res[j,i] = 0
        #         else:
        #             res[j,i] = weak
        # strong_i, strong_j = np.where(img >= highThreshold)
        # zeros_i, zeros_j = np.where(img < lowThreshold)
        
        # weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
        
        # res[strong_i, strong_j] = strong
        # res[weak_i, weak_j] = weak
        
        return (res, weak, strong)
    
    def hysteresis(self, res, weak, strong=255):
        # Initialize the hysteresis function, res is the double threshold array
        # weak and strong are the pixel values for weak and strong pixels
        N, M = res.shape
        img = Image.new("L", (M, N), 0)

        # Loop through the image and set the pixel values based on the weak and strong pixels
        # The weak pixels are set to 255 if they are connected to a strong pixel (8-connectivity)
        # Otherwise, they are set to 0
        for i in range(1, N-1):
            for j in range(1, M-1):
                if (res[i][j] == weak):
                    try:
                        if ((res[i+1][j-1] == strong) or (res[i+1][j] == strong) or (res[i+1][j+1] == strong)
                            or (res[i][j-1] == strong) or (res[i][j+1] == strong)
                            or (res[i-1][j-1] == strong) or (res[i-1][j] == strong) or (res[i-1][j+1] == strong)):
                            img.putpixel((j, i), strong)
                            res[i][j] = strong
                        else:
                            img.putpixel((j, i), 0)
                    except IndexError as e:
                        pass
                elif (res[i][j] == strong):
                    img.putpixel((j, i), 255)
        return img
    def get_canny_image(self):
        return self.canny_image

if __name__ == "__main__":
    image_path = "flower_frames\\43.png"
    im = Image.open(image_path).convert('L')
    canny = Canny(im)
    edges = canny.get_canny_image()
    edges.save("result_images\\canny_edge.png")

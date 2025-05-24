import numpy as np
from PIL import Image
from convolve import Convolve


class Sobel:
    # Initialize global variables
    
    def __init__(self, image):
        """
        Initializes the Sobel class with the image path.

        Parameters:
            image_path (str): Path to the input image.
        """
        # Load image
        self.image = image.convert("L")  # Convert to grayscale
        # Convert image to numpy array
        self.i_x = Image.new(mode="F", size=self.image.size)
        self.i_y = Image.new(mode="F", size=self.image.size)
        self.edge_orientation_image = Image.new(mode="HSV", size=self.image.size, color=(0, 0, 0))
        self.edge_orientation_array = np.zeros((self.image.size[1], self.image.size[0]), dtype=np.float32)
        self.edge_strength_image = Image.new(mode="L", size=self.image.size) 
        self.cartoon_image = Image.new(mode="L", size=self.image.size)
    
    
    
    def Sobel(self, image):
        """
        Applies the Sobel operator to an image to detect edges.

        Parameters:
            image (numpy.ndarray): Input image.

        Returns:
            numpy.ndarray: Image with Sobel operator applied.
        """
        # Convert to grayscale
        image = image.convert("L")  
        # Create Convolve object
        convolve_obj = Convolve()

        # Define Sobel kernels
        sobel_x_vertical = sobel_y_horizontal = np.array([3, 10, 3])
        sobel_x_horizontal = sobel_y_vertical = np.array([-1, 0, 1])

        # Convolve with Sobel kernels
        i_x = convolve_obj.convolve_horizontal(image, sobel_x_horizontal)
        i_x = convolve_obj.convolve_vertical(i_x, sobel_x_vertical)
        
        i_y = convolve_obj.convolve_vertical(image, sobel_y_vertical)
        i_y = convolve_obj.convolve_horizontal(i_y, sobel_y_horizontal)

        # Convert to Image objects
        self.i_x = i_x
        self.i_y = i_y
        self.edge_strength_image = self.create_edge_strength_image(image, i_x, i_y)
        self.edge_orientation_image, self.edge_orientation_array = self.create_edge_orientation_image(image, i_x, i_y)
        return self.combine_images(image, i_x, i_y)
    
    def combine_images(self, image, i_x, i_y):
        #Merge the two images
        image_combined = Image.new(mode="F", size=image.size)
        for v in range(image_combined.size[1]):
            for u in range(image_combined.size[0]):
                # Get pixel values from both images
                pixel_x = i_x.getpixel((u, v))
                pixel_y = i_y.getpixel((u, v))

                # Set pixel value in combined image
                image_combined.putpixel((u, v), (pixel_x + pixel_y) / 2)
        return image_combined

    def create_edge_orientation_image(self, image, i_x, i_y):
         # Calculate magnitude of gradients
        edge_orientation_image = Image.new(mode="HSV", size=image.size)
        edge_orientation_array = np.zeros((image.size[1], image.size[0]), dtype=np.float32)
        for v in range(edge_orientation_image.size[1]):
            for u in range(edge_orientation_image.size[0]):
                # Get pixel values from both images
                pixel_x = i_x.getpixel((u, v))
                pixel_y = i_y.getpixel((u, v))

                # Calculate orientation of gradients
                if pixel_x == 0:
                    orientation = 0
                else:
                    orientation = np.arctan(pixel_y / pixel_x)
                # Convert orientation to degrees
                orientation = np.degrees(orientation) % 360
                # Normalize orientation to range [0, 255]
                orientation = int(orientation / 360 * 255)
                # Set pixel value in edge orientation image
                edge_orientation_image.putpixel((u, v), (int(orientation), 100, 100))
                edge_orientation_array[v, u] = int(orientation)  # Set pixel value in edge orientation image
        return edge_orientation_image, edge_orientation_array
    
    def create_edge_strength_image(self, image, i_x, i_y):
        #Create Edge Strength Image
        edge_strength_image = Image.new(mode="L", size=image.size)
        # Calculate magnitude of gradients
        for v in range(edge_strength_image.size[1]):
            for u in range(edge_strength_image.size[0]):
                # Get pixel values from both images
                pixel_x = i_x.getpixel((u, v))
                pixel_y = i_y.getpixel((u, v))

                # Calculate magnitude of gradients
                magnitude = np.sqrt(pixel_x**2 + pixel_y**2)
                # Normalize magnitude to range [0, 255]
                magnitude = int(magnitude / np.sqrt(255**2 + 255**2) * 255)
                # Set pixel value in edge strength image
                edge_strength_image.putpixel((u, v), magnitude)
        return edge_strength_image
    
    def get_i_x(self):
        return self.i_x

    def get_i_y(self):
        return self.i_y
    
    def get_edge_strength(self):
        self.edge_strength_image.show()
        return self.edge_strength_image
    
    def get_edge_orientation(self):
        return self.edge_orientation_array
    
    def get_edge_orientation_image(self):
        return self.edge_orientation_image

def main():
        # Example usage
        image_path = "image.png"
        image = Image.open(image_path).convert("L")
        sobel_obj = Sobel(image)
        sobel_image = sobel_obj.Sobel(image)
        sobel_image = sobel_image.convert("L")
        sobel_image.save("sobel_output.png")

        # Uncomment below to save intermediate images
        # i_x = sobel_obj.get_i_x()
        # i_x = i_x.convert("L")
        # i_x.save("i_x.png")
        
        # i_y = sobel_obj.get_i_y()
        # i_y = i_y.convert("L")
        # i_y.save("i_y.png")
        
        # edge_strength_image = sobel_obj.get_edge_strength()
        # edge_strength_image = edge_strength_image.convert("L")
        # edge_strength_image.save("edge_strength.png")
        
        # edge_orientation_image = sobel_obj.get_edge_orientation()
        # edge_orientation_image = edge_orientation_image.convert("RGB")
        # edge_orientation_image.save("edge_orientation.png")
        
        # cartoon_image = sobel_obj.cartoon_edge()
        # cartoon_image.save("cartoon_output.png")
    
if __name__ == "__main__":
    main()

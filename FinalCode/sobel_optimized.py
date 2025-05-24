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
        # Convert gradient images to NumPy arrays
        arr_x = np.array(i_x, dtype=np.float32)
        arr_y = np.array(i_y, dtype=np.float32)

        # Avoid division by zero: use np.arctan2 which handles this gracefully
        orientation_rad = np.arctan2(arr_y, arr_x)  # Returns angle in radians [-π, π]

        # Convert to degrees and map to [0, 360)
        orientation_deg = (np.degrees(orientation_rad) + 360) % 360

        # Normalize to [0, 255]
        orientation_norm = (orientation_deg / 360 * 255).astype(np.uint8)

        # Save to array
        edge_orientation_array = orientation_norm.copy()

        # Create HSV image: (H, S, V) -> (orientation, 100, 100)
        h = orientation_norm
        s = np.full_like(h, 100, dtype=np.uint8)
        v = np.full_like(h, 100, dtype=np.uint8)
        hsv_image_array = np.stack((h, s, v), axis=-1)

        # Convert to PIL image
        edge_orientation_image = Image.fromarray(hsv_image_array, mode="HSV")

        return edge_orientation_image, edge_orientation_array
    
    def create_edge_strength_image(self, image, i_x, i_y):
        arr_x = np.array(i_x, dtype=np.float32)
        arr_y = np.array(i_y, dtype=np.float32)

        # Compute the gradient magnitude
        magnitude = np.sqrt(arr_x**2 + arr_y**2)

        # Normalize to [0, 255]
        magnitude = (magnitude / np.sqrt(255**2 + 255**2)) * 255
        magnitude = magnitude.astype(np.uint8)

        # Create new edge strength image from the array
        return magnitude
    
    def get_i_x(self):
        return self.i_x

    def get_i_y(self):
        return self.i_y
    
    def get_edge_strength(self):
        # self.edge_strength_image.show()
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
        sobel_image.save("sobel_optimized_output.png")

       
    
if __name__ == "__main__":
    main()

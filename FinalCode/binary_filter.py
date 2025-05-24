from PIL import Image
import random
from convolve import Convolve
from sobel_optimized import Sobel
from canny_edge_detector import Canny
import os
def main():
    # Load an image (convert to grayscale for consistency)
    image_path = "flower_frames\\43.png"
    im = Image.open(image_path).convert('L')
    convolve_obj = Convolve()
    sobel_obj = Sobel(im)

    # gaussian_filter = convolve_obj.gauss_1d(1.0)
    # gaussian_image = convolve_obj.convolve_horizontal(im, gaussian_filter)
    # gaussian_image = convolve_obj.convolve_vertical(gaussian_image, gaussian_filter)
    # # gaussian_image.save("result_images\\gaussian_image.png")

    # otsu_image = threshold_otsu(gaussian_image)
    # otsu_image.save("result_images\\otsu_thresholded.png")
    # global_threshold_image_100 = threshold_global(gaussian_image, 100)
    # global_threshold_image_100.save("result_images\\global_thresholded_100.png")
    # global_threshold_image_150 = threshold_global(gaussian_image, 150)
    # global_threshold_image_150.save("result_images\\global_thresholded_150.png")
    # global_threshold_image_200 = threshold_global(gaussian_image, 200)
    # global_threshold_image_200.save("result_images\\global_thresholded_200.png")

    
    # sobel_image = sobel_obj.Sobel(im)
    # sobel_image = sobel_image.convert("L")  # Convert to grayscale
    
    # combined_image_otsu_sobel = combine_images(otsu_image, sobel_image)
    # combined_image_otsu_sobel.save("result_images\\combined_sobel.png")
    # combined_image_global_sobel = combine_images(global_threshold_image_200, sobel_image)
    # combined_image_global_sobel.save("result_images\\combined_sobel_100.png")

    # # sobel_image.save("result_images\\sobel_image.png")
    # # edge_image = sobel_obj.get_edge_strength()
    # # edge_image.save("result_images\\edge_strength.png")
    # # direction_image = sobel_obj.get_edge_orientation_image()
    # # direction_image = direction_image.convert("RGB")  # Convert to RGB for saving
    # # direction_image.save("result_images\\edge_orientation.png")

    # canny_obj = Canny(im, low_threshold=0.1, high_threshold=0.7)
    # canny_image = canny_obj.get_canny_image()
    # combined_image_otsu_canny = combine_images(otsu_image, canny_image)
    # combined_image_otsu_canny.save("result_images\\combined_canny.png")
    # combined_image_global_canny = combine_images(global_threshold_image_200, canny_image)
    # combined_image_global_canny.save("result_images\\combined_canny_100.png")
    # image = Image.open("montage_image\\1.png")
    # image.show()
    folder_path = "og_images"

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            im = Image.open(file_path).convert('L')
            # im.show()

            gaussian_filter = convolve_obj.gauss_1d(1.0)
            gaussian_image = convolve_obj.convolve_horizontal(im, gaussian_filter)
            gaussian_image = convolve_obj.convolve_vertical(gaussian_image, gaussian_filter)

            thresholded_image = threshold_otsu(gaussian_image)
            
            canny_obj = Canny(im, low_threshold=0.1, high_threshold=0.7)
            canny_image = canny_obj.get_canny_image()
            canny_image = canny_image.convert("L")  # Convert to grayscale

            combined_image = combine_images(thresholded_image, canny_image)
            combined_image.save("final_montage\\" + filename)

def combine_images(thresh_im, edge_im):
    """Combines the thresholded image with the edge image."""
    M, N = thresh_im.size
    combined_image = Image.new('1', (M, N), 1)  # Start with a white image

    for v in range(N):
        for u in range(M):
            thresh_value = thresh_im.getpixel((u, v))
            edge_value = edge_im.getpixel((u, v))
            if edge_value > 0:  # If is an edge edge image
                if thresh_value == 0:
                    combined_image.putpixel((u, v), 1)  # Set to white
                else:
                    combined_image.putpixel((u, v), 0)  # Set to black
            else:
                combined_image.putpixel((u, v), thresh_value)  # Keep the thresholded value
    return combined_image

def threshold_global(im, threshold):
    """Applies a binary threshold to the image."""
    M, N = im.size
    binary_image = Image.new('1', (M, N), 1)  # Start with a white image

    for v in range(N):
        for u in range(M):
            pixel_value = im.getpixel((u, v))
            if pixel_value < threshold:
                binary_image.putpixel((u, v), 0)  # Set to black
            # else:  
            #     binary_image.putpixel((u, v), 1) # Set to white

    return binary_image

def threshold_otsu(im):
    """Applies Otsu's thresholding method to the image."""
    im = im.convert('L')  # Ensure the image is in grayscale
    histogram = im.histogram()
    total_pixels = sum(histogram)

    # Step 1: Compute total sum (mean * pixels)
    sum_total = sum(i * histogram[i] for i in range(256))

    sum_background = 0
    weight_background = 0
    max_variance = 0
    threshold = 0

    for i in range(256):
        weight_background += histogram[i]
        if weight_background == 0:
            continue

        weight_foreground = total_pixels - weight_background
        if weight_foreground == 0:
            break

        sum_background += i * histogram[i]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground

        between_class_variance = (weight_background * weight_foreground) * ((mean_background - mean_foreground) ** 2)

        if between_class_variance > max_variance:
            max_variance = between_class_variance
            threshold = i

    return threshold_global(im, threshold)
     

if __name__ == "__main__":
    main()
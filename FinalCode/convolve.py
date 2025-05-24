from PIL import Image
import random
import math


class Convolve:
    def __init__(self):
        pass
    def convolve_2d(self, im, H):
        """Applies a 2D convolution with mirrored edge handling."""
        M, N = im.size
        m, n = len(H), len(H[0])
        pad_h, pad_w = m // 2, n // 2
        
        # Normalize filter
        filter_sum = 0
        m, n = len(H), len(H[0])
        for i in range(m):
            for j in range(n):
                filter_sum += H[i][j]
        
        # Convolution operation
        convolved_image = Image.new('L', (M, N))
        for v in range(N):
            for u in range(M):
                if u < pad_w or u >= M - pad_w or v < pad_h or v >= N - pad_h:
                    x, y = u, v
                    if u < pad_w:
                        x = u + pad_w
                    if v < pad_h:
                        y = v + pad_h
                    if u >= M - pad_w:
                        x = u - pad_w
                    if v >= N - pad_h:
                        y = v - pad_h
                    convolved_image.putpixel((u, v), im.getpixel((x, y)))
                    continue
                sum_value = 0
                x = -1
                y = -1
                for j in range(v - pad_h, v + pad_h + 1):
                    y += 1
                    for i in range(u - pad_w, u + pad_w + 1):
                        x += 1                 
                        sum_value += im.getpixel((i, j)) * H[x][y]
                    x = -1
                pixel_value = round(sum_value / filter_sum)
                convolved_image.putpixel((u,v), pixel_value)
        
        return convolved_image

    def box(self, w, h):
        """Creates a box filter ensuring w and h are odd."""
        if w % 2 == 0:
            w += 1
        if h % 2 == 0:
            h += 1
        box_filter = [[1 for i in range(w)] for j in range(h)]
        return box_filter

    def gauss_1d(self, sigma):
        """Creates a 1D Gaussian filter based on sigma."""
        width = int(5 * sigma) | 1  # Ensure odd width
        half = width // 2
        
        filter_1d = [math.exp(- (x ** 2) / (2 * sigma ** 2)) for x in range(-half, half + 1)]
        
        # Normalize filter
        total = sum(filter_1d)
        for i in range(len(filter_1d)):
            filter_1d[i] /= total
        return filter_1d

    def convolve_horizontal(self, image, filter_1d):
        """Applies 1D convolution horizontally."""
        pixels = image.load()
        w, h = image.size
        half = len(filter_1d) // 2
        output = Image.new("L", (w, h))
        output_pixels = output.load()
        
        for i in range(h):
            for j in range(w):
                sum_value = 0
                for k in range(-half, half + 1):
                    idx = j + k
                    idx = max(0, min(w - 1, idx))  # Mirror edges
                    sum_value += pixels[idx, i] * filter_1d[k + half]
                output_pixels[j, i] = int(sum_value)
        
        return output

    def convolve_vertical(self, image, filter_1d):
        """Applies 1D convolution vertically."""
        pixels = image.load()
        w, h = image.size
        half = len(filter_1d) // 2
        output = Image.new("L", (w, h))
        output_pixels = output.load()
        
        for i in range(h):
            for j in range(w):
                sum_value = 0
                for k in range(-half, half + 1):
                    idx = i + k
                    idx = max(0, min(h - 1, idx))  # Mirror edges
                    sum_value += pixels[j, idx] * filter_1d[k + half]
                output_pixels[j, i] = int(sum_value)
        
        return output
    


from PIL import Image, ImageOps
import numpy as np
from skimage.measure import shannon_entropy
from skimage.util import view_as_blocks

secret_img = Image.open('secret_image.png')
def encode_image(cover_path, secret_path, output_path, start_x, start_y, lsb_layers=4):
    cover_img = Image.open(cover_path).convert('RGB')
    secret_img = Image.open(secret_path).convert('RGB')


    if secret_img.width > cover_img.width - start_x or secret_img.height > cover_img.height - start_y:
        raise ValueError("Secret image is too large for the cover image at the given position.")

    # Histogram Equalization
    cover_img = ImageOps.equalize(cover_img)


    mask = 255 - (1 << lsb_layers) + 1
    shift = 8 - lsb_layers

    for y in range(secret_img.height):
        for x in range(secret_img.width):
            cover_pixel = list(cover_img.getpixel((x + start_x, y + start_y)))
            secret_pixel = list(secret_img.getpixel((x, y)))

            for i in range(3):  # RGB channels
                cover_pixel[i] = (cover_pixel[i] & mask) | (secret_pixel[i] >> shift)

            cover_img.putpixel((x + start_x, y + start_y), tuple(cover_pixel))

    cover_img.save(output_path)

def detect_modified_region_entropy(image, block_size=8, threshold=1.0):
    grayscale = image.convert('L')
    array = np.array(grayscale)


    height, width = array.shape
    cropped_height = height - height % block_size
    cropped_width = width - width % block_size
    array = array[:cropped_height, :cropped_width]

    blocks = view_as_blocks(array, block_shape=(block_size, block_size))


    entropy_map = np.apply_along_axis(shannon_entropy, 2, blocks.reshape(-1, block_size, block_size))


    entropy_map = entropy_map.flatten()

    print("Flattened entropy_map size:", entropy_map.size)

    if entropy_map.max() < threshold:
        return None


    target_shape = (blocks.shape[0], blocks.shape[1])
    reshaped_entropy_map = entropy_map[:target_shape[0] * target_shape[1]].reshape(target_shape)

    print("Reshaped entropy_map shape:", reshaped_entropy_map.shape)


    max_y, max_x = np.unravel_index(np.argmax(reshaped_entropy_map), reshaped_entropy_map.shape)
    return max_x * block_size, max_y * block_size


def decode_image(encoded_path, output_path, secret_width, secret_height, lsb_layers=1):
    encoded_img = Image.open(encoded_path)
    region_start = detect_modified_region_entropy(encoded_img)

    if region_start is None:
        raise ValueError("No modified region detected.")

    start_x, start_y = region_start
    decoded_img = Image.new('RGB', (secret_width, secret_height))

    mask = (1 << lsb_layers) - 1
    shift = 8 - lsb_layers

    for y in range(secret_height):
        for x in range(secret_width):
            if start_x + x < encoded_img.width and start_y + y < encoded_img.height:
                pixel = encoded_img.getpixel((start_x + x, start_y + y))
                decoded_pixel = [(channel & mask) << shift for channel in pixel]

                decoded_img.putpixel((x, y), tuple(decoded_pixel))

    decoded_img.save(output_path)


encode_image('cover_image.png', 'secret_image.png', 'encoded_image.png', 50, 50, lsb_layers=2)
decode_image('encoded_image.png', 'decoded_image.png', secret_img.width, secret_img.height, lsb_layers=2)

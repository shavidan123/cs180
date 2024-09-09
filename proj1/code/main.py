# CS180 (CS280A): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import skimage as sk
import skimage.io as skio
from skimage.transform import rescale
from scipy.ndimage import sobel
import os


def align(im1, im2, window=(-15, 15)):
    #implemented fixed cropping to not get the edges in the caculation
    crop= 0.15
    h, w = im1.shape
    crop_h = int(crop * h)
    crop_w = int(crop * w)
    im1_c = im1[crop_h:h-crop_h, crop_w:w-crop_w]
    h2, w2 = im2.shape
    crop_h2 = int(crop * h2)
    crop_w2 = int(crop * w2)
    im2_c = im2[crop_h2:h2-crop_h2, crop_w2:w2-crop_w2]
    
    min_diff = float('inf')
    best_offset = [0, 0]
    s_pos = window[0]
    e_pos = window[1] + 1
    #roll over all x, y offsets in the window given from start position to end position
    for x in range(s_pos, e_pos):
        for y in range(s_pos, e_pos):
            curr_diff = np.sum((np.roll(im1_c, (x, y), (0, 1)) - im2_c)**2)
            if curr_diff < min_diff:
                min_diff = curr_diff
                best_offset = [x, y]

    res = np.roll(im1, best_offset, (0, 1))
    best_offset = np.array(best_offset)

    return res, best_offset

def image_pyramid(im1, im2):
    if im1.shape[0] < 50: #base case
        return align(im1, im2)
    #recursive step
    res, best_offset = image_pyramid(rescale(im1, 0.5), rescale(im2, 0.5))
    best_offset *= 2
    result, next_offset = align(np.roll(im1, best_offset, (0, 1)), im2, (-1, 1)) #smaller window for pyramid steps
    best_offset += next_offset
    return result, best_offset

if not os.path.exists('output'):
    os.makedirs('output')

image_names = [
    'data/cathedral.jpg',
    'data/church.tif',
    'data/emir.tif',
    'data/harvesters.tif',
    'data/monastery.jpg',
    'data/icon.tif',
    'data/lady.tif',
    'data/melons.tif',
    'data/onion_church.tif',
    'data/sculpture.tif',
    'data/self_portrait.tif',
    'data/three_generations.tif',
    'data/tobolsk.jpg',
    'data/train.tif',
]

small_images = [
    'data/cathedral.jpg',
    'data/monastery.jpg',
    'data/toobolsk.jpg',
]

for imname in image_names:
    cur_dir = os.getcwd()
    #get everything up to the dot (slice out the last 3 chars)
    output_path = os.path.join(cur_dir, 'output',  os.path.basename(imname)[:-4] + ".jpg")
    print(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"calculating alignment for: {imname}")

    # Read the image, convert to double
    im = sk.img_as_float(skio.imread(imname))

    # Compute the height of each part (1/3 of total)
    height = np.floor(im.shape[0] / 3.0).astype(int)

    # Separate the color channels
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]

    if imname in small_images:
        ag, best_offset_green = align(g, b)
        ar, best_offset_red = align(r, b)
        print(best_offset_green, best_offset_red)

        # ceate colored image
        im_out = np.dstack([ar, ag, b])

        # Save the aligned image
        im_out = sk.img_as_ubyte(im_out)
        skio.imsave(output_path, im_out)
        skio.imshow(im_out)
        skio.show()

    else:
        #sobel filter
        sobel_b, sobel_g, sobel_r = np.abs(sobel(b)), np.abs(sobel(g)), np.abs(sobel(r))
        res, ag = image_pyramid(sobel_g, sobel_b)
        print(ag) #best offset
        res, ar = image_pyramid(sobel_r, sobel_b)
        print(ar)

        ag, ar = np.roll(g, ag, (0, 1)), np.roll(r, ar, (0, 1))

        im_out = np.dstack([ar, ag, b])
        im_out = sk.img_as_ubyte(im_out)
        skio.imsave(output_path, im_out)
        skio.imshow(im_out)
        skio.show()

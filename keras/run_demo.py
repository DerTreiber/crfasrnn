"""
MIT License

Copyright (c) 2017 Sadeep Jayasumana

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import sys
sys.path.insert(1, './src')
from crfrnn_model import get_crfrnn_model_def
import util
import cv2
import PIL

def resize_input(img):
    height, width = img.shape[:2]
    res_factor_h = min(1, 500/height)
    res_factor_w = min(1, 500/width)

    img_res = cv2.resize(img, (int(res_factor_w*width), int(res_factor_h*height)) )#, interpolation=cv2.INTER_AREA)
    return img_res, height, width

def resize_output(output, height, width):
    return output.resize((width, height), resample=PIL.Image.NEAREST)

def main():
    input_file = 'image.jpg'
    output_file = 'labels.png'

    # Download the model from https://goo.gl/ciEYZi
    saved_model_path = 'crfrnn_keras_model.h5'

    model = get_crfrnn_model_def()
    model.load_weights(saved_model_path)

    # resize with INTER_AREA
    img_data = cv2.imread(input_file)
    img_data, org_h, org_w = resize_input(img_data)

    img_data, img_h, img_w = util.get_preprocessed_image(img_data)
    probs = model.predict(img_data, verbose=False)[0, :, :, :]
    segmentation = util.get_label_image(probs, img_h, img_w)

    print(segmentation)
    # resize with INTER_NEAREST
    segmentation = resize_output(segmentation, org_h, org_w)

    segmentation.save(output_file)

if __name__ == '__main__':
    main()

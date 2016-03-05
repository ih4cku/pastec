/*****************************************************************************
 * Copyright (C) 2014 Visualink
 *
 * Authors: Adrien Maglo <adrien@visualink.io>
 *
 * This file is part of Pastec.
 *
 * Pastec is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Pastec is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with Pastec.  If not, see <http://www.gnu.org/licenses/>.
 *****************************************************************************/

#include <iostream>
#include <vector>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <glog/logging.h>

#include "imageloader.h"
#include "messages.h"

#define IMAGE_MAX_SIZE 1000

void ImageLoader::normImageSize(Mat &img) 
{
    int img_wid = img.cols;
    int img_hei = img.rows;

    CHECK(img_wid > 150 && img_hei > 150) << "image too small: " << img.size;

    if (img_wid > IMAGE_MAX_SIZE || img_hei > IMAGE_MAX_SIZE) {
        Size size;
        if (img_wid > img_hei) {
            size.width = IMAGE_MAX_SIZE;
            size.height = (float)img_hei / img_wid * IMAGE_MAX_SIZE;
        } else {
            size.width = (float)img_wid / img_hei * IMAGE_MAX_SIZE;
            size.height = IMAGE_MAX_SIZE;
        }
        resize(img, img, size);
        LOG(INFO) << "Image resize to " << size;
    }
}

u_int32_t ImageLoader::loadImage(unsigned i_imgSize, char *p_imgData, Mat &img)
{
    vector<char> imgData(i_imgSize);
    memcpy(imgData.data(), p_imgData, i_imgSize);

    try
    {
        img = imdecode(imgData, CV_LOAD_IMAGE_GRAYSCALE);
    }
    catch (cv::Exception& e) // The decoding of an image can raise an exception.
    {
        const char* err_msg = e.what();
        cout << "Exception caught: " << err_msg << endl;
        return IMAGE_NOT_DECODED;
    }

    if (!img.data)
    {
        cout << "Error reading the image." << std::endl;
        return IMAGE_NOT_DECODED;
    }

    normImageSize(img);

    return OK;
}

u_int32_t ImageLoader::loadImage(string image_path, Mat &img)
{
    img = imread(image_path, CV_LOAD_IMAGE_GRAYSCALE);
    CHECK(img.data) << "error reading " << image_path;
    normImageSize(img);
}

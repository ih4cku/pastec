#include <iostream>
#include <string>
#include <unordered_set>
#include <glog/logging.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "imageadder.h"
#include "imageloader.h"

using namespace std;
using namespace cv;


void ImageAdder::addImage(string image_path, int image_id, int &num_feat_extracted) 
{
    Mat img;
    ImageLoader::loadImage(image_path, img);
    LOG(INFO) << "adding " << image_path << img.size();
    
    vector<KeyPoint> keypoints;
    Mat descriptors;

    ORB(1000, 1.09, 16)(img, noArray(), keypoints, descriptors);
    num_feat_extracted = keypoints.size();
    LOG(INFO) << num_feat_extracted << " features extracted.";

    list<HitForward> imageHits;
    unordered_set<u_int32_t> matchedWords;

    // record for visualization
#ifdef PASTEC_DEBUG
    vector<KeyPoint> keep_keypoints;
#endif

    for (unsigned i = 0; i < keypoints.size(); ++i)
    {
        // Recording the angle on 16 bits.
        u_int16_t angle = keypoints[i].angle / 360 * (1 << 16);
        u_int16_t x = keypoints[i].pt.x;
        u_int16_t y = keypoints[i].pt.y;

        vector<int> indices(1);
        vector<int> dists(1);
        word_index_->knnSearch(descriptors.row(i), indices, dists, 1);

        for (unsigned j = 0; j < indices.size(); ++j)
        {
            const unsigned word_id = indices[j];
            if (matchedWords.find(word_id) == matchedWords.end())
            {
                HitForward newHit;
                newHit.i_wordId = word_id;
                newHit.i_imageId = image_id;
                newHit.i_angle = angle;
                newHit.x = x;
                newHit.y = y;
                imageHits.push_back(newHit);
                matchedWords.insert(word_id);
#ifdef PASTEC_DEBUG
                keep_keypoints.push_back(keypoints[i]);
#endif
            }
        }
    }


#ifdef PASTEC_DEBUG
    LOG(INFO) << "total keypoints: " << num_feat_extracted;
    LOG(INFO) << "keeped keypoints: " << keep_keypoints.size();

    // Draw keypoints.
    Mat img_res;
    drawKeypoints(img, keypoints, img_res, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    ostringstream oss;
    oss << image_id << "_feats.jpg";
    imwrite(oss.str(), img_res);
#endif

    // Record the hits.
    index_->addImage(image_id, imageHits);
    index_->addTag(image_id, image_path);
}

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
#include <cassert>
#include <math.h>
#include <algorithm>
#include <memory>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <glog/logging.h>

#include "imagereranker.h"
#include "imageloader.h"
#include "logging.h"

using namespace cv;
using namespace std;

// build keypoints and matches with task
void buildMatches(const RANSACTask &task, vector<KeyPoint> &kps_index, vector<KeyPoint> &kps_req, vector<DMatch> &matches) 
{
    const vector<Point2f> &req_pts = task.points1;
    const vector<Point2f> &index_pts = task.points2;
    CHECK(req_pts.size() == index_pts.size()) << "req_pts.size() != index_pts.size()";

    int num_pts = req_pts.size();
    for(int i = 0; i < num_pts; ++i) {
        KeyPoint _kp_idx, _kp_req;
        _kp_idx.pt = index_pts[i];
        _kp_req.pt = req_pts[i];
        kps_index.push_back(_kp_idx);
        kps_req.push_back(_kp_req);
        matches.push_back(DMatch(i, i, -1));
    }
}

void *RANSACThread::run()
{
    for (unsigned i = 0; i < imageIds_.size(); ++i)
    {
        const unsigned i_imageId = imageIds_[i];
        const Histogram histogram = histograms_[i];
        unsigned i_binMax = max_element(histogram.bins, histogram.bins + HISTOGRAM_NB_BINS) - histogram.bins;
        float i_maxVal = histogram.bins[i_binMax];

#ifdef PASTEC_DEBUG
        string s_ok = " bad";
        if (i_maxVal >= HISTOGRAM_MIN_VALUE && imgTasks_[i_imageId].points1.size() >= RANSAC_MIN_INLINERS)
            s_ok = " good";
        LOG(INFO) << DEBUG_ID_STRING << i_imageId <<  s_ok\
                  << ", max bin: " << i_maxVal \
                  << ", matched points: " << imgTasks_[i_imageId].points1.size();
        
        string image_path;
        index_->getTag(i_imageId, image_path);
        Mat img_index;
        ImageLoader::loadImage(image_path, img_index);

        vector<KeyPoint> kps_index, kps_req;
        vector<DMatch> matches;
        buildMatches(imgTasks_[i_imageId], kps_index, kps_req, matches);

        Mat img_match;
        drawMatches(img_index, kps_index, reqImage_, kps_req, matches, img_match);

        ostringstream oss;
        oss << i_imageId << ".jpg";
        imwrite(oss.str(), img_match);
        LOG(INFO) << DEBUG_ID_STRING << i_imageId << ", matches save to: " << oss.str();
#endif


        if (i_maxVal >= HISTOGRAM_MIN_VALUE)
        {
            RANSACTask &task = imgTasks_[i_imageId];
            assert(task.points1.size() == task.points2.size());

            if (task.points1.size() >= RANSAC_MIN_INLINERS)
            {
                Mat H = pastecEstimateRigidTransform(task.points2, task.points1, true);

                if (countNonZero(H) == 0)
                {
#ifdef PASTEC_DEBUG
                    LOG(INFO) << DEBUG_ID_STRING << i_imageId << ", H zeros: " << countNonZero(H);
#endif
                    continue;
                }

                Rect bRect1 = boundingRect(task.points1);

                pthread_mutex_lock(&mutex_);
                rankedResultsOut_.push(SearchResult(i_maxVal, i_imageId, bRect1));
                pthread_mutex_unlock(&mutex_);
            }
        }
    }
}


void ImageReranker::rerank(const unordered_map<u_int32_t, list<Hit> > &imagesReqHits,
                           const unordered_map<u_int32_t, vector<Hit> > &indexHits,
                           const priority_queue<SearchResult> &rankedResultsIn,
                           priority_queue<SearchResult> &rankedResultsOut,
                           unsigned i_nbResults)
{
    unordered_set<u_int32_t> topImageIds;

    // Extract the first i_nbResults ranked images.
    getFirstImageIds(rankedResultsIn, i_nbResults, topImageIds);
    
#ifdef PASTEC_DEBUG
    ostringstream oss;
    oss << "Top " << i_nbResults << " image ids (unordered): ";
    for (auto &e : topImageIds)
        oss << e << ", ";
    LOG(INFO) << oss.str();
#endif
    
    unordered_map<u_int32_t, RANSACTask> imgTasks; // key: library image id, value: matching points

    // Compute the histograms.
    unordered_map<u_int32_t, Histogram> histograms; // key: the image id, value: the corresponding histogram.

    for (unordered_map<u_int32_t, list<Hit> >::const_iterator it = imagesReqHits.begin();
         it != imagesReqHits.end(); ++it)
    {
        // Try to match all the visual words of the request image.
        const unsigned i_wordId = it->first;
        const list<Hit> &hits = it->second;

        assert(hits.size() == 1);

        // If there is several hits for the same word in the image...
        const u_int16_t i_angle1 = hits.front().i_angle;
        const Point2f point1(hits.front().x, hits.front().y);
        auto tmp_it = indexHits.find(i_wordId);
        CHECK(tmp_it != indexHits.end()) << i_wordId << " not in indexHits.";
        const vector<Hit> &hitsOfWord = tmp_it->second;

        for (unsigned i = 0; i < hitsOfWord.size(); ++i)
        {
            const u_int32_t i_imageId = hitsOfWord[i].i_imageId;
            // Test if the image belongs to the image to rerank.
            if (topImageIds.find(i_imageId) != topImageIds.end())
            {
                const u_int16_t i_angle2 = hitsOfWord[i].i_angle;
                float f_diff = angleDiff(i_angle1, i_angle2);
                unsigned bin = (f_diff - DIFF_MIN) / 360 * HISTOGRAM_NB_BINS;
                assert(bin < HISTOGRAM_NB_BINS);

                Histogram &histogram = histograms[i_imageId];
                histogram.bins[bin]++;
                histogram.i_total++;

                const Point2f point2(hitsOfWord[i].x, hitsOfWord[i].y);
                RANSACTask &imgTask = imgTasks[i_imageId];

                imgTask.points1.push_back(point1);
                imgTask.points2.push_back(point2);
            }
        }
    }

    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

    #define NB_RANSAC_THREAD 4
    unique_ptr<RANSACThread> threads[NB_RANSAC_THREAD];

    for (unsigned i = 0; i < NB_RANSAC_THREAD; ++i)
        threads[i] = unique_ptr<RANSACThread>(new RANSACThread(mutex, imgTasks, rankedResultsOut,
                                                               searcher_->reqImage_, searcher_->index_));

    // Rank the images according to their histogram.
    unsigned i = 0;
    for (unordered_map<unsigned, Histogram>::iterator it = histograms.begin();
         it != histograms.end(); ++it, ++i)
    {
        unsigned i_imageId = it->first;
        Histogram histogram = it->second;
        threads[i % NB_RANSAC_THREAD]->imageIds_.push_back(i_imageId);
        threads[i % NB_RANSAC_THREAD]->histograms_.push_back(histogram);
    }

    LOG(INFO) << "HISTOGRAM_MIN_VALUE: " << HISTOGRAM_MIN_VALUE;
    LOG(INFO) << "RANSAC_MIN_INLINERS: " << RANSAC_MIN_INLINERS;

    // Compute
    for (unsigned i = 0; i < NB_RANSAC_THREAD; ++i)
        threads[i]->start();
    for (unsigned i = 0; i < NB_RANSAC_THREAD; ++i)
        threads[i]->join();

    pthread_mutex_destroy(&mutex);
}


class Pos {
public:
    Pos(int x, int y) : x(x), y(y) {}

    inline bool operator< (const Pos &rhs) const {
        if (x != rhs.x)
            return x < rhs.x;
        else
            return y < rhs.y;
    }

private:
    int x, y;
};


/**
 * @brief Return the first ids of ranked images.
 * @param rankedResultsIn the ranked images.
 * @param i_nbResults the number of images to return.
 * @param firstImageIds a set to return the image ids.
 */
void ImageReranker::getFirstImageIds(priority_queue<SearchResult> rankedResultsIn,
                                     unsigned i_nbResults, unordered_set<u_int32_t> &firstImageIds)
{
    unsigned i_res = 0;
    while(!rankedResultsIn.empty()
          && i_res < i_nbResults)
    {
        const SearchResult &res = rankedResultsIn.top();
        firstImageIds.insert(res.i_imageId);
        rankedResultsIn.pop();
        i_res++;
    }
}


float ImageReranker::angleDiff(unsigned i_angle1, unsigned i_angle2)
{
    // Convert the angle in the [-180, 180] range.
    float i1 = (float)i_angle1 * 360 / (1 << 16);
    float i2 = (float)i_angle2 * 360 / (1 << 16);

    i1 = (i1 <= 180) ? i1 : (i1 - 360);
    i2 = (i2 <= 180) ? i2 : (i2 - 360);

    // Compute the difference between the two angles.
    float diff = i1 - i2;
    if (diff < DIFF_MIN)
        diff += 360;
    else if (diff >= 360 + DIFF_MIN)
        diff -= 360;

    assert(diff >= DIFF_MIN);
    assert(diff < 360 + DIFF_MIN);

    return diff;
}

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

#ifndef PASTEC_IMAGERERANKER_H
#define PASTEC_IMAGERERANKER_H

#include <sys/types.h>

#include <queue>
#include <list>
#include <unordered_map>
#include <unordered_set>

#include <opencv2/core/core.hpp>

#include <thread.h>

#include "searchResult.h"
#include "hit.h"
#include "orb/orbsearcher.h"

using namespace std;
using namespace cv;

class ORBSearcher;

class ImageReranker
{
public:
    ImageReranker(ORBSearcher *sch) : searcher_(sch) {}
    void rerank(const std::unordered_map<u_int32_t, list<Hit> > &imagesReqHits,
                const std::unordered_map<u_int32_t, vector<Hit> > &indexHits,
                const priority_queue<SearchResult> &rankedResultsIn,
                priority_queue<SearchResult> &rankedResultsOut,
                unsigned i_nbResults);

private:
    ORBSearcher *searcher_;
    float angleDiff(unsigned i_angle1, unsigned i_angle2);
    void getFirstImageIds(priority_queue<SearchResult> rankedResultsIn,
                          unsigned i_nbResults, unordered_set<u_int32_t> &firstImageIds);
};


// A task that must be performed when the rerankRANSAC function is called.
struct RANSACTask
{
    vector<Point2f> points1;
    vector<Point2f> points2;
};


#define HISTOGRAM_NB_BINS 36
#define DIFF_MIN -360.0f / (2.0f * HISTOGRAM_NB_BINS)


struct Histogram
{
    Histogram() : i_total(0)
    {
        for (unsigned i = 0; i < HISTOGRAM_NB_BINS; ++i)
            bins[i] = 0;
    }
    unsigned bins[HISTOGRAM_NB_BINS];
    unsigned i_total;
};


#define HISTOGRAM_MIN_VALUE 10
#define RANSAC_MIN_INLINERS 12


class RANSACThread : public Thread
{
public:
    RANSACThread(pthread_mutex_t &mutex,
                 std::unordered_map<u_int32_t, RANSACTask> &imgTasks,
                 priority_queue<SearchResult> &rankedResultsOut,
                 Mat reqImg, ORBIndex *index)
        : mutex_(mutex), imgTasks_(imgTasks), rankedResultsOut_(rankedResultsOut),
          reqImage_(reqImg), index_(index)
    { }

public:
    void *run();

    pthread_mutex_t &mutex_;
    std::unordered_map<u_int32_t, RANSACTask> &imgTasks_;
    priority_queue<SearchResult> &rankedResultsOut_;
    deque<unsigned> imageIds_;
    deque<Histogram> histograms_;

private:
    Mat reqImage_;
    ORBIndex *index_;
    void getRTMatrix(const Point2f* a, const Point2f* b,
                     int count, Mat& M, bool fullAffine);
    cv::Mat pastecEstimateRigidTransform(InputArray src1, InputArray src2,
                                         bool fullAffine);
};


#endif // PASTEC_IMAGERERANKER_H

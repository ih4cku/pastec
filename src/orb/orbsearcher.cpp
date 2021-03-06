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
#include <fstream>
#include <memory>
#include <sys/time.h>

#include <set>
#ifndef __APPLE__
#include <tr1/unordered_set>
#include <tr1/unordered_map>
#else
#include <unordered_set>
#include <unordered_map>
#endif
#include <queue>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <glog/logging.h>

#include "orbsearcher.h"
#include "messages.h"
#include "imageloader.h"

#include "logging.h"

#ifndef __APPLE__
using namespace std::tr1;
#endif

ORBSearcher::ORBSearcher(ORBIndex *index, ORBWordIndex *wordIndex)
    : index_(index), wordIndex_(wordIndex)
{ 
    reranker_ = unique_ptr<ImageReranker>(new ImageReranker(this));
}


ORBSearcher::~ORBSearcher()
{ }


/**
 * @brief The RankingThread class
 * This threads computes the tf-idf weights of the images that contains the words
 * given in argument.
 */
class TfidfThread : public Thread
{
public:
    TfidfThread(ORBIndex *index, const unsigned i_nbTotalIndexedImages,
                  std::unordered_map<u_int32_t, vector<Hit> > &indexHits)
        : index_(index), i_nbTotalIndexedImages_(i_nbTotalIndexedImages),
          indexHits_(indexHits) { }

    void addWord(u_int32_t i_wordId)
    {
        wordIds_.push_back(i_wordId);
    }

    void *run()
    {
        weights_.rehash(wordIds_.size());

        for (deque<u_int32_t>::const_iterator it = wordIds_.begin();
            it != wordIds_.end(); ++it)
        {
            const vector<Hit> &hits = indexHits_[*it];

            const float f_weight = log((float)i_nbTotalIndexedImages_ / hits.size());

            for (vector<Hit>::const_iterator it2 = hits.begin();
                 it2 != hits.end(); ++it2)
            {
                /* TF-IDF according to the paper "Video Google:
                 * A Text Retrieval Approach to Object Matching in Videos" */
                unsigned i_totalNbWords = index_->countTotalNbWord(it2->i_imageId);
                weights_[it2->i_imageId] += f_weight / i_totalNbWords;
            }
        }

        return NULL;
    }

    ORBIndex *index_;
    const unsigned i_nbTotalIndexedImages_;
    std::unordered_map<u_int32_t, vector<Hit> > &indexHits_;
    deque<u_int32_t> wordIds_;
    std::unordered_map<u_int32_t, float> weights_; // key: image id, value: image score.
};


/**
 * @brief Processed a search request.
 * @param request the request to proceed.
 */
u_int32_t ORBSearcher::searchImage(SearchRequest &request)
{
    timeval t[3];
    gettimeofday(&t[0], NULL);

    LOG(INFO) << "Loading the image and extracting the ORBs.";

    u_int32_t i_ret = ImageLoader::loadImage(request.imageData.size(),
                                             request.imageData.data(), reqImage_);

    if (i_ret != OK)
        return i_ret;

    vector<KeyPoint> keypoints;
    Mat descriptors;

    ORB(2000, 1.02, 100)(reqImage_, noArray(), keypoints, descriptors);

    gettimeofday(&t[1], NULL);

    LOG(INFO) << "time: " << getTimeDiff(t[0], t[1]) << " ms.";
    LOG(INFO) << "Looking for the visual words.";

    const unsigned i_nbTotalIndexedImages = index_->getTotalNbIndexedImages();
    const unsigned i_maxNbOccurences = i_nbTotalIndexedImages > 10000 ?
                                       0.15 * i_nbTotalIndexedImages
                                       : i_nbTotalIndexedImages;

    std::unordered_map<u_int32_t, list<Hit> > imageReqHits; // key: visual word, value: the found angles
    for (unsigned i = 0; i < keypoints.size(); ++i)
    {
        #define NB_NEIGHBORS 1

        vector<int> indices(NB_NEIGHBORS);
        vector<int> dists(NB_NEIGHBORS);
        wordIndex_->knnSearch(descriptors.row(i), indices,
                           dists, NB_NEIGHBORS);

        for (unsigned j = 0; j < indices.size(); ++j)
        {
            const unsigned i_wordId = indices[j];

            if (index_->getWordNbOccurences(i_wordId) > i_maxNbOccurences)
                continue;

            if (imageReqHits.find(i_wordId) == imageReqHits.end())
            {
                // Convert the angle to a 16 bit integer.
                Hit hit;
                hit.i_imageId = 0;
                hit.i_angle = keypoints[i].angle / 360 * (1 << 16);
                hit.x = keypoints[i].pt.x;
                hit.y = keypoints[i].pt.y;
                imageReqHits[i_wordId].push_back(hit);
            }
        }
    }

    gettimeofday(&t[2], NULL);
    LOG(INFO) << "time: " << getTimeDiff(t[1], t[2]) << " ms.";

    LOG(INFO) << "request hits " << imageReqHits.size() << " words";
    return processSimilar(request, imageReqHits);
}


/**
 * @brief Processed a similarity request.
 * @param request the request to proceed.
 */
u_int32_t ORBSearcher::searchSimilar(SearchRequest &request)
{
    timeval t[2];
    gettimeofday(&t[0], NULL);

    cout << "Loading the image words from the index." << endl;

    // key: visual word, value: the found angles
    std::unordered_map<u_int32_t, list<Hit> > imageReqHits;
    u_int32_t i_ret = index_->getImageWords(request.imageId, imageReqHits);

    if (i_ret != OK)
        return i_ret;

    gettimeofday(&t[1], NULL);
    cout << "time: " << getTimeDiff(t[0], t[1]) << " ms." << endl;

    return processSimilar(request, imageReqHits);
}


u_int32_t ORBSearcher::processSimilar(SearchRequest &request,
        const std::unordered_map<u_int32_t, list<Hit> > &imageReqHits)
{
    timeval t[7];
    gettimeofday(&t[0], NULL);

    const unsigned i_nbTotalIndexedImages = index_->getTotalNbIndexedImages();

    LOG(INFO) << imageReqHits.size() << " visual words kept for the request.";
    LOG(INFO) << i_nbTotalIndexedImages << " images indexed in the index.";

    std::unordered_map<u_int32_t, vector<Hit> > indexHits; // key: visual word id, values: index hits.
    indexHits.rehash(imageReqHits.size());
    index_->getImagesWithVisualWords(imageReqHits, indexHits);

    gettimeofday(&t[1], NULL);
    LOG(INFO) << "time: " << getTimeDiff(t[0], t[1]) << " ms.";
    LOG(INFO) << "Ranking the images.";

    //--------------------- Tf-Idf --------------------- 
    index_->readLock();
    #define NB_RANKING_THREAD 4

    // Map the ranking to threads.
    unsigned i_wordsPerThread = indexHits.size() / NB_RANKING_THREAD + 1;
    unique_ptr<TfidfThread> threads[NB_RANKING_THREAD];

    std::unordered_map<u_int32_t, vector<Hit> >::const_iterator it = indexHits.begin();
    for (unsigned i = 0; i < NB_RANKING_THREAD; ++i)
    {
        threads[i] = unique_ptr<TfidfThread>(new TfidfThread(index_, i_nbTotalIndexedImages, indexHits));

        unsigned i_nbWords = 0;
        for (; it != indexHits.end() && i_nbWords < i_wordsPerThread; ++it, ++i_nbWords)
            threads[i]->addWord(it->first);
    }

    gettimeofday(&t[2], NULL);
    LOG(INFO) << "init threads time: " << getTimeDiff(t[1], t[2]) << " ms.";

    // Compute
    for (unsigned i = 0; i < NB_RANKING_THREAD; ++i)
        threads[i]->start();
    for (unsigned i = 0; i < NB_RANKING_THREAD; ++i)
        threads[i]->join();

    gettimeofday(&t[3], NULL);
    LOG(INFO) << "compute time: " << getTimeDiff(t[2], t[3]) << " ms.";

    // Reduce...
    std::unordered_map<u_int32_t, float> weights; // key: image id, value: image score.
    weights.rehash(i_nbTotalIndexedImages);
    for (unsigned i = 0; i < NB_RANKING_THREAD; ++i)
        for (std::unordered_map<u_int32_t, float>::const_iterator it = threads[i]->weights_.begin();
            it != threads[i]->weights_.end(); ++it)
            weights[it->first] += it->second;

    gettimeofday(&t[4], NULL);
    LOG(INFO) << "reduce time: " << getTimeDiff(t[3], t[4]) << " ms.";

    index_->unlock();

    priority_queue<SearchResult> rankedResults;
    for (std::unordered_map<unsigned, float>::const_iterator it = weights.begin();
         it != weights.end(); ++it)
    {
        //cout << "Second: " << it->second << " First: " << it->first << endl;
        rankedResults.push(SearchResult(it->second, it->first, Rect()));
    }

#ifdef PASTEC_DEBUG
    printPriorityQueue(rankedResults, "[TF-IDF ranking result] ");
#endif

    int top_k_rerank = 300;
    gettimeofday(&t[5], NULL);
    LOG(INFO) << "rankedResult time: " << getTimeDiff(t[4], t[5]) << " ms.";
    LOG(INFO) << "Reranking " << top_k_rerank << " among " << rankedResults.size() << " images.";

    //--------------------- RANSAC --------------------- 
    priority_queue<SearchResult> rerankedResults;
    reranker_->rerank(imageReqHits, indexHits, rankedResults, rerankedResults, top_k_rerank);

#ifdef PASTEC_DEBUG
    printPriorityQueue(rerankedResults, "[RANSAC reranking result] ");
#endif

    gettimeofday(&t[6], NULL);
    LOG(INFO) << "time: " << getTimeDiff(t[5], t[6]) << " ms.";
    LOG(INFO) << "Returning the results. ";

    returnResults(rerankedResults, request, 100);

    return SEARCH_RESULTS;
}


/**
 * @brief Return to the client the found results.
 * @param rankedResults the ranked list of results.
 * @param req the received search request.
 * @param i_maxNbResults the maximum number of results returned.
 */
void ORBSearcher::returnResults(priority_queue<SearchResult> &rankedResults,
                                  SearchRequest &req, unsigned i_maxNbResults)
{
    list<u_int32_t> imageIds;

    unsigned i_res = 0;
    while(!rankedResults.empty()
          && i_res < i_maxNbResults)
    {
        const SearchResult &res = rankedResults.top();
        imageIds.push_back(res.i_imageId);
        i_res++;
        cout << "Id: " << res.i_imageId << ", score: " << res.f_weight << endl;
        req.results.push_back(res.i_imageId);
        req.boundingRects.push_back(res.boundingRect);
        req.scores.push_back(res.f_weight);

        string tag;
        if (index_->getTag(res.i_imageId, tag) == OK)
            req.tags.push_back(tag);
        else
            req.tags.push_back("");

        rankedResults.pop();
    }
}


/**
 * @brief Get the time difference in ms between two instants.
 * @param t1
 * @param t2
 */
unsigned long ORBSearcher::getTimeDiff(const timeval t1, const timeval t2) const
{
    return ((t2.tv_sec - t1.tv_sec) * 1000000
            + (t2.tv_usec - t1.tv_usec)) / 1000;
}

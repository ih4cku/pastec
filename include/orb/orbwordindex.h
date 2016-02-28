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

#ifndef PASTEC_ORBWORDINDEX_H
#define PASTEC_ORBWORDINDEX_H

#include <string>
#include <vector>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/flann/flann.hpp>

using namespace cv;
using namespace std;

typedef cvflann::HierarchicalClusteringIndex<cvflann::Hamming<unsigned char> > FlannIndex;

class ORBWordIndex
{
public:
    ORBWordIndex(string visualWordsPath);
    ORBWordIndex(string visualWordsPath, string treePath);
    ~ORBWordIndex() {};
    void knnSearch(const Mat &query, vector<int>& indices,
                   vector<int> &dists, int knn);

private:
    bool readVisualWords(string fileName);
    void initIndex(string visualWordsPath);
    void saveIndex(string treePath);
    bool loadIndex(string treePath);

    unique_ptr<Mat> words;  // The matrix that stores the visual words.
    unique_ptr<FlannIndex> kdIndex; // The kd-tree index.
};

#endif // PASTEC_ORBWORDINDEX_H

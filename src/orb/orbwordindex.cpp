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
#include <glog/logging.h>

#include <orbwordindex.h>


ORBWordIndex::ORBWordIndex(string visualWordsPath) : words(new Mat(0, 32, CV_8U))
{
    initIndex(visualWordsPath);

    LOG(INFO) << "building index";
    kdIndex->buildIndex();
}

ORBWordIndex::ORBWordIndex(string visualWordsPath, string treePath) : words(new Mat(0, 32, CV_8U))
{
    initIndex(visualWordsPath);

    if (!loadIndex(treePath))
        saveIndex(treePath);
}

void ORBWordIndex::initIndex(string visualWordsPath) {
    LOG(INFO) << "init index";
    if (!readVisualWords(visualWordsPath))
        exit(1);
    assert(words->rows == 1000000);

    cvflann::Matrix<unsigned char> m_features
            ((unsigned char*)words->ptr<unsigned char>(0), words->rows, words->cols);
    kdIndex = unique_ptr<FlannIndex>(
        new FlannIndex(m_features, cvflann::HierarchicalClusteringIndexParams(10, cvflann::FLANN_CENTERS_RANDOM, 8, 100)));
}

void ORBWordIndex::saveIndex(string treePath) {
    LOG(INFO) << "building index";
    kdIndex->buildIndex();

    LOG(INFO) << "saving tree to " << treePath;
    FILE* fp = fopen(treePath.c_str(), "wb");
    assert(fp);
    kdIndex->saveIndex(fp);
    fclose(fp);
}

bool ORBWordIndex::loadIndex(string treePath) {
    LOG(INFO) << "loading index from " << treePath;
    FILE* fp = fopen(treePath.c_str(), "rb");
    if (fp) {
        kdIndex->loadIndex(fp);
        fclose(fp);
        return true;
    } else {
        return false;
    }
}


void ORBWordIndex::knnSearch(const Mat& query, vector<int>& indices,
                          vector<int>& dists, int knn)
{
    cvflann::KNNResultSet<int> m_indices(knn);

    m_indices.init(indices.data(), dists.data());

    kdIndex->findNeighbors(m_indices, (unsigned char*)query.ptr<unsigned char>(0),
                           cvflann::SearchParams(2000));
}


/**
 * @brief Read the list of visual words from an external file.
 * @param fileName the path of the input file name.
 * @param words a pointer to a matrix to store the words.
 * @return true on success else false.
 */
bool ORBWordIndex::readVisualWords(string fileName)
{
    LOG(INFO) << "Reading the visual words file.";

    // Open the input file.
    ifstream ifs;
    ifs.open(fileName.c_str(), ios_base::binary);

    if (!ifs.good())
    {
        LOG(INFO) << "Could not open the input file.";
        return false;
    }

    unsigned char c;
    while (ifs.good())
    {
        Mat line(1, 32, CV_8U);
        for (unsigned i_col = 0; i_col < 32; ++i_col)
        {
            ifs.read((char*)&c, sizeof(unsigned char));
            line.at<unsigned char>(0, i_col) = c;
        }
        if (!ifs.good())
            break;
        words->push_back(line);
    }

    ifs.close();

    return true;
}

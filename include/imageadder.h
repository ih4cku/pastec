#ifndef IMAGEADDER_H
#define IMAGEADDER_H

#include <string>
#include <memory>

#include "orb/orbindex.h"
#include "orb/orbwordindex.h"

using namespace std;

class ImageAdder {
public:
    ImageAdder(shared_ptr<ORBIndex> index, shared_ptr<ORBWordIndex> word_index):
        index_(index), word_index_(word_index) 
    {}

    void addImage(string image_path, int image_id, int &num_feat_extracted);

private:
    shared_ptr<ORBIndex> index_;
    shared_ptr<ORBWordIndex> word_index_;
};

#endif
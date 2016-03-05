#include <glog/logging.h>
#include "logging.h"

using namespace std;

void printRankedResult(priority_queue<SearchResult> ranked_res, string log_info)
{
    int i_rank = 0;
    while (!ranked_res.empty()){
        auto r = ranked_res.top();
        LOG(INFO) << log_info << " : " << r.i_imageId << " -> " << i_rank;
        ranked_res.pop();
        i_rank++;
    }
}

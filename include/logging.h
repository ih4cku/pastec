#ifndef LOGGING_H
#define LOGGING_H

#include <queue>
#include <string>

#include "searchResult.h"

#define DEBUG_ID_STRING "LIBRARY_IMAGE_ID: "

void printPriorityQueue(std::priority_queue<SearchResult> ranked_res, std::string log_info);

#endif

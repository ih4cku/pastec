#ifndef LOGGING_H
#define LOGGING_H

#include <queue>
#include <string>

#include "searchResult.h"

void printRankedResult(std::priority_queue<SearchResult> ranked_res, std::string log_info);

#endif
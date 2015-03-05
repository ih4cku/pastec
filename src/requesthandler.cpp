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
#include <stdlib.h>

#include <jsoncpp/json/json.h>

#include <requesthandler.h>
#include <messages.h>
#include <featureextractor.h>
#include <searcher.h>
#include <index.h>

#include <imageloader.h>
#include <opencv2/highgui/highgui.hpp>


RequestHandler::RequestHandler(FeatureExtractor *featureExtractor,
               Searcher *imageSearcher, Index *index)
    : featureExtractor(featureExtractor), imageSearcher(imageSearcher),
      index(index)
{ }


/**
 * @brief Parse an URI.
 * @param uri the uri string.
 * @return the vector containing all the URI element between slashes.
 */
vector<string> RequestHandler::parseURI(string uri)
{
    vector<string> ret;

    if (uri[0] != '/')
        return ret;

    size_t pos1 = 1;
    size_t pos2;

    while ((pos2 = uri.find('/', pos1)) != string::npos)
    {
        ret.push_back(uri.substr(pos1, pos2 - pos1));
        pos1 = pos2 + 1;
    }

    ret.push_back(uri.substr(pos1, uri.length() - pos1));

    return ret;
}


/**
 * @brief Test that a given parsed URI corresponds to a given request pattern.
 * @param parsedURI the parsed URI.
 * @param p_pattern the request pattern.
 * @return true if there is a correspondance, else false.
 */
bool RequestHandler::testURIWithPattern(vector<string> parsedURI, string p_pattern[])
{
    for (unsigned i = 0; ; ++i)
    {
        if (p_pattern[i] == "")
            break;
        else if (p_pattern[i] == "IDENTIFIER")
        {
            // Test we have a number here.
            if (parsedURI[i].length() == 0)
                return false;
            char* p;
            long n = strtol(parsedURI[i].c_str(), &p, 10);
            if (*p != 0)
                return false;
            if (n < 0)
                return false;
        }
        else if (p_pattern[i] != parsedURI[i])
            return false;
    }

    return true;
}


/**
 * @brief RequestHandler::handlePost
 * @param uri
 * @param p_data
 * @return
 */
void RequestHandler::handleRequest(ConnectionInfo &conInfo)
{
    vector<string> parsedURI = parseURI(conInfo.url);

    string p_image[] = {"index", "images", "IDENTIFIER", ""};
    string p_searchImage[] = {"index", "searcher", ""};
    string p_ioIndex[] = {"index", "io", ""};
    string p_root[] = {""};

    Json::Value ret;
    conInfo.answerCode = MHD_HTTP_OK;

    if (testURIWithPattern(parsedURI, p_image)
        && conInfo.connectionType == PUT)
    {
        u_int32_t i_imageId = atoi(parsedURI[2].c_str());

        u_int32_t i_ret = featureExtractor->processNewImage(
            i_imageId, conInfo.uploadedData.size(), conInfo.uploadedData.data());

        ret["type"] = Converter::codeToString(i_ret);
        ret["image_id"] = Json::Value(i_imageId);
    }
    else if (testURIWithPattern(parsedURI, p_image)
             && conInfo.connectionType == DELETE)
    {
        u_int32_t i_imageId = atoi(parsedURI[2].c_str());

        u_int32_t i_ret = index->removeImage(i_imageId);
        ret["type"] = Converter::codeToString(i_ret);
        ret["image_id"] = Json::Value(i_imageId);
    }
    else if (testURIWithPattern(parsedURI, p_searchImage)
             && conInfo.connectionType == POST)
    {
        SearchRequest req;

        req.imageData = conInfo.uploadedData;
        req.client = NULL;
        u_int32_t i_ret = imageSearcher->searchImage(req);

        ret["type"] = Converter::codeToString(i_ret);
        if (i_ret == SEARCH_RESULTS)
        {
            // Return the image ids
            Json::Value imageIds(Json::arrayValue);
            for (unsigned i = 0; i < req.results.size(); ++i)
                imageIds.append(req.results[i]);
            ret["image_ids"] = imageIds;

            // Return the bounding rects
            Json::Value boundingRects(Json::arrayValue);
            for (unsigned i = 0; i < req.boundingRects.size(); ++i)
            {
                Rect r = req.boundingRects[i];
                Json::Value rVal;
                rVal["x"] = r.x; rVal["y"] = r.y;
                rVal["width"] = r.width; rVal["height"] = r.height;
                boundingRects.append(rVal);
            }
            ret["bounding_rects"] = boundingRects;

            // Return the scores
            Json::Value scores(Json::arrayValue);
            for (unsigned i = 0; i < req.scores.size(); ++i)
                scores.append(req.scores[i]);
            ret["scores"] = scores;
        }
    }
    else if (testURIWithPattern(parsedURI, p_ioIndex)
             && conInfo.connectionType == POST)
    {
        string dataStr(conInfo.uploadedData.begin(),
                       conInfo.uploadedData.end());

        Json::Value data = StringToJson(dataStr);
        u_int32_t i_ret;
        if (data["type"] == "LOAD")
            i_ret = index->load(data["index_path"].asString());
        else if (data["type"] == "WRITE")
            i_ret = index->write(data["index_path"].asString());
        else if (data["type"] == "CLEAR")
            i_ret = index->clear();
        else
            i_ret = MISFORMATTED_REQUEST;

        ret["type"] = Converter::codeToString(i_ret);
    }
    else if (testURIWithPattern(parsedURI, p_root)
             && conInfo.connectionType == POST)
    {
        string dataStr(conInfo.uploadedData.begin(),
                       conInfo.uploadedData.end());

        Json::Value data = StringToJson(dataStr);
        u_int32_t i_ret;
        if (data["type"] == "PING")
        {
            cout << "Ping received." << endl;
            i_ret = PONG;
        }
        else
            i_ret = MISFORMATTED_REQUEST;

        ret["type"] = Converter::codeToString(i_ret);
    }
    else
    {
        conInfo.answerCode = MHD_HTTP_INTERNAL_SERVER_ERROR;
        ret["type"] = Converter::codeToString(MISFORMATTED_REQUEST);
    }

    conInfo.answerString = JsonToString(ret);
}


/**
 * @brief Conver to JSON value to a string.
 * @param data the JSON value.
 * @return the converted string.
 */
string RequestHandler::JsonToString(Json::Value data)
{
    Json::StyledWriter writer;
    return writer.write(data);
}


/**
 * @brief Convert a string to a JSON value.
 * @param str the string
 * @return the converted JSON value.
 */
Json::Value RequestHandler::StringToJson(string str)
{
    Json::Reader reader;
    Json::Value data;
    reader.parse(str, data);
    return data;
}

//
// Created by kata_ha on 2019-12-17.
//

#include "CSVReader.h"
#include <sstream>

std::vector<std::string> CSVReader::split(const std::string &s, char delim) {
    std::stringstream ss(s);
    std::string item;
    std::vector<std::string> elems;
    while(std::getline(ss, item, delim)){
        elems.push_back(std::move(item));
    }
    return elems;
}
std::vector<std::vector<std::string> > CSVReader::getData()
{
    std::ifstream file(m_fileName);

    std::vector<std::vector<std::string> > dataList;

    std::string line = "";
    // Iterate through each line and split the content using delimiter
    while (getline(file, line))
    {
        std::vector<std::string> vec = split(line, m_delimiter);
        dataList.push_back(vec);
    }
    // Close the File
    file.close();

    return dataList;
}
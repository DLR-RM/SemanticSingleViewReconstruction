//
// Created by kata_ha on 2019-12-17.
//

#ifndef SDFGEN_CSVREADER_H
#define SDFGEN_CSVREADER_H

#include <fstream>
#include <vector>
#include <iterator>
#include <string>
#include <algorithm>
#include <iostream>

class CSVReader {
public:
    CSVReader(std::string filename, char delm = ',') :
            m_fileName(std::move(filename)), m_delimiter(delm)
    { }

    // Function to fetch data from a CSV File
    std::vector<std::vector<std::string> > getData();

    // Function to split a string with a certain delimiter
    std::vector<std::string> split(const std::string &s, char delim);

private:
    std::string m_fileName;
    char m_delimiter;
};


#endif //SDFGEN_CSVREADER_H

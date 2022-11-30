//
// Created by kata_ha on 2019-10-14.
//

#ifndef SDFGEN_SUNCGREADER_H
#define SDFGEN_SUNCGREADER_H


#include "../geom/Polygon.h"
#include "rapidjson/document.h"
#include <rapidjson/istreamwrapper.h>
#include "ObjReader.h"
#include "PolygonReader.h"
#include <map>

#include <utility>

class SuncgLoader: public PolygonReader {

public:

    explicit SuncgLoader(std::string suncgPath, const std::string& modelCategoryMappingPath, const std::string& csvNYUPath, const std::string& housePath): m_suncgDir(suncgPath), m_housePath(housePath) {
        load_model_class_information(modelCategoryMappingPath, csvNYUPath);
    };

    void read() override;

    void addInfoToHdf5Container(int64_t fileId) override;

private:

    std::string m_suncgDir;
    std::string m_housePath;
    std::map<std::string, int> m_modelIDToObjectClass;
    std::map<std::string, std::string> m_modelIDToClassName;
    std::vector<double> default_transformation_array = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

    void load_model_class_information(const std::string& csvFileName, const std::string& csvNYUClasses);

    void load_objects(const rapidjson::Value &node, const std::vector<double>& transformation);
    void load_room(const rapidjson::Value &node, const std::vector<double>& transformation, const std::string& houseId);
    void load_ground(const rapidjson::Value &node, const std::vector<double>& transformation, const std::string& houseId);
    void loadBox(const rapidjson::Value &node, const std::vector<double>& transformation, const std::string& houseId, const int objectClass);
    void load_obj_file(const std::string& fileName, const std::vector<double>& transformation, const int objectClass);
    int getClassIdForModelName(std::string modelName);
    static std::vector<double> get_transformation_array(const rapidjson::Value &transform);

};

#endif //SDFGEN_SUNCGREADER_H

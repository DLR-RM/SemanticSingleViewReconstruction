//
// Created by kata_ha on 2019-10-14.
//

#include "SuncgReader.h"
#include "../util/CSVReader.h"
#include "../geom/Cube.h"
#include "../util/ObjWriter.h"
#include "../util/Hdf5Writer.h"
#include <set>



int SuncgLoader::getClassIdForModelName(std::string modelName){
    // make the modelName lower case:
    std::for_each(modelName.begin(), modelName.end(), [](char & c) {c = (char) std::tolower(c);});
    const auto it = m_modelIDToObjectClass.find(modelName);
    if(it != m_modelIDToObjectClass.end()){
        return it->second;
    }
    printError("This model name was not found: " << modelName);
    return -1;
}

std::vector<double> SuncgLoader::get_transformation_array(const rapidjson::Value &transform) {

    std::vector<double> transformation_array;

    for (rapidjson::SizeType i = 0; i < transform.Size(); i++) // Uses SizeType instead of size_t
        transformation_array.push_back(transform[i].GetDouble());

    return transformation_array;

}

void SuncgLoader::read(){
    m_polygons.clear();

    std::ifstream house(m_housePath);
    rapidjson::IStreamWrapper isw{house};

    rapidjson::Document houseJson{};

    houseJson.ParseStream(isw);

    std::string houseId = houseJson["id"].GetString();
    const rapidjson::Value &levels = houseJson["levels"];

    for (rapidjson::Value::ConstValueIterator level_itr = levels.Begin(); level_itr != levels.End(); ++level_itr) {

        const rapidjson::Value &level = *level_itr;
        const rapidjson::Value &nodes = level["nodes"];

        for (rapidjson::Value::ConstValueIterator node_itr = nodes.Begin(); node_itr != nodes.End(); ++node_itr) {

            const rapidjson::Value &node = *node_itr;

            std::string object = "Object";
            std::string room = "Room";
            std::string ground = "Ground";
            std::string box = "Box";

            std::vector<double> transformation = default_transformation_array;
            if (node.HasMember("transform")) {
                transformation = get_transformation_array(node["transform"]);
            }

            std::string currentType = node["type"].GetString();
            if(object == currentType){
                load_objects(node, transformation);
            }else if(room == currentType){
                load_room(node, transformation, houseId);
            }else if(ground == currentType){
                load_ground(node, transformation, houseId);
            }else if(box == currentType){
                loadBox(node, transformation, houseId, getClassIdForModelName("box"));
            }
        }
    }

}

void SuncgLoader::load_objects(const rapidjson::Value &node, const std::vector<double> &transformation) {

    std::string modelId = "Empty";
    std::string objFilePath;

    if (node.HasMember("modelId"))
        modelId = node["modelId"].GetString();

    if (!node.HasMember("state") || node["state"].GetInt() == 0) {
        objFilePath = m_suncgDir + "/object/" + modelId + "/" + modelId + ".obj";
    } else {
        objFilePath = m_suncgDir + "/object/" + modelId + "/" + modelId + "_0.obj";
    }

    load_obj_file(objFilePath, transformation, getClassIdForModelName(modelId));
}

void SuncgLoader::load_ground(const rapidjson::Value &node, const std::vector<double> &transformation,
                              const std::string &houseId) {

    std::string modelId = "Empty";
    if (node.HasMember("modelId"))
        modelId = node["modelId"].GetString();

    std::string groundObjFilePath = m_suncgDir + "/room/" + houseId + "/" + modelId + "f.obj";

    load_obj_file(groundObjFilePath, transformation, getClassIdForModelName("Floor"));
}

void SuncgLoader::loadBox(const rapidjson::Value &node, const std::vector<double> &transformation,
                              const std::string &houseId, const int objectClass) {

    dPoint scale = {1, 1, 1};
    if(node.HasMember("dimensions")){
        const rapidjson::Value& dimension = node["dimensions"];
        for(rapidjson::SizeType i = 0; i < dimension.Size(); i++){ // Uses SizeType instead of size_t
            scale[i] = dimension[i].GetDouble();
        }
    }
    Cube cube(scale, transformation, objectClass);
    m_polygons.insert(m_polygons.end(), cube.getPolygons().begin(), cube.getPolygons().end());
}

void SuncgLoader::load_room(const rapidjson::Value &node, const std::vector<double> &transformation,
                            const std::string &houseId) {

    std::string modelId = "Empty";
    if (node.HasMember("modelId"))
        modelId = node["modelId"].GetString();

    if (!node.HasMember("hideFloor") || node["hideFloor"].GetInt() != 1) {

        std::string floorObjFilePath = m_suncgDir + "/room/" + houseId + "/" + modelId + "f.obj";
        load_obj_file(floorObjFilePath, transformation, getClassIdForModelName("Floor"));

    }

    if (!node.HasMember("hideCeiling") || node["hideCeiling"].GetInt() != 1) {

        std::string ceilingObjFilePath = m_suncgDir + "/room/" + houseId + "/" + modelId + "c.obj";
        load_obj_file(ceilingObjFilePath, transformation, getClassIdForModelName("Ceiling"));

    }

    if (!node.HasMember("hideWalls") || node["hideWalls"].GetInt() != 1) {

        std::string wallsObjFilePath = m_suncgDir + "/room/" + houseId + "/" + modelId + "w.obj";
        load_obj_file(wallsObjFilePath, transformation, getClassIdForModelName("Wall"));

    }

}

void SuncgLoader::load_obj_file(const std::string &fileName, const std::vector<double> &transformation, const int objectClass) {

    ObjReader objReader;
    objReader.read(fileName, transformation, objectClass);
    m_polygons.insert(m_polygons.end(), objReader.getPolygon().begin(), objReader.getPolygon().end());

}

void SuncgLoader::load_model_class_information(const std::string& csvFileName, const std::string& csvNYUClasses) {

    CSVReader reader(csvFileName);
    std::map<std::string,int> NYUClassToObjectClass;
    std::map<std::string,int>::iterator it;
    std::map<std::string, std::string>::iterator itClassName;

    std::vector<std::vector<std::string> > dataList = reader.getData();

    for(std::vector<std::string> vec : dataList){
        if(vec[5] == "nyuv2_40class"){
            // skip the first row
            continue;;
        }

        itClassName = m_modelIDToClassName.find(vec[5]);
        if(itClassName == m_modelIDToClassName.end()){
            m_modelIDToClassName.insert(std::pair<std::string, std::string>(vec[1], vec[5]));
        }

    }

    // map the model id names to the nyu classes
    CSVReader nyuReader(csvNYUClasses);
    std::vector<std::vector<std::string> > dataNYUList = nyuReader.getData();
    for(const auto& vec: dataNYUList){
        if(vec[0] == "id"){
            // skip the first line
            continue;
        }
        bool found = false;
        for(const auto& ele: m_modelIDToClassName){
            std::string nameCopy(ele.second);
            std::for_each(nameCopy.begin(), nameCopy.end(), [](char & c) {c = (char) std::tolower(c);});
            if(nameCopy == vec[1]){
                std::string usedNameCopy(ele.first);
                std::for_each(usedNameCopy.begin(), usedNameCopy.end(), [](char & c) {c = (char) std::tolower(c);});
                m_modelIDToObjectClass.insert(std::pair<std::string, int>(usedNameCopy, std::stoi(vec[0])));
                found = true;
            }
        }
        if(!found){
            std::string nameCopy(vec[1]);
            std::for_each(nameCopy.begin(), nameCopy.end(), [](char & c) {c = (char) std::tolower(c);});
            m_modelIDToObjectClass.insert(std::pair<std::string, int>(nameCopy, std::stoi(vec[0])));
        }
    }
    // the classes towel, paper and bag do not appear in the dataset
    // the box label can not be identified, the box class is not identifiable, therefore it is void
}

void SuncgLoader::addInfoToHdf5Container(int64_t fileId){
    Hdf5Writer::writeStringToFileId(fileId, "used_json_path", m_housePath);
    Hdf5Writer::writeStringToFileId(fileId, "used_suncg_path", m_suncgDir);
    Hdf5Writer::writeStringToFileId(fileId, "dataset", std::string("suncg"));
}

#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <iostream>
#include <iomanip>

namespace csv {

class Document {
public:
    Document(const std::string &filePath, char delimiter = ',')
    {
        std::ifstream file(filePath);
        if (!file.is_open()) {
            printf("Open file failed. path = %s", filePath.c_str());
        }

        std::string line;
        std::getline(file, line);
        std::istringstream headerStream(line);
        std::string header;
        size_t index = 0;
        while (std::getline(headerStream, header, delimiter)) {
            columnNameToIndex.insert_or_assign(header, index++);
        }

        while (std::getline(file, line)) {
            std::istringstream lineStream(line);
            std::string cell;
            std::vector<std::string> row;
            row.reserve(index);
            while (std::getline(lineStream, cell, delimiter)) {
                row.push_back(cell);
            }
            data.push_back(row);
        }
        file.close();
    }

    size_t GetRowCount() const
    {
        return data.size();
    }

    template <typename T = std::string>
    T GetCell(const std::string &columnName, const size_t rowIdx) const;

    template <>
    std::string GetCell<std::string>(const std::string &columnName, const size_t rowIdx) const
    {
        return data[rowIdx][columnNameToIndex.at(columnName)];
    }

    template <>
    int GetCell<int>(const std::string &columnName, const size_t rowIdx) const
    {
        return std::stoi(data[rowIdx][columnNameToIndex.at(columnName)]);
    }

    template <>
    uint32_t GetCell<uint32_t>(const std::string &columnName, const size_t rowIdx) const
    {
        return std::stoul(data[rowIdx][columnNameToIndex.at(columnName)]);
    }

    template <>
    size_t GetCell<size_t>(const std::string &columnName, const size_t rowIdx) const
    {
        return std::stoull(data[rowIdx][columnNameToIndex.at(columnName)]);
    }

    template <>
    float GetCell<float>(const std::string &columnName, const size_t rowIdx) const
    {
        return std::stof(data[rowIdx][columnNameToIndex.at(columnName)]);
    }

    template <>
    double GetCell<double>(const std::string &columnName, const size_t rowIdx) const
    {
        return std::stod(data[rowIdx][columnNameToIndex.at(columnName)]);
    }

private:
    std::map<std::string, size_t> columnNameToIndex;
    std::vector<std::vector<std::string>> data;
};
}
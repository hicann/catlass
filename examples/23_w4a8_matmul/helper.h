#pragma once

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cmath>

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stdout, "[ERROR]  " fmt "\n", ##args)

#define ACL_CHECK(status)                                                                    \
    do {                                                                                     \
        aclError error = status;                                                             \
        if (error != ACL_ERROR_NONE) {                                                       \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << error << std::endl;  \
        }                                                                                    \
    } while (0)

inline bool ReadFile(const std::string &filePath, void *buffer, size_t bufferSize)
{
    struct stat sBuf;
    int fileStatus = stat(filePath.data(), &sBuf);
    if (fileStatus == -1) {
        ERROR_LOG("Failed to get file");
        return false;
    }
    if (S_ISREG(sBuf.st_mode) == 0) {
        ERROR_LOG("%s is not a file, please enter a file.", filePath.c_str());
        return false;
    }

    std::ifstream file;
    file.open(filePath, std::ios::binary);
    if (!file.is_open()) {
        ERROR_LOG("Open file failed. path = %s.", filePath.c_str());
        return false;
    }

    std::filebuf *buf = file.rdbuf();
    size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
    if (size == 0) {
        ERROR_LOG("File size is 0");
        file.close();
        return false;
    }
    if (size > bufferSize) {
        ERROR_LOG("File size is larger than buffer size.");
        file.close();
        return false;
    }
    buf->pubseekpos(0, std::ios::in);
    buf->sgetn(static_cast<char *>(buffer), size);
    file.close();
    return true;
}

inline bool WriteFile(const std::string &filePath, const void *buffer, size_t size)
{
    if (buffer == nullptr) {
        ERROR_LOG("Write file failed. Buffer is nullptr.");
        return false;
    }

    std::ofstream fd(filePath, std::ios::binary);
    if (!fd) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    fd.write(static_cast<const char *>(buffer), size);
    if (!fd) {
        ERROR_LOG("Write file failed.");
        return false;
    }
    return true;
}

template<typename RET_TYPE, typename REF_TYPE>
void CompareResults(RET_TYPE *result, REF_TYPE *except, uint32_t M, uint32_t K, uint32_t N) {
    size_t errorCount = 0;
    REF_TYPE err = pow(2, -8);
    if (K > 2048) {
        err = pow(2, -7);
    }
    for (size_t i = 0; i < M * N; ++i) {
        REF_TYPE a = static_cast<REF_TYPE>(result[i]);
        REF_TYPE e = except[i];
        REF_TYPE diff = std::fabs(a - e);
        if (std::isnan(diff) || diff > err * std::max(1.0f, std::fabs(e))) {
            errorCount++;
        }
    }
    std::cout << "error count: " << errorCount << std::endl;
    if (errorCount > 0) {
        std::cout << "FAILED" << std::endl;
    } else {
        std::cout << "SUCCESS" << std::endl;
    }
}

#include <vector>
#include <iomanip>

using namespace std;

constexpr uint32_t SOFTMAX_DEFAULT_BLK_SIZE = 32;
constexpr uint32_t SOFTMAX_TMPBUFFER_COUNT = 2;
constexpr uint32_t SOFTMAX_HALF_SIZE = 2;
constexpr uint32_t SOFTMAX_FLOAT_SIZE = 4;
constexpr uint32_t SOFTMAXGRAD_TMPBUFFER_COUNT = 3;
constexpr uint32_t BASIC_TILE_NUM = SOFTMAX_DEFAULT_BLK_SIZE / SOFTMAX_FLOAT_SIZE;
constexpr uint32_t SOFTMAX_BASICBLOCK_MIN_SIZE = 256;
constexpr uint32_t SOFTMAX_BASICBLOCK_UNIT = 64;

struct SoftMaxTilingLocal
{
    uint32_t srcM = 0;
    uint32_t srcK = 0;
    uint32_t srcSize = 0;
    uint32_t outMaxM = 0;
    uint32_t outMaxK = 0;
    uint32_t outMaxSize = 0;
    uint32_t splitM = 0;
    uint32_t splitK = 0;
    uint32_t splitSize = 0;
    uint32_t reduceM = 0;
    uint32_t reduceK = 0;
    uint32_t reduceSize = 0;
    uint32_t rangeM = 0;
    uint32_t tailM = 0;
    uint32_t tailSplitSize = 0;
    uint32_t tailReduceSize = 0;
};

template<typename T>
inline vector<uint32_t> GetLastAxisShapeND(const vector<T> srcShape)
{
    std::vector<uint32_t> ret;
    std::vector<int64_t> shapeDims(srcShape.begin(), srcShape.end());
    uint32_t calculateSize = 1;
    for (uint32_t i = 0; i < shapeDims.size(); i++) {
        calculateSize *= shapeDims[i];
    }

    const uint32_t srcK = shapeDims.back();
    uint32_t srcM = calculateSize / srcK;
    ret = { srcM, srcK };
    return ret;
}

inline void AdjustToBasicBlockBaseM(uint32_t& baseM, const uint32_t srcM, const uint32_t srcK)
{
    if (baseM > BASIC_TILE_NUM && srcM % BASIC_TILE_NUM == 0 && srcK % SOFTMAX_BASICBLOCK_UNIT == 0) { // basicblock
        baseM = baseM / BASIC_TILE_NUM * BASIC_TILE_NUM;
        while (srcM % baseM != 0) {
            baseM -= BASIC_TILE_NUM;
        }
        // max repeat only support 255
        while (baseM * srcK >= SOFTMAX_BASICBLOCK_UNIT * SOFTMAX_BASICBLOCK_MIN_SIZE) {
            baseM = baseM / SOFTMAX_HALF_SIZE;
        }
    }
}

void SoftMaxTilingFunc(const vector<uint32_t>& srcShape, const uint32_t dataTypeSize, const uint32_t localWorkSpaceSize,
    SoftMaxTiling& softmaxTiling)
{
    std::vector<uint32_t> retVec = GetLastAxisShapeND(srcShape);
    if (retVec.size() <= 1 || dataTypeSize == 0) {
        return;
    }
    const uint32_t elementNumPerBlk = SOFTMAX_DEFAULT_BLK_SIZE / dataTypeSize;
    const uint32_t workLocalSize = localWorkSpaceSize / SOFTMAX_FLOAT_SIZE;
    const uint32_t srcK = retVec[1];
    const uint32_t srcM = retVec[0];
    uint32_t baseM = std::min(workLocalSize / (elementNumPerBlk + srcK + SOFTMAX_BASICBLOCK_UNIT), srcM);
    if (baseM < srcM && baseM > BASIC_TILE_NUM) {
        baseM = baseM / BASIC_TILE_NUM * BASIC_TILE_NUM;
    }

    AdjustToBasicBlockBaseM(baseM, srcM, srcK);

    softmaxTiling.srcM = srcM;
    softmaxTiling.srcK =srcK;
    softmaxTiling.srcSize = srcM * srcK;

    softmaxTiling.outMaxM = srcM;
    softmaxTiling.outMaxK = elementNumPerBlk;
    softmaxTiling.outMaxSize = srcM * elementNumPerBlk;

    softmaxTiling.splitM = baseM;
    softmaxTiling.splitK = srcK;
    softmaxTiling.splitSize = baseM * srcK;

    softmaxTiling.reduceM = baseM;
    softmaxTiling.reduceK = elementNumPerBlk;
    softmaxTiling.reduceSize = baseM * elementNumPerBlk;

    const uint32_t range = srcM / baseM;
    uint32_t tail = srcM % baseM;
    softmaxTiling.rangeM = range;
    softmaxTiling.tailM = tail;

    softmaxTiling.tailSplitSize = tail * srcK;
    softmaxTiling.tailReduceSize = tail * elementNumPerBlk;
}

void SoftMaxGradTilingFunc(const vector<int64_t>& srcShape, const uint32_t dataTypeSize, const uint32_t localWorkSpaceSize,
    SoftMaxTiling& softmaxGradTiling)
{
    std::vector<uint32_t> retVec = GetLastAxisShapeND(srcShape);
    if (retVec.size() <= 1 || dataTypeSize == 0) {
        return;
    }
    const uint32_t elementNumPerBlk = SOFTMAX_DEFAULT_BLK_SIZE / dataTypeSize;
    const uint32_t workLocalSize = localWorkSpaceSize / SOFTMAX_FLOAT_SIZE;
    const uint32_t srcK = retVec[1];
    const uint32_t srcM = retVec[0];
    uint32_t baseM = 0;
    if (dataTypeSize == SOFTMAX_HALF_SIZE) {
        baseM = workLocalSize /
            (elementNumPerBlk * SOFTMAX_TMPBUFFER_COUNT + srcK * SOFTMAXGRAD_TMPBUFFER_COUNT + SOFTMAX_BASICBLOCK_UNIT);
    } else {
        baseM = workLocalSize / (elementNumPerBlk + srcK + SOFTMAX_BASICBLOCK_UNIT);
    }

    baseM = std::min(baseM, srcM);
    if (baseM < srcM && baseM > BASIC_TILE_NUM) {
        baseM = baseM / BASIC_TILE_NUM * BASIC_TILE_NUM;
    }

    AdjustToBasicBlockBaseM(baseM, srcM, srcK);

    softmaxGradTiling.srcM = srcM;
    softmaxGradTiling.srcK = srcK;
    softmaxGradTiling.srcSize = srcM * srcK;

    softmaxGradTiling.outMaxM = srcM;
    softmaxGradTiling.outMaxK = elementNumPerBlk;
    softmaxGradTiling.outMaxSize = srcM * elementNumPerBlk;

    softmaxGradTiling.splitM = baseM;
    softmaxGradTiling.splitK = srcK;
    softmaxGradTiling.splitSize = baseM * srcK;

    softmaxGradTiling.reduceM = baseM;
    softmaxGradTiling.reduceK = elementNumPerBlk;
    softmaxGradTiling.reduceSize = baseM * elementNumPerBlk;

    uint32_t range = srcM / baseM;
    const uint32_t tail = srcM % baseM;
    softmaxGradTiling.rangeM = range;
    softmaxGradTiling.tailM = tail;

    softmaxGradTiling.tailSplitSize = tail * srcK;
    softmaxGradTiling.tailReduceSize = tail * elementNumPerBlk;
}

void printSoftmaxTilingData(SoftMaxTiling &softmaxTilingData) {
    cout << "softmaxTilingData srcM:  " << softmaxTilingData.srcM << endl;
    cout << "softmaxTilingData srcK:  " << softmaxTilingData.srcK << endl;
    cout << "softmaxTilingData srcSize:  " << softmaxTilingData.srcSize << endl;
    cout << "softmaxTilingData outMaxK:  " << softmaxTilingData.outMaxK << endl;
    cout << "softmaxTilingData outMaxM:  " << softmaxTilingData.outMaxM << endl;
    cout << "softmaxTilingData: outMaxSize  " << softmaxTilingData.outMaxSize << endl;
    cout << "softmaxTilingData: splitM  " << softmaxTilingData.splitM << endl;
    cout << "softmaxTilingData: splitK  " << softmaxTilingData.splitK << endl;
    cout << "softmaxTilingData: splitSize  " << softmaxTilingData.splitSize << endl;
    cout << "softmaxTilingData: reduceM  " << softmaxTilingData.reduceM << endl;
    cout << "softmaxTilingData: reduceK  " << softmaxTilingData.reduceK << endl;
    cout << "softmaxTilingData: reduceSize  " << softmaxTilingData.reduceSize << endl;
    cout << "softmaxTilingData: rangeM  " << softmaxTilingData.rangeM << endl;
    cout << "softmaxTilingData: tailM  " << softmaxTilingData.tailM << endl;
    cout << "softmaxTilingData: tailSplitSize  " << softmaxTilingData.tailSplitSize << endl;
    cout << "softmaxTilingData: tailReduceSize  " << softmaxTilingData.tailReduceSize << endl;
}
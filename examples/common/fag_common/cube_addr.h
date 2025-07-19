#ifndef __CUBEAddr_H__
#define __CUBEAddr_H__

#include "common_header.h"
using namespace AscendC;

class CubeAddr {

public:
    int32_t batch;
    int32_t headNum;
    int32_t headDim;
    int32_t g;
    int32_t n2;
    int32_t coreId = 0;

    int32_t coreSegmentBlockNum = 0;
    int32_t SegmentBlockNum = 0;
    int32_t blockNum = 0;

    int32_t batchIdx;
    int32_t headNumIdx;
    int32_t s1Idx;
    int32_t s2Idx;
    int32_t s1;
    int32_t s2;
    int32_t s1BlockNum;
    int32_t s1TailLength;
    int32_t s2BlockNum;
    int32_t s2TailLength;
    int32_t s1GuardInterval;

    int32_t limit;
    int32_t coreNum;
    int32_t roundId;

    int32_t overFlag = 1;
    int32_t lastBatchSum = 0; 

    struct CubeAddrInfo * globalCubeAddr;

    __gm__ uint8_t *actual_seq_qlen_addr;
    __gm__ uint8_t *actual_seq_kvlen_addr;
    __gm__ uint8_t *q_gm_addr;
    __gm__ uint8_t *k_gm_addr;
    __gm__ uint8_t *v_gm_addr;
    __gm__ uint8_t *dy_gm_addr;
    __gm__ uint8_t *user_gm_addr;

    __aicore__ uint64_t getSeqRealLength(int32_t sIdx, int32_t len, int32_t s_block_num, int32_t s_tail) {
        if (s_tail > 0 && (sIdx + len == s_block_num)) {
            return (len - 1) * 128 + s_tail;
        } else {
            return len * 128;
        }
    }
    
    __aicore__ int64_t getTotalLen(int32_t i) {
        int64_t actualTotalSeqQlen = ((__gm__ int64_t *)actual_seq_qlen_addr)[i];
        return actualTotalSeqQlen;
    }

    __aicore__ uint64_t getLeftAddr(int32_t batchIdx, int32_t headNumIdx, int32_t s1, int32_t s1Idx, int32_t headDim) {
        return lastBatchSum * headNum * headDim + (s1Idx * 128 * headNum + headNumIdx) * headDim;
    }

    __aicore__ uint64_t getRightAddr(int32_t batchIdx, int32_t headNumIdx, int32_t s2, int32_t s2Idx, int32_t headDim) {
        return lastBatchSum * n2 * headDim + (s2Idx * 128 * n2 + (headNumIdx / g)) * headDim;
    }

    __aicore__ uint64_t getOutAddr(int32_t workspacePos) {
        return workspacePos * 128 * 128;
    }

    __aicore__ int32_t addr_mapping(struct CubeAddrInfo * cubeAddrInfo) {
        globalCubeAddr = cubeAddrInfo;
        globalCubeAddr->blockLength = 0;

        int32_t loopCnt = 0;
        while (overFlag) {
            int32_t guardLen = s1Idx + s1GuardInterval - s2Idx;
            int32_t reserve = limit - blockNum;
            int32_t realLenAlign = (reserve + s1GuardInterval - 1) / s1GuardInterval;
            if (realLenAlign >= guardLen) {
                if (coreSegmentBlockNum % coreNum == coreId) {
                    int32_t index = globalCubeAddr->blockLength;
                    globalCubeAddr->addrInfo[index].left = getLeftAddr(batchIdx, headNumIdx, s1, s1Idx, headDim);
                    globalCubeAddr->addrInfo[index].right = getRightAddr(batchIdx, headNumIdx, s2, s2Idx, headDim);
                    globalCubeAddr->addrInfo[index].out = getOutAddr(blockNum);
                    globalCubeAddr->addrInfo[index].ky = getSeqRealLength(s1Idx, s1GuardInterval, s1BlockNum, s1TailLength);
                    globalCubeAddr->addrInfo[index].kx = getSeqRealLength(s2Idx, guardLen, s2BlockNum, s2TailLength);
                    globalCubeAddr->addrInfo[index].lowerLeft = 1;
                    globalCubeAddr->addrInfo[index].upperRight = (s1GuardInterval % 2);
                    globalCubeAddr->blockLength ++;
                }
                blockNum += s1GuardInterval * guardLen - (s1GuardInterval + 1) % 2;
                s1Idx += s1GuardInterval;
                s2Idx = 0;
                if (s1Idx == s1BlockNum - 1) {
                    s1GuardInterval = 1;
                }
            } else {
                int32_t realLen = (reserve / s1GuardInterval);
                if (coreSegmentBlockNum % coreNum == coreId) {
                    int32_t index = globalCubeAddr->blockLength;
                    globalCubeAddr->addrInfo[index].left = getLeftAddr(batchIdx, headNumIdx, s1, s1Idx, headDim);
                    globalCubeAddr->addrInfo[index].right = getRightAddr(batchIdx, headNumIdx, s2, s2Idx, headDim);
                    globalCubeAddr->addrInfo[index].out = getOutAddr(blockNum);
                    globalCubeAddr->addrInfo[index].ky = getSeqRealLength(s1Idx, s1GuardInterval, s1BlockNum, s1TailLength);
                    globalCubeAddr->addrInfo[index].kx = getSeqRealLength(s2Idx, realLen, s2BlockNum, s2TailLength);
                    globalCubeAddr->addrInfo[index].lowerLeft = 1;
                    globalCubeAddr->addrInfo[index].upperRight = 1;
                    globalCubeAddr->blockLength ++;
                }
                blockNum += s1GuardInterval * realLen;
                s2Idx += realLen;
            }
            SegmentBlockNum++;
            if ((s1Idx == s1BlockNum) && (batchIdx == batch - 1) && (headNumIdx == headNum - 1)) {
                overFlag = 0;
                break;
            }

            if (s1Idx == s1BlockNum) {
                if (headNumIdx == headNum - 1) {
                    lastBatchSum = getTotalLen(batchIdx);
                    batchIdx++;
                    headNumIdx = 0;
                    s1 = getSeqLen(batchIdx);
                    s2 = getSeqLen(batchIdx);
                    s1BlockNum = (s1 + 127) / 128;
                    s1TailLength = s1 % 128;
                    s2BlockNum = (s2 + 127) / 128;
                    s2TailLength = s2 % 128;
                } else {
                    headNumIdx++;
                }
                s1Idx = 0;
                s2Idx = 0;
                s1GuardInterval = (s1BlockNum == 1) ? 1 : 2;
            }

            if (blockNum >= 15) {
                coreSegmentBlockNum++;
                SegmentBlockNum = 0;
                blockNum = 0;
                if (coreSegmentBlockNum == roundId * coreNum) {
                    break;
                }
            }
        }

        roundId++;
        return overFlag;
    }

    __aicore__ int64_t getSeqLen(int32_t i) {
        int64_t actualSeqQlen;
        if (i == 0) {
            actualSeqQlen = ((__gm__ int64_t *)actual_seq_qlen_addr)[0];
        } else {
            actualSeqQlen = ((__gm__ int64_t *)actual_seq_qlen_addr)[i] - ((__gm__ int64_t *)actual_seq_qlen_addr)[i - 1];
        }
        return actualSeqQlen;
    }

    __aicore__ void init(int32_t batchIn, int32_t headNumIn, int32_t gIn, int32_t headDimIn, uint32_t coreIdx, 
        __gm__ uint8_t *actual_seq_qlen, __gm__ uint8_t *actual_seq_kvlen, uint32_t totalCoreNum) {
        
        batch = batchIn;
        headNum = headNumIn;
        g = gIn;
        headDim = headDimIn;
        n2 = headNum / g;

        actual_seq_qlen_addr = actual_seq_qlen;
        actual_seq_kvlen_addr = actual_seq_kvlen;

        coreSegmentBlockNum = 0;
        SegmentBlockNum = 0;
        blockNum = 0;

        batchIdx = 0;
        headNumIdx = 0;
        s1Idx = 0;
        s2Idx = 0;        
        s1 = getSeqLen(batchIdx);
        s2 = getSeqLen(batchIdx);
        s1BlockNum = (s1 + 127) / 128;
        s1TailLength = s1 % 128;
        s2BlockNum = (s2 + 127) / 128;
        s2TailLength = s2 % 128;
        s1GuardInterval = (s1BlockNum == 1) ? 1 : 2;

        limit = 16;
        coreNum = totalCoreNum;
        coreId = coreIdx;

        roundId = 1;
        overFlag = 1;
        lastBatchSum = 0;
    }
};
#endif
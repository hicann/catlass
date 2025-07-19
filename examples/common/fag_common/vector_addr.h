#ifndef __VECTORAddr_H__
#define __VECTORAddr_H__

#include "common_header.h"
using namespace AscendC;

class VectorAddr {

public:
    int32_t batch;
    int32_t headNum;
    int32_t headDim;
    int32_t g;
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

    struct VecAddrInfo * globalVecAddr;

    __gm__ uint8_t *actual_seq_qlen_addr;
    __gm__ uint8_t *actual_seq_kvlen_addr;
    __gm__ uint8_t *q_gm_addr;
    __gm__ uint8_t *k_gm_addr;
    __gm__ uint8_t *v_gm_addr;
    __gm__ uint8_t *dy_gm_addr;
    __gm__ uint8_t *user_gm_addr;

    __aicore__ uint64_t getSeqRealLength(int32_t sIdx, int32_t len, int32_t s_block_num, int32_t s_tail) {
        if (s_tail == 0) {
            return len * 128;
        } else {
            if (sIdx + len == s_block_num) {
                return (len - 1) * 128 + s_tail;
            } else {
                return len * 128;
            }
        }
    }
    
    __aicore__ int64_t getTotalLen(int32_t i) {
        int64_t actualTotalSeqQlen = ((__gm__ int64_t *)actual_seq_qlen_addr)[i];
        return actualTotalSeqQlen;
    }

    __aicore__ uint64_t getLeftAddr(int32_t batchIdx, int32_t headNumIdx, int32_t s1, int32_t s1Idx, int32_t headDim) {
        if (batchIdx == 0) {
            return (s1Idx * 128 * headNum + headNumIdx) * headDim;
        } else {
            return getTotalLen(batchIdx - 1) * headNum * headDim + (s1Idx * 128 * headNum + headNumIdx) * headDim;
        }
    }

    __aicore__ uint64_t getRightAddr(int32_t batchIdx, int32_t headNumIdx, int32_t s2, int32_t s2Idx, int32_t headDim) {
        if (batchIdx == 0) {
            return (s2Idx * 128 * (headNum / g) + (headNumIdx / g)) * headDim;
        } else {
            return getTotalLen(batchIdx - 1) * (headNum / g) * headDim + (s2Idx * 128 * (headNum / g) + (headNumIdx / g)) * headDim;
        }
    }

    __aicore__ uint64_t getOutAddr(int32_t workspacePos) {
        return workspacePos * 128 * 128;
    }

    __aicore__ __inline__ void getOffset(VecBlockInfo &vecPhyAddr, int32_t blockId, int row, int col) 
    {
        vecPhyAddr.batchIdx = batchIdx;
        vecPhyAddr.headNumIdx = headNumIdx;
        vecPhyAddr.S1Idx = s1Idx + row;
        vecPhyAddr.S2Idx = s2Idx + col;
        vecPhyAddr.n2Idx = headNumIdx / g;
        vecPhyAddr.gIdx = headNumIdx % g;
        vecPhyAddr.offset = blockId * 128 * 128;
        
        vecPhyAddr.lengthy = 128;
        if ((row + s1Idx == s1BlockNum - 1) && s1TailLength > 0) {
            vecPhyAddr.lengthy = s1TailLength;
        } 

        vecPhyAddr.lengthx = 128;
        if ((col + s2Idx == s2BlockNum - 1) && s2TailLength > 0) {
            vecPhyAddr.lengthx = s2TailLength;
        }
    }

    __aicore__ int32_t addr_mapping(struct VecAddrInfo * vecAddrInfo) {
        globalVecAddr = vecAddrInfo;
        globalVecAddr->blockLength = 0;

        int32_t loopCnt = 0;
        while (overFlag) {
            int32_t guardLen = s1Idx + s1GuardInterval - s2Idx;
            int32_t reserve = limit - blockNum;

            int32_t realLenAlign = (reserve + s1GuardInterval - 1) / s1GuardInterval;
            if (realLenAlign >= guardLen) {
                if (coreSegmentBlockNum % coreNum == coreId) {
                    int32_t blockId = blockNum;
                    for (int x = 0; x < guardLen; x++){
                        for (int y = 0; y < s1GuardInterval; y++){
                                if (s1Idx + y >= s2Idx + x){
                                    getOffset(globalVecAddr->VecBlkInfo[blockId], blockId, y, x);
                                    blockId ++;
                                }
                        }
                    }
                    globalVecAddr->blockLength = blockNum + s1GuardInterval * guardLen - (s1GuardInterval + 1) % 2;
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
                    int32_t blockId = blockNum;
                    for (int x = 0; x < realLen; x++){
                        for (int y = 0; y < s1GuardInterval; y++) {
                                if (s1Idx + y >= s2Idx + x) {
                                    getOffset(globalVecAddr->VecBlkInfo[blockId], blockId, y, x);
                                    blockId ++;
                                }
                        }
                    }
                    globalVecAddr->blockLength = blockNum + s1GuardInterval * realLen;
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
    }
};
#endif
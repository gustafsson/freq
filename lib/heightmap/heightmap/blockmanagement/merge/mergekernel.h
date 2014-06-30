#ifndef MERGEKERNEL_H
#define MERGEKERNEL_H

#include "resampletypes.h"
#include "datastorage.h"

typedef DataStorage<float> BlockData;

extern "C"
void blockMerge( BlockData::ptr inBlock,
                 BlockData::ptr outBlock,
                 ResampleArea in_area,
                 ResampleArea out_area );

#endif // MERGEKERNEL_H

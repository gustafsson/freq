#ifndef TOOLS_SUPPORT_HEIGHTMAPPROCESSINGPUBLISHER_H
#define TOOLS_SUPPORT_HEIGHTMAPPROCESSINGPUBLISHER_H

#include "heightmap/tfrmapping.h"
#include "signal/operation.h"
#include "signal/processing/targetmarker.h"

namespace Tools {
namespace Support {

/**
 * @brief The HeightmapProcessingPublisher class should update a processing
 * target depending on which things that are missing in a heightmap block cache
 *
 * It is worth noting that HeightmapProcessingPublisher doesn't depend on any
 * actual worker nor on any signal processing chain. It just asynchronously
 * publishes work prioritization to a target and assumes that there is a worker
 * somewhere that will detect this. That worker mayfetch the required data
 * through some signal processing chain but this publisher doesn't care.
 */
class HeightmapProcessingPublisher
{
public:
    HeightmapProcessingPublisher(
            Signal::Processing::TargetNeeds::Ptr target_needs,
            Heightmap::TfrMapping::Collections collections);

    void update(float t_center, Signal::OperationDesc::Extent x, Signal::UnsignedIntervalType preferred_update_size);

    bool isHeightmapDone() const;
    bool failedAllocation() const;

private:
    Signal::Processing::TargetNeeds::Ptr   target_needs_;
    Heightmap::TfrMapping::Collections      collections_;
    bool failed_allocation_;

public:
    static void test();
};

} // namespace Support
} // namespace Tools

#endif // TOOLS_SUPPORT_HEIGHTMAPPROCESSINGPUBLISHER_H

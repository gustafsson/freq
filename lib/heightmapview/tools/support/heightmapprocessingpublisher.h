#ifndef TOOLS_SUPPORT_HEIGHTMAPPROCESSINGPUBLISHER_H
#define TOOLS_SUPPORT_HEIGHTMAPPROCESSINGPUBLISHER_H

#include "heightmap/tfrmapping.h"
#include "signal/intervals.h"
#include "signal/processing/targetmarker.h"

#include <QObject>

namespace Tools {
namespace Support {

/**
 * @brief The HeightmapProcessingPublisher class should update a processing
 * target depending on which things that are missing in a heightmap block cache
 *
 * It is worth noting that HeightmapProcessingPublisher doesn't depend on any
 * actual worker nor on any signal processing chain. It just asynchronously
 * publishes work prioritization to a target and assumes that there is a worker
 * somewhere that will detect this. That worker may fetch the required data
 * through some signal processing chain but this publisher doesn't care.
 */
class HeightmapProcessingPublisher: public QObject
{
    Q_OBJECT
public:
    HeightmapProcessingPublisher(
            Signal::Processing::TargetMarker::ptr target_marker,
            Heightmap::TfrMapping::const_ptr tfrmapping,
            float* t_center,
            QObject* parent=0);

public slots:
    void setLastUpdatedInterval( Signal::Interval last_update );
    void update();

private:
    std::shared_ptr<Signal::Processing::TargetNeeds> target_needs_;
    shared_state<Signal::Processing::Dag>   dag_;
    Heightmap::TfrMapping::const_ptr        tfrmapping_;
    float*                                  t_center_;
    Signal::Interval                        last_update_;
    bool                                    failed_allocation_;

    bool isHeightmapDone() const;
    bool failedAllocation() const;

public:
    static void test();
};

} // namespace Support
} // namespace Tools

#endif // TOOLS_SUPPORT_HEIGHTMAPPROCESSINGPUBLISHER_H

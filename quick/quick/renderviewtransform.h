#ifndef RENDERVIEWTRANSFORM_H
#define RENDERVIEWTRANSFORM_H

#include "heightmap/tfrmappings/mergechunk.h"
#include "tfr/transform.h"
#include "tools/support/transformdescs.h"
#include "heightmap/update/updatequeue.h"
#include "heightmap/tfrmapping.h"
#include "signal/processing/targetneeds.h"
#include "tools/rendermodel.h"

class RenderViewTransform
{
public:
    RenderViewTransform(Tools::RenderModel& render_model);

    void receiveSetTransform_Stft();

private:
    Tools::RenderModel& render_model;

    void setBlockFilter(Heightmap::MergeChunkDesc::ptr mcdp, Tfr::TransformDesc::ptr transform_desc);
    void setBlockFilter(Tfr::ChunkFilterDesc::ptr kernel);
    void setBlockFilter(Signal::OperationDesc::ptr o);
};

#endif // RENDERVIEWTRANSFORM_H

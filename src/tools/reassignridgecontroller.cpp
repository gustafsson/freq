#include "reassignridgecontroller.h"

#include "filters/reassign.h"
#include "filters/ridge.h"

namespace Tools {

ReassignRidgeController::ReassignRidgeController()
{
    //        connect(ui->actionTransform_Cwt_reassign, SIGNAL(triggered()), SLOT(receiveSetTransform_Cwt_reassign()));
    //        connect(ui->actionTransform_Cwt_ridge, SIGNAL(triggered()), SLOT(receiveSetTransform_Cwt_ridge()));
}

/*
{
//        connect(ui->actionTransform_Cwt_phase, SIGNAL(triggered()), SLOT(receiveSetTransform_Cwt_phase()));
//        connect(ui->actionTransform_Cwt_reassign, SIGNAL(triggered()), SLOT(receiveSetTransform_Cwt_reassign()));
//        connect(ui->actionTransform_Cwt_ridge, SIGNAL(triggered()), SLOT(receiveSetTransform_Cwt_ridge()));
//        connect(ui->actionTransform_Cwt_weight, SIGNAL(triggered()), SLOT(receiveSetTransform_Cwt_weight()));


//        transform->addActionItem( ui->actionTransform_Cwt_phase );
//        transform->addActionItem( ui->actionTransform_Cwt_reassign );
//        transform->addActionItem( ui->actionTransform_Cwt_ridge );
//        transform->addActionItem( ui->actionTransform_Cwt_weight );
}

#ifdef USE_CUDA
void RenderController::
        receiveSetTransform_Cwt_reassign()
{
    Heightmap::CwtToBlock* cwtblock = new Heightmap::CwtToBlock(&model()->collections, model()->renderer.get());
    cwtblock->complex_info = Heightmap::ComplexInfo_Amplitude_Weighted;

    Signal::PostSink* ps = setBlockFilter( cwtblock );

    ps->filter( Signal::pOperation(new Filters::Reassign()));
}
#endif


void RenderController::
        receiveSetTransform_Cwt_ridge()
{
    Heightmap::CwtToBlock* cwtblock = new Heightmap::CwtToBlock(model()->tfr_map (), model()->renderer.get());
    cwtblock->complex_info = Heightmap::ComplexInfo_Amplitude_Non_Weighted;

    Signal::PostSink* ps = setBlockFilter( cwtblock );

    ps->filter( Signal::pOperation(new Filters::Ridge()));
}
*/
} // namespace Tools

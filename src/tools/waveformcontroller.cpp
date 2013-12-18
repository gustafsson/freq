#include "waveformcontroller.h"
#include "heightmap/tfrmappings/waveformblockfilter.h"
#include "tfr/dummytransform.h"

#include "ui_mainwindow.h"
#include "ui/comboboxaction.h"

namespace Tools {

WaveformController::WaveformController(Tools::RenderController* parent) :
    QObject(parent)
{
    setupGui ();
}


void WaveformController::
        setupGui()
{
    Tools::RenderController* r = render_controller();
    ::Ui::MainWindow* ui = r->getItems();

    showWaveform = ui->actionTransform_Waveform;
    connect(ui->actionTransform_Waveform, SIGNAL(triggered()), SLOT(receiveSetTransform_DrawnWaveform()));
    r->transform->addActionItem( ui->actionTransform_Waveform );

    ui->actionTransform_Waveform->setShortcut ('1' + r->transform->actions ().size ()-1);
}


Tools::RenderController* WaveformController::
        render_controller()
{
    return dynamic_cast<Tools::RenderController*>(parent());
}


void WaveformController::
        receiveSetTransform_DrawnWaveform()
{
    Tools::RenderController* r = render_controller();

    bool enabled = showWaveform->isEnabled();
    ::Ui::MainWindow* ui = r->getItems();


    r->hz_scale->setEnabled( !enabled );
    ui->actionToggle_piano_grid->setVisible( !enabled );
    if (enabled) {
        if (ui->actionToggle_piano_grid->isChecked())
            r->hzmarker->setChecked( false );

        r->linearScale->trigger();
    }


    // Setup the kernel that will take the transform data and create an image
    Heightmap::MergeChunkDesc::Ptr mcdp(new Heightmap::TfrMappings::WaveformBlockFilterDesc);

    // Get a copy of the transform to use
    Tfr::TransformDesc::Ptr t = write1(r->model()->transform_descs ())->getParam<Tfr::DummyTransformDesc>().copy();

    r->setBlockFilter(mcdp, t);
}



} // namespace Tools

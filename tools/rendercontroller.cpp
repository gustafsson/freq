#include "rendercontroller.h"
#include "ui/comboboxaction.h"
#include "ui/displaywidget.h"
#include "ui_mainwindow.h" // Locate actions for binding
#include "ui/mainwindow.h"

// Setting different transforms for rendering
#include "tfr/cwt.h"
#include "heightmap/blockfilter.h"
#include "filters/reassign.h"
#include "filters/ridge.h"

#include <CudaException.h>
#include <cuda.h>

// To create Gui
#include <QToolBar>
#include <QSlider>

using namespace Ui;

namespace Tools
{

RenderController::
        RenderController( RenderView *view )
            :
            model(view->model),
            view(view),
            toolbar_render(0),
            hzmarker(0),
            color(0),
            transform(0)
{
    setupGui();

    // Default values
    float l = model->project->worker.source()->length();
    view->setPosition( std::min(l, 10.f)*0.5f, 0.5f );

    receiveSetTimeFrequencyResolution( 50 );
}


RenderController::
        ~RenderController()
{
    clearCachedHeightmap();
}


void RenderController::
        receiveSetRainbowColors()
{
    model->renderer->color_mode = Heightmap::Renderer::ColorMode_Rainbow;
    view->update();
}


void RenderController::
        receiveSetGrayscaleColors()
{
    model->renderer->color_mode = Heightmap::Renderer::ColorMode_Grayscale;
    view->update();
}


void RenderController::
        receiveToogleHeightlines(bool value)
{
    model->renderer->draw_height_lines = value;
    view->update();
}


void RenderController::
        receiveTogglePiano(bool value)
{
    model->renderer->draw_piano = value;
    view->update();
}


void RenderController::
        receiveToggleHz(bool value)
{
    model->renderer->draw_hz = value;
    view->update();
}


void RenderController::
        receiveSetYScale( int value )
{
    float f = value / 50.f - 1.f;
    model->renderer->y_scale = exp( 4.f*f*f * (f>0?1:-1));
    view->update();
}


void RenderController::
        receiveSetTimeFrequencyResolution( int value )
{
    unsigned FS = model->project->worker.source()->sample_rate();

    Tfr::Cwt& c = Tfr::Cwt::Singleton();
    c.tf_resolution( exp( 4*(value / 50.f - 1.f)) );

    float std_t = c.morlet_std_t(0, FS);

    // One standard deviation is not enough, but heavy. Two standard deviations are even more heavy.
    c.wavelet_std_t( 1.5f * std_t );

    Tfr::Stft& s = Tfr::Stft::Singleton();
    s.set_approximate_chunk_size( c.wavelet_std_t() * FS );

    model->collection->invalidate_samples( Signal::Intervals::Intervals_ALL );
    view->update();
}


void RenderController::
        receiveSetTransform_Cwt()
{
    Signal::pOperation s = model->collection->postsink();
    Signal::PostSink* ps = dynamic_cast<Signal::PostSink*>(s.get());

    if (!ps)
        return;

    std::vector<Signal::pOperation> v;
    Heightmap::CwtToBlock* cwtblock = new Heightmap::CwtToBlock(model->collection.get());
    v.push_back( Signal::pOperation( cwtblock ) );
    ps->sinks(v);
    cwtblock->complex_info = Heightmap::ComplexInfo_Amplitude_Weighted;

    view->update();
}


void RenderController::
        receiveSetTransform_Stft()
{
    Signal::pOperation s = model->collection->postsink();
    Signal::PostSink* ps = dynamic_cast<Signal::PostSink*>(s.get());

    if (!ps)
        return;

    std::vector<Signal::pOperation> v;
    Heightmap::StftToBlock* cwtblock = new Heightmap::StftToBlock(model->collection.get());
    v.push_back( Signal::pOperation( cwtblock ) );
    ps->sinks(v);

    view->update();
}


void RenderController::
        receiveSetTransform_Cwt_phase()
{
    Signal::pOperation s = model->collection->postsink();
    Signal::PostSink* ps = dynamic_cast<Signal::PostSink*>(s.get());

    if (!ps)
        return;

    std::vector<Signal::pOperation> v;
    Heightmap::CwtToBlock* cwtblock = new Heightmap::CwtToBlock(model->collection.get());
    v.push_back( Signal::pOperation( cwtblock ) );
    ps->sinks(v);
    cwtblock->complex_info = Heightmap::ComplexInfo_Phase;

    view->update();
}


void RenderController::
        receiveSetTransform_Cwt_reassign()
{
    Signal::pOperation s = model->collection->postsink();
    Signal::PostSink* ps = dynamic_cast<Signal::PostSink*>(s.get());

    if (!ps)
        return;

    std::vector<Signal::pOperation> v;
    Heightmap::CwtToBlock* cwtblock = new Heightmap::CwtToBlock(model->collection.get());
    v.push_back( Signal::pOperation( cwtblock ) );
    ps->sinks(v);
    cwtblock->complex_info = Heightmap::ComplexInfo_Amplitude_Non_Weighted;

    ps->filter( Signal::pOperation(new Filters::Reassign()));

    view->update();
}


void RenderController::
        receiveSetTransform_Cwt_ridge()
{
    Signal::pOperation s = model->collection->postsink();
    Signal::PostSink* ps = dynamic_cast<Signal::PostSink*>(s.get());

    if (!ps)
        return;

    std::vector<Signal::pOperation> v;
    Heightmap::CwtToBlock* cwtblock = new Heightmap::CwtToBlock(model->collection.get());
    v.push_back( Signal::pOperation( cwtblock ) );
    ps->sinks(v);
    cwtblock->complex_info = Heightmap::ComplexInfo_Amplitude_Weighted;

    ps->filter( Signal::pOperation(new Filters::Ridge()));

    view->update();
}


void RenderController::
        setupGui()
{
    Ui::SaweMainWindow* main = dynamic_cast<Ui::SaweMainWindow*>(model->project->mainWindow());
    toolbar_render = new QToolBar(main);
    toolbar_render->setObjectName(QString::fromUtf8("toolBarTool"));
    toolbar_render->setEnabled(true);
    toolbar_render->setContextMenuPolicy(Qt::NoContextMenu);
    toolbar_render->setToolButtonStyle(Qt::ToolButtonIconOnly);
    //main->addToolBar(Qt::BottomToolBarArea, toolbar_render);

    // Find Qt Creator managed actions
    Ui::MainWindow* ui = main->ui;


    // ComboBoxAction* hzmarker
    {   hzmarker = new ComboBoxAction();
        hzmarker->addActionItem( ui->actionToggle_hz_grid );
        hzmarker->addActionItem( ui->actionToggle_piano_grid );
        toolbar_render->addWidget( hzmarker );

        connect(ui->actionToggle_piano_grid, SIGNAL(toggled(bool)), SLOT(receiveTogglePiano(bool)));
        connect(ui->actionToggle_hz_grid, SIGNAL(toggled(bool)), SLOT(receiveToggleHz(bool)));
    }


    // ComboBoxAction* color
    {   color = new ComboBoxAction();
        color->decheckable( false );
        color->addActionItem( ui->actionSet_rainbow_colors );
        color->addActionItem( ui->actionSet_grayscale );
        toolbar_render->addWidget( color );

        connect(ui->actionSet_rainbow_colors, SIGNAL(triggered()), SLOT(receiveSetRainbowColors()));
        connect(ui->actionSet_grayscale, SIGNAL(triggered()), SLOT(receiveSetGrayscaleColors()));
    }


    // ComboBoxAction* transform
    {   transform = new ComboBoxAction();
        transform->addActionItem( ui->actionTransform_Cwt );
        transform->addActionItem( ui->actionTransform_Stft );
        transform->addActionItem( ui->actionTransform_Cwt_phase );
        transform->addActionItem( ui->actionTransform_Cwt_reassign );
        transform->addActionItem( ui->actionTransform_Cwt_ridge );
        transform->decheckable( false );
        toolbar_render->addWidget( transform );

        connect(ui->actionTransform_Cwt, SIGNAL(triggered()), SLOT(receiveSetTransform_Cwt()));
        connect(ui->actionTransform_Stft, SIGNAL(triggered()), SLOT(receiveSetTransform_Stft()));
        connect(ui->actionTransform_Cwt_phase, SIGNAL(triggered()), SLOT(receiveSetTransform_Cwt_phase()));
        connect(ui->actionTransform_Cwt_reassign, SIGNAL(triggered()), SLOT(receiveSetTransform_Cwt_reassign()));
        connect(ui->actionTransform_Cwt_ridge, SIGNAL(triggered()), SLOT(receiveSetTransform_Cwt_ridge()));
    }


    // QSlider * yscale
    {   yscale = new QSlider();
        yscale->setOrientation( Qt::Horizontal );
        yscale->setValue( 50 );
        yscale->setToolTip( "Intensity level" );
        toolbar_render->addWidget( yscale );

        connect(yscale, SIGNAL(valueChanged(int)), SLOT(receiveSetYScale(int)));
    }


    // QSlider * tf_resolution
    {   tf_resolution = new QSlider();
        tf_resolution->setOrientation( Qt::Horizontal );
        tf_resolution->setValue( 50 );
        tf_resolution->setToolTip( "Time/frequency resolution. If set higher than the middle, the audio reconstruction will be incorrect." );
        toolbar_render->addWidget( tf_resolution );

        connect(tf_resolution, SIGNAL(valueChanged(int)), SLOT(receiveSetTimeFrequencyResolution(int)));
    }


    // QAction *actionSet_heightlines
    toolbar_render->addAction(ui->actionSet_heightlines);
    connect(ui->actionSet_heightlines, SIGNAL(toggled(bool)), SLOT(receiveToogleHeightlines(bool)));


    // Release cuda buffers and disconnect them from OpenGL before destroying
    // OpenGL rendering context. Just good housekeeping.
    connect(view, SIGNAL(destroyingRenderView()), SLOT(clearCachedHeightmap()));
    connect(view, SIGNAL(paintedView()), SLOT(frameTick()));


    // Embed the god object: DisplayWidget
    Ui::DisplayWidget* d = new Ui::DisplayWidget( model->project, view, model );
    view->makeCurrent();
    view->displayWidget = d;

    d->setLayout(new QHBoxLayout());
    d->layout()->setMargin(0);
    d->layout()->addWidget(view);

    main->centralWidget()->layout()->setMargin(0);
    main->centralWidget()->layout()->addWidget(d);


    // TODO remove
    // connect(d, SIGNAL(operationsUpdated(Signal::pOperation)), this, SLOT(updateLayerList(Signal::pOperation)));
    // connect(d, SIGNAL(operationsUpdated(Signal::pOperation)), this, SLOT(updateOperationsTree(Signal::pOperation)));
    //connect(this, SIGNAL(sendCurrentSelection(int, bool)), d, SLOT(receiveCurrentSelection(int, bool)));
    //connect(this, SIGNAL(sendRemoveItem(int)), d, SLOT(receiveFilterRemoval(int)));

    connect(ui->actionActivateSelection, SIGNAL(toggled(bool)), d, SLOT(receiveToggleSelection(bool)));
    connect(ui->actionActivateNavigation, SIGNAL(toggled(bool)), d, SLOT(receiveToggleNavigation(bool)));
    connect(ui->actionActivateInfoTool, SIGNAL(toggled(bool)), d, SLOT(receiveToggleInfoTool(bool)));
    connect(ui->actionPlaySelection, SIGNAL(triggered()), d, SLOT(receivePlaySound()));
    connect(ui->actionFollowPlayMarker, SIGNAL(triggered(bool)), d, SLOT(receiveFollowPlayMarker(bool)));
    connect(ui->actionActionAdd_selection, SIGNAL(triggered(bool)), d, SLOT(receiveAddSelection(bool)));
    connect(ui->actionActionRemove_selection, SIGNAL(triggered(bool)), d, SLOT(receiveAddClearSelection(bool)));
    connect(ui->actionCropSelection, SIGNAL(triggered()), d, SLOT(receiveCropSelection()));
    connect(ui->actionMoveSelection, SIGNAL(triggered(bool)), d, SLOT(receiveMoveSelection(bool)));
    connect(ui->actionMoveSelectionTime, SIGNAL(triggered(bool)), d, SLOT(receiveMoveSelectionInTime(bool)));
    connect(ui->actionMatlabOperation, SIGNAL(triggered(bool)), d, SLOT(receiveMatlabOperation(bool)));
    connect(ui->actionMatlabFilter, SIGNAL(triggered(bool)), d, SLOT(receiveMatlabFilter(bool)));
    connect(ui->actionTonalizeFilter, SIGNAL(triggered(bool)), d, SLOT(receiveTonalizeFilter(bool)));
    connect(ui->actionReassignFilter, SIGNAL(triggered(bool)), d, SLOT(receiveReassignFilter(bool)));
    connect(ui->actionRecord, SIGNAL(triggered(bool)), d, SLOT(receiveRecord(bool)));
    connect(d, SIGNAL(setSelectionActive(bool)), ui->actionActivateSelection, SLOT(setChecked(bool)));
    connect(d, SIGNAL(setNavigationActive(bool)), ui->actionActivateNavigation, SLOT(setChecked(bool)));
    connect(d, SIGNAL(setInfoToolActive(bool)), ui->actionActivateInfoTool, SLOT(setChecked(bool)));


    ui->actionActivateNavigation->setChecked(true);

    // updateOperationsTree( project->worker.source() );
    //d->getCwtFilterHead();

    if (d->isRecordSource()) {
        ui->actionRecord->setEnabled(true);
    } else {
        ui->actionRecord->setEnabled(false);
    }
}


void RenderController::
        clearCachedHeightmap()
{
    if (model->collection->empty())
        return;

    // Assuming calling thread is the GUI thread.

    // Clear all cached blocks and release cuda memory befure destroying cuda
    // context
    model->collection->reset();

    // Because the cuda context was created with cudaGLSetGLDevice it is bound
    // to OpenGL. If we don't have an OpenGL context anymore the Cuda context
    // is corrupt and can't be destroyed nor used properly.
    BOOST_ASSERT( QGLContext::currentContext() );

    // Destroy the cuda context for this thread
    CudaException_SAFE_CALL( cudaThreadExit() );
}


void RenderController::
        frameTick()
{
    QMutexLocker l(&_invalidRangeMutex); // 0.00 ms
    if (!_invalidRange.isEmpty()) {
        Signal::Intervals blur = _invalidRange;
        unsigned fuzzy = Tfr::Cwt::Singleton().wavelet_std_samples(model->project->worker.source()->sample_rate());
        blur += fuzzy;
        _invalidRange |= blur;

        blur = _invalidRange;
        blur -= fuzzy;
        _invalidRange |= blur;

        model->collection->invalidate_samples( _invalidRange );
        _invalidRange = Signal::Intervals();
    }
}

} // namespace Tools

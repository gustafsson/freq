#include "rendercontroller.h"

// tools
#include "support/sinksignalproxy.h"
#include "signal/worker.h"

// ui
#include "ui/comboboxaction.h"
#include "ui_mainwindow.h" // Locate actions for binding
#include "ui/mainwindow.h"

// Sonic AWE, Setting different transforms for rendering
#include "filters/reassign.h"
#include "filters/ridge.h"
#include "heightmap/blockfilter.h"
#include "signal/postsink.h"
#include "tfr/cwt.h"
#include "graphicsview.h"

#include "signal/buffersource.h"

// gpumisc
#include <CudaException.h>
#include <cuda.h>
#include <demangle.h>

// Qt
#include <QToolBar>
#include <QSlider>
#include <QGraphicsView>
#include <QResizeEvent>
#include <QMetaClassInfo>
// todo remove
#include "navigationcontroller.h"
#include <QTimer>

using namespace Ui;

namespace Tools
{

class ForAllChannelsOperation: public Signal::Operation
{
public:
    ForAllChannelsOperation( Signal::pOperation o )
    :   Signal::Operation(o)
    {}


    virtual Signal::pBuffer read( const Signal::Interval& I )
    {
		Signal::FinalSource* fs = dynamic_cast<Signal::FinalSource*>(root());
        if (0==fs)
            return Signal::Operation::read( I );

        unsigned N = fs->num_channels();
        Signal::pBuffer r;
        for (unsigned i=0; i<N; ++i)
        {
			fs->set_channel( i );
            r = Signal::Operation::read( I );
        }
        return r;
    }

    virtual void source(Signal::pOperation v) { _source->source(v); }
};


RenderController::
        RenderController( RenderView *view )
            :
            view(view),
            toolbar_render(0),
            hzmarker(0),
            color(0),
            transform(0)
{
    setupGui();

    // Default values
    float l = model()->project()->worker.source()->length();
    view->setPosition( std::min(l, 10.f)*0.5f, 0.5f );

    receiveSetTimeFrequencyResolution( 50 );
    receiveSetTransform_Cwt();
    receiveSetColorscaleColors();
}


RenderController::
        ~RenderController()
{
    if (QGLContext::currentContext())
        clearCachedHeightmap();
}


void RenderController::
        receiveSetRainbowColors()
{
    model()->renderer->color_mode = Heightmap::Renderer::ColorMode_Rainbow;
    view->userinput_update();
}


void RenderController::
        receiveSetGrayscaleColors()
{
    model()->renderer->color_mode = Heightmap::Renderer::ColorMode_Grayscale;
    view->userinput_update();
}


void RenderController::
        receiveSetColorscaleColors()
{
    model()->renderer->color_mode = Heightmap::Renderer::ColorMode_FixedColor;
    view->userinput_update();
}


void RenderController::
        receiveToogleHeightlines(bool value)
{
    model()->renderer->draw_height_lines = value;
    view->userinput_update();
}


void RenderController::
        receiveTogglePiano(bool value)
{
    model()->renderer->draw_piano = value;
    view->userinput_update();
}


void RenderController::
        receiveToggleHz(bool value)
{
    model()->renderer->draw_hz = value;
    view->userinput_update();
}


void RenderController::
        receiveSetYScale( int value )
{
    float f = value / 50.f - 1.f;
    model()->renderer->y_scale = exp( 4.f*f*f * (f>0?1:-1));
    view->userinput_update();
}


void RenderController::
        receiveSetTimeFrequencyResolution( int value )
{
    float FS = model()->project()->worker.source()->sample_rate();

    Tfr::Cwt& c = Tfr::Cwt::Singleton();
    c.tf_resolution( 2.5f * exp( 4*(value / 50.f - 1.f)) );

    Tfr::Stft& s = Tfr::Stft::Singleton();
    s.set_approximate_chunk_size( c.wavelet_time_support_samples(FS) );

    foreach( const boost::shared_ptr<Heightmap::Collection>& collection, model()->collections )
        collection->invalidate_samples( Signal::Intervals::Intervals_ALL );
    view->userinput_update();
}


Signal::PostSink* RenderController::
        setBlockFilter(Signal::Operation* blockfilter)
{
    Signal::pOperation s = model()->postsink();
    Signal::PostSink* ps = dynamic_cast<Signal::PostSink*>(s.get());

    BOOST_ASSERT( ps );

    std::vector<Signal::pOperation> v;
    Signal::pOperation blockop( blockfilter );
    Signal::pOperation channelop( new ForAllChannelsOperation(blockop));
    v.push_back( channelop );
    ps->sinks(v);

    foreach( const boost::shared_ptr<Heightmap::Collection>& collection, model()->collections )
    {
        collection->invalidate_samples(Signal::Intervals::Intervals_ALL);
    }

    view->userinput_update();

    return ps;
}


void RenderController::
        receiveSetTransform_Cwt()
{
    Heightmap::CwtToBlock* cwtblock = new Heightmap::CwtToBlock(model()->collections);
    cwtblock->complex_info = Heightmap::ComplexInfo_Amplitude_Non_Weighted;

    setBlockFilter( cwtblock );
}


void RenderController::
        receiveSetTransform_Stft()
{
    Heightmap::StftToBlock* stftblock = new Heightmap::StftToBlock(model()->collections);

    setBlockFilter( stftblock );
}


void RenderController::
        receiveSetTransform_Cwt_phase()
{
    Heightmap::CwtToBlock* cwtblock = new Heightmap::CwtToBlock(model()->collections);
    cwtblock->complex_info = Heightmap::ComplexInfo_Phase;

    setBlockFilter( cwtblock );
}


void RenderController::
        receiveSetTransform_Cwt_reassign()
{
    Heightmap::CwtToBlock* cwtblock = new Heightmap::CwtToBlock(model()->collections);
    cwtblock->complex_info = Heightmap::ComplexInfo_Amplitude_Weighted;

    Signal::PostSink* ps = setBlockFilter( cwtblock );

    ps->filter( Signal::pOperation(new Filters::Reassign()));
}


void RenderController::
        receiveSetTransform_Cwt_ridge()
{
    Heightmap::CwtToBlock* cwtblock = new Heightmap::CwtToBlock(model()->collections);
    cwtblock->complex_info = Heightmap::ComplexInfo_Amplitude_Non_Weighted;

    Signal::PostSink* ps = setBlockFilter( cwtblock );

    ps->filter( Signal::pOperation(new Filters::Ridge()));
}


void RenderController::
        receiveSetTransform_Cwt_weight()
{
    Heightmap::CwtToBlock* cwtblock = new Heightmap::CwtToBlock(model()->collections);
    cwtblock->complex_info = Heightmap::ComplexInfo_Amplitude_Weighted;

    setBlockFilter( cwtblock );
}


RenderModel *RenderController::
        model()
{
    return view->model;
}


void RenderController::
        setupGui()
{
    Ui::SaweMainWindow* main = dynamic_cast<Ui::SaweMainWindow*>(model()->project()->mainWindow());
    toolbar_render = new QToolBar(main);
    toolbar_render->setObjectName(QString::fromUtf8("toolBarTool"));
    toolbar_render->setEnabled(true);
    toolbar_render->setContextMenuPolicy(Qt::NoContextMenu);
    toolbar_render->setToolButtonStyle(Qt::ToolButtonIconOnly);
    main->addToolBar(Qt::BottomToolBarArea, toolbar_render);


    // Find Qt Creator managed actions
    Ui::MainWindow* ui = main->getItems();


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
        color->addActionItem( ui->actionSet_colorscale );
        toolbar_render->addWidget( color );

        connect(ui->actionSet_rainbow_colors, SIGNAL(triggered()), SLOT(receiveSetRainbowColors()));
        connect(ui->actionSet_grayscale, SIGNAL(triggered()), SLOT(receiveSetGrayscaleColors()));
        connect(ui->actionSet_colorscale, SIGNAL(triggered()), SLOT(receiveSetColorscaleColors()));
    }


    // ComboBoxAction* transform
    {   transform = new ComboBoxAction();
        transform->addActionItem( ui->actionTransform_Cwt );
        transform->addActionItem( ui->actionTransform_Stft );
        transform->addActionItem( ui->actionTransform_Cwt_phase );
        transform->addActionItem( ui->actionTransform_Cwt_reassign );
        transform->addActionItem( ui->actionTransform_Cwt_ridge );
        transform->addActionItem( ui->actionTransform_Cwt_weight );

        transform->decheckable( false );
        toolbar_render->addWidget( transform );

        connect(ui->actionTransform_Cwt, SIGNAL(triggered()), SLOT(receiveSetTransform_Cwt()));
        connect(ui->actionTransform_Stft, SIGNAL(triggered()), SLOT(receiveSetTransform_Stft()));
        connect(ui->actionTransform_Cwt_phase, SIGNAL(triggered()), SLOT(receiveSetTransform_Cwt_phase()));
        connect(ui->actionTransform_Cwt_reassign, SIGNAL(triggered()), SLOT(receiveSetTransform_Cwt_reassign()));
        connect(ui->actionTransform_Cwt_ridge, SIGNAL(triggered()), SLOT(receiveSetTransform_Cwt_ridge()));
        connect(ui->actionTransform_Cwt_weight, SIGNAL(triggered()), SLOT(receiveSetTransform_Cwt_weight()));

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


    // Update view whenever worker is invalidated
    Support::SinkSignalProxy* proxy;
    _updateViewSink.reset( proxy = new Support::SinkSignalProxy() );
    _updateViewCallback.reset( new Signal::WorkerCallback( &model()->project()->worker, _updateViewSink ));
    connect( proxy, SIGNAL(recievedInvalidSamples(const Signal::Intervals &)), view, SLOT(update()));


    // Release cuda buffers and disconnect them from OpenGL before destroying
    // OpenGL rendering context. Just good housekeeping.
    connect(view, SIGNAL(destroying()), SLOT(clearCachedHeightmap()));
    connect(view, SIGNAL(postPaint()), SLOT(frameTick()));

    // Create the OpenGL rendering context early. Because we want to create the
    // cuda context (in main.cpp) and bind it to an OpenGL context before the
    // context is required to be created by lazy initialization when painting
    // the widget
    //view->makeCurrent();

    view->glwidget = new QGLWidget(QGLFormat(QGL::SampleBuffers));
    view->glwidget->makeCurrent();
    //view->glwidget->setMouseTracking(true);

    GraphicsView* g = new GraphicsView(view);
    g->setViewport(view->glwidget);
    view->tool_selector.reset( new Support::ToolSelector(g->toolParent));

    main->centralWidget()->layout()->setMargin(0);
    main->centralWidget()->layout()->addWidget(g);
}


void RenderController::
        clearCachedHeightmap()
{
    // Stop worker from producing any more heightmaps by disconnecting
    // the collection callback from worker.
    model()->collectionCallback.reset();

    // Assuming calling thread is the GUI thread.

    // Clear all cached blocks and release cuda memory befure destroying cuda
    // context
    foreach( const boost::shared_ptr<Heightmap::Collection>& collection, model()->collections )
        collection->reset();
}

void RenderController::
        frameTick()
{
    QMutexLocker l(&_invalidRangeMutex); // 0.00 ms
    if (_invalidRange)
    {
        Signal::Intervals blur = _invalidRange;
        float fs = model()->project()->worker.source()->sample_rate();
        unsigned fuzzy = Tfr::Cwt::Singleton().wavelet_time_support_samples( fs );
        blur <<= fuzzy;
        _invalidRange |= blur;

        blur = _invalidRange;
        blur >>= fuzzy;
        _invalidRange |= blur;

        foreach( const boost::shared_ptr<Heightmap::Collection>& collection, model()->collections )
            collection->invalidate_samples( _invalidRange );
        _invalidRange = Signal::Intervals();
    }
}

} // namespace Tools

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

// gpumisc
#include <CudaException.h>
#include <cuda.h>

// Qt
#include <QToolBar>
#include <QSlider>
#include <QGraphicsView>
#include <QResizeEvent>

// todo remove
#include "navigationcontroller.h"

using namespace Ui;

namespace Tools
{

class GraphicsView : public QGraphicsView
{
public:
    GraphicsView()
    {
        setWindowTitle(tr("Boxes"));
        setRenderHints(QPainter::Antialiasing | QPainter::SmoothPixmapTransform);
        //setRenderHints(QPainter::SmoothPixmapTransform);
    }

    ~GraphicsView()
    {
        if (scene())
            delete scene();
    }

protected:
    void resizeEvent(QResizeEvent *event) {
        if (scene())
            scene()->setSceneRect(QRect(QPoint(0, 0), event->size()));
        QGraphicsView::resizeEvent(event);
    }
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

    model()->collection->invalidate_samples( Signal::Intervals::Intervals_ALL );
    view->userinput_update();
}


void RenderController::
        receiveSetTransform_Cwt()
{
    Signal::pOperation s = model()->collection->postsink();
    Signal::PostSink* ps = dynamic_cast<Signal::PostSink*>(s.get());

    if (!ps)
        return;

    std::vector<Signal::pOperation> v;
    Heightmap::CwtToBlock* cwtblock = new Heightmap::CwtToBlock(model()->collection.get());
    v.push_back( Signal::pOperation( cwtblock ) );
    ps->sinks(v);
    cwtblock->complex_info = Heightmap::ComplexInfo_Amplitude_Non_Weighted;

    model()->collection->invalidate_samples(Signal::Intervals::Intervals_ALL);
    view->userinput_update();
}


void RenderController::
        receiveSetTransform_Stft()
{
    Signal::pOperation s = model()->collection->postsink();
    Signal::PostSink* ps = dynamic_cast<Signal::PostSink*>(s.get());

    if (!ps)
        return;

    std::vector<Signal::pOperation> v;
    Heightmap::StftToBlock* cwtblock = new Heightmap::StftToBlock(model()->collection.get());
    v.push_back( Signal::pOperation( cwtblock ) );
    ps->sinks(v);

    model()->collection->invalidate_samples(Signal::Intervals::Intervals_ALL);
    view->userinput_update();
}


void RenderController::
        receiveSetTransform_Cwt_phase()
{
    Signal::pOperation s = model()->collection->postsink();
    Signal::PostSink* ps = dynamic_cast<Signal::PostSink*>(s.get());

    if (!ps)
        return;

    std::vector<Signal::pOperation> v;
    Heightmap::CwtToBlock* cwtblock = new Heightmap::CwtToBlock(model()->collection.get());
    v.push_back( Signal::pOperation( cwtblock ) );
    ps->sinks(v);
    cwtblock->complex_info = Heightmap::ComplexInfo_Phase;

    model()->collection->invalidate_samples(Signal::Intervals::Intervals_ALL);
    view->userinput_update();
}


void RenderController::
        receiveSetTransform_Cwt_reassign()
{
    Signal::pOperation s = model()->collection->postsink();
    Signal::PostSink* ps = dynamic_cast<Signal::PostSink*>(s.get());

    if (!ps)
        return;

    std::vector<Signal::pOperation> v;
    Heightmap::CwtToBlock* cwtblock = new Heightmap::CwtToBlock(model()->collection.get());
    v.push_back( Signal::pOperation( cwtblock ) );
    ps->sinks(v);
    cwtblock->complex_info = Heightmap::ComplexInfo_Amplitude_Weighted;

    ps->filter( Signal::pOperation(new Filters::Reassign()));

    model()->collection->invalidate_samples(Signal::Intervals::Intervals_ALL);
    view->userinput_update();
}


void RenderController::
        receiveSetTransform_Cwt_ridge()
{
    Signal::pOperation s = model()->collection->postsink();
    Signal::PostSink* ps = dynamic_cast<Signal::PostSink*>(s.get());

    if (!ps)
        return;

    std::vector<Signal::pOperation> v;
    Heightmap::CwtToBlock* cwtblock = new Heightmap::CwtToBlock(model()->collection.get());
    v.push_back( Signal::pOperation( cwtblock ) );
    ps->sinks(v);
    cwtblock->complex_info = Heightmap::ComplexInfo_Amplitude_Non_Weighted;

    ps->filter( Signal::pOperation(new Filters::Ridge()));

    model()->collection->invalidate_samples(Signal::Intervals::Intervals_ALL);
    view->userinput_update();
}


void RenderController::
        receiveSetTransform_Cwt_weight()
{
    Signal::pOperation s = model()->collection->postsink();
    Signal::PostSink* ps = dynamic_cast<Signal::PostSink*>(s.get());

    if (!ps)
        return;

    std::vector<Signal::pOperation> v;
    Heightmap::CwtToBlock* cwtblock = new Heightmap::CwtToBlock(model()->collection.get());
    v.push_back( Signal::pOperation( cwtblock ) );
    ps->sinks(v);
    cwtblock->complex_info = Heightmap::ComplexInfo_Amplitude_Weighted;

    model()->collection->invalidate_samples(Signal::Intervals::Intervals_ALL);
    view->userinput_update();
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

    // Make all child widgets occupy the entire area
    //view->setLayout(new QHBoxLayout());
    //view->layout()->setMargin(0);

    view->glwidget = new QGLWidget(QGLFormat(QGL::SampleBuffers));
    view->glwidget->makeCurrent();
    view->glwidget->setLayout(new QHBoxLayout());
    view->glwidget->layout()->setMargin(0);

    GraphicsView* g = new GraphicsView();
    g->setLayout(new QHBoxLayout());
    g->layout()->setMargin(0);
    g->setViewport(view->glwidget);
    g->setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
    g->setScene( view );

    view->tool_selector.reset( new Support::ToolSelector(view->glwidget));

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
    model()->collection->reset();
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

        model()->collection->invalidate_samples( _invalidRange );
        _invalidRange = Signal::Intervals();
    }
}

} // namespace Tools

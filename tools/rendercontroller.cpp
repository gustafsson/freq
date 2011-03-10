#include "rendercontroller.h"

// tools
#include "support/sinksignalproxy.h"
#include "support/toolbar.h"

// ui
#include "ui/comboboxaction.h"
#include "ui_mainwindow.h" // Locate actions for binding
#include "ui/mainwindow.h"

// Sonic AWE, Setting different transforms for rendering
#include "filters/reassign.h"
#include "filters/ridge.h"
#include "heightmap/blockfilter.h"
#include "heightmap/renderer.h"
#include "signal/postsink.h"
#include "tfr/cwt.h"
#include "tfr/cepstrum.h"
#include "graphicsview.h"
#include "sawe/application.h"
#include "signal/buffersource.h"
#include "signal/worker.h"

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
#include <QGLContext>

// todo remove
#include "navigationcontroller.h"
#include <QTimer>

using namespace Ui;

#ifdef min
#undef min
#endif

namespace Tools
{

class BlockFilterSink: public Signal::Sink
{
public:
    BlockFilterSink
        (
            Signal::pOperation o,
            RenderModel* model,
            RenderController* controller
        )
        :
            model_(model),
            controller_(controller)
    {
        BOOST_ASSERT( o );
        Operation::source(o);
    }


    virtual void source(Signal::pOperation v) { Operation::source()->source(v); }
    virtual bool deleteMe() { return false; } // Never delete this sink

    virtual Signal::pBuffer read(const Signal::Interval& I) {
        Signal::pBuffer r = Signal::Operation::read( I );
        return r;
    }

    virtual void invalidate_samples(const Signal::Intervals& I)
    {
        validateSize();

        // If BlockFilter is a CwtFilter wavelet time support has already been included in I

        foreach(boost::shared_ptr<Heightmap::Collection> c, model_->collections)
        {
            c->invalidate_samples( I );
        }

        Operation::invalidate_samples( I );
    }


    virtual Signal::Intervals invalid_samples()
    {
        Signal::Intervals I;
        foreach ( boost::shared_ptr<Heightmap::Collection> c, model_->collections)
        {
            Signal::Intervals inv_coll = c->invalid_samples();
            I |= inv_coll;
        }

        return I;
    }


    void validateSize()
    {
        unsigned N = num_channels();
        if ( N != model_->collections.size())
        {
            model_->collections.resize(N);
            for (unsigned c=0; c<N; ++c)
            {
                if (!model_->collections[c])
                    model_->collections[c].reset( new Heightmap::Collection(model_->renderSignalTarget->source()));
            }
        }

        foreach(boost::shared_ptr<Heightmap::Collection> c, model_->collections)
        {
            c->block_filter( Operation::source() );
        }
    }


private:
    RenderModel* model_;
    RenderController* controller_;
};


RenderController::
        RenderController( QPointer<RenderView> view )
            :
            view(view),
            toolbar_render(0),
            hzmarker(0),
            color(0),
            transform(0)
{
    setupGui();

    // Default values for rendermodel are set in rendermodel constructor

    {
        // Default values for rendercontroller
        Ui::SaweMainWindow* main = dynamic_cast<Ui::SaweMainWindow*>(model()->project()->mainWindow());
        Ui::MainWindow* ui = main->getItems();
#ifdef TARGET_sss
        tf_resolution->setValue( 10 );
        ui->actionToggleOrientation->setChecked(true);
        transform->actions().at(0)->trigger();
#else
        transform->actions().at(1)->trigger();
#endif
        ui->actionSet_colorscale->trigger();
    }
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
        receiveToggleOrientation(bool value)
{
    model()->renderer->left_handed_axes = !value;

    view->graphicsview->setLayoutDirection( value
                                            ? QBoxLayout::RightToLeft
                                            : QBoxLayout::TopToBottom );

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
    emit transformChanged();
}


void RenderController::
        receiveSetTimeFrequencyResolution( int value )
{
    float FS = model()->project()->worker.source()->sample_rate();

    Tfr::Cwt& c = Tfr::Cwt::Singleton();

    // Keep in sync with emitTransformChanged()
    //float f = value / 50.f - 1.f;
    //c.scales_per_octave( 20.f * exp( 4*f ) );
    float f = value / (float)tf_resolution->maximum();
    c.scales_per_octave( exp( 7*f ) ); // scales_per_octave >= 1

    Tfr::Stft& s = Tfr::Stft::Singleton();
    s.set_approximate_chunk_size( c.wavelet_time_support_samples(FS)/c.wavelet_time_support()/c.wavelet_time_support() );

    zscale->defaultAction()->trigger();

    model()->renderSignalTarget->post_sink()->invalidate_samples( Signal::Intervals::Intervals_ALL );

    // Don't lock the UI, instead wait a moment before any change is made
    view->userinput_update();

    emit transformChanged();
}


Signal::PostSink* RenderController::
        setBlockFilter(Signal::Operation* blockfilter)
{
    BlockFilterSink* bfs;
    Signal::pOperation blockop( blockfilter );
    Signal::pOperation channelop( bfs = new BlockFilterSink(blockop, model(), this));

    std::vector<Signal::pOperation> v;
    v.push_back( channelop );
    Signal::PostSink* ps = model()->renderSignalTarget->post_sink();
    ps->sinks(v);
    bfs->validateSize();
    bfs->invalidate_samples( Signal::Intervals::Intervals_ALL );

    // Don't lock the UI, instead wait a moment before any change is made
    view->userinput_update();

    emit transformChanged();

    return ps;
}


void RenderController::
        receiveSetTransform_Cwt()
{
    Heightmap::CwtToBlock* cwtblock = new Heightmap::CwtToBlock(&model()->collections);
    cwtblock->complex_info = Heightmap::ComplexInfo_Amplitude_Non_Weighted;

    setBlockFilter( cwtblock );
}


void RenderController::
        receiveSetTransform_Stft()
{
    Heightmap::StftToBlock* stftblock = new Heightmap::StftToBlock(&model()->collections);

    setBlockFilter( stftblock );
}


void RenderController::
        receiveSetTransform_Cwt_phase()
{
    Heightmap::CwtToBlock* cwtblock = new Heightmap::CwtToBlock(&model()->collections);
    cwtblock->complex_info = Heightmap::ComplexInfo_Phase;

    setBlockFilter( cwtblock );
}


void RenderController::
        receiveSetTransform_Cwt_reassign()
{
    Heightmap::CwtToBlock* cwtblock = new Heightmap::CwtToBlock(&model()->collections);
    cwtblock->complex_info = Heightmap::ComplexInfo_Amplitude_Weighted;

    Signal::PostSink* ps = setBlockFilter( cwtblock );

    ps->filter( Signal::pOperation(new Filters::Reassign()));
}


void RenderController::
        receiveSetTransform_Cwt_ridge()
{
    Heightmap::CwtToBlock* cwtblock = new Heightmap::CwtToBlock(&model()->collections);
    cwtblock->complex_info = Heightmap::ComplexInfo_Amplitude_Non_Weighted;

    Signal::PostSink* ps = setBlockFilter( cwtblock );

    ps->filter( Signal::pOperation(new Filters::Ridge()));
}


void RenderController::
        receiveSetTransform_Cwt_weight()
{
    Heightmap::CwtToBlock* cwtblock = new Heightmap::CwtToBlock(&model()->collections);
    cwtblock->complex_info = Heightmap::ComplexInfo_Amplitude_Weighted;

    setBlockFilter( cwtblock );
}


void RenderController::
        receiveSetTransform_Cepstrum()
{
    Heightmap::CepstrumToBlock* cepstrumblock = new Heightmap::CepstrumToBlock(&model()->collections);

    setBlockFilter( cepstrumblock );
}


void RenderController::
        receiveLinearScale()
{
    float fs = model()->project()->head->head_source()->sample_rate();

    Tfr::FreqAxis fa;
    fa.setLinear( fs );

    model()->display_scale( fa );
    view->userinput_update();
}


void RenderController::
        receiveLogScale()
{
    float fs = model()->project()->head->head_source()->sample_rate();

    Tfr::FreqAxis fa;
    fa.setLogarithmic(
            Tfr::Cwt::Singleton().wanted_min_hz(),
            Tfr::Cwt::Singleton().get_max_hz(fs) );

    model()->display_scale( fa );
    view->userinput_update();
}


void RenderController::
        receiveCepstraScale()
{
    float fs = model()->project()->head->head_source()->sample_rate();

    Tfr::FreqAxis fa;
    fa.setQuefrencyNormalized( fs, Tfr::Cepstrum::Singleton().chunk_size() );

    model()->display_scale( fa );
    view->userinput_update();
}


RenderModel *RenderController::
        model()
{
    BOOST_ASSERT( view );
    return view->model;
}


void RenderController::
        emitTransformChanged()
{
    Tfr::Cwt& c = Tfr::Cwt::Singleton();

    // keep in sync with receiveSetTimeFrequencyResolution
    float f = log(c.scales_per_octave())/7;
    int value = f * tf_resolution->maximum() + .5;

    this->tf_resolution->setValue( value );
    emit transformChanged();
}


void RenderController::
        setupGui()
{
    Ui::SaweMainWindow* main = dynamic_cast<Ui::SaweMainWindow*>(model()->project()->mainWindow());
    toolbar_render = new Support::ToolBar(main);
    toolbar_render->setObjectName(QString::fromUtf8("toolBarRenderController"));
    toolbar_render->setEnabled(true);
    toolbar_render->setContextMenuPolicy(Qt::NoContextMenu);
    toolbar_render->setToolButtonStyle(Qt::ToolButtonIconOnly);
    main->addToolBar(Qt::BottomToolBarArea, toolbar_render);

    connect(main->getItems()->actionToggleTransformToolBox, SIGNAL(toggled(bool)), toolbar_render, SLOT(setVisible(bool)));
    connect((Support::ToolBar*)toolbar_render, SIGNAL(visibleChanged(bool)), main->getItems()->actionToggleTransformToolBox, SLOT(setChecked(bool)));


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

    connect(ui->actionResetGraphics, SIGNAL(triggered()), view, SLOT(clearCaches()));


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
        color->setCheckedAction(ui->actionSet_colorscale);
    }

    // QAction *actionSet_heightlines
    toolbar_render->addAction(ui->actionSet_heightlines);
    connect(ui->actionSet_heightlines, SIGNAL(toggled(bool)), SLOT(receiveToogleHeightlines(bool)));

    toolbar_render->addAction(ui->actionToggleOrientation);
    connect(ui->actionToggleOrientation, SIGNAL(toggled(bool)), SLOT(receiveToggleOrientation(bool)));

    // ComboBoxAction* transform
    {   connect(ui->actionTransform_Cwt, SIGNAL(triggered()), SLOT(receiveSetTransform_Cwt()));
        connect(ui->actionTransform_Stft, SIGNAL(triggered()), SLOT(receiveSetTransform_Stft()));
        connect(ui->actionTransform_Cwt_phase, SIGNAL(triggered()), SLOT(receiveSetTransform_Cwt_phase()));
        connect(ui->actionTransform_Cwt_reassign, SIGNAL(triggered()), SLOT(receiveSetTransform_Cwt_reassign()));
        connect(ui->actionTransform_Cwt_ridge, SIGNAL(triggered()), SLOT(receiveSetTransform_Cwt_ridge()));
        connect(ui->actionTransform_Cwt_weight, SIGNAL(triggered()), SLOT(receiveSetTransform_Cwt_weight()));
        connect(ui->actionTransform_Cepstrum, SIGNAL(triggered()), SLOT(receiveSetTransform_Cepstrum()));

        transform = new ComboBoxAction();
        transform->addActionItem( ui->actionTransform_Stft );
        transform->addActionItem( ui->actionTransform_Cwt );
        transform->addActionItem( ui->actionTransform_Cepstrum );
//        transform->addActionItem( ui->actionTransform_Cwt_phase );
//        transform->addActionItem( ui->actionTransform_Cwt_reassign );
//        transform->addActionItem( ui->actionTransform_Cwt_ridge );
//        transform->addActionItem( ui->actionTransform_Cwt_weight );
        transform->decheckable( false );
        toolbar_render->addWidget( transform );

        unsigned k=0;
        foreach( QAction* a, transform->actions())
        {
            a->setShortcut('1' + k++);
        }
    }


    // ComboBoxAction* zscale
    {   QAction* linearScale = new QAction( main );
        QAction* logScale = new QAction( main );
        QAction* cepstraScale = new QAction( main );

        linearScale->setText("Linear scale");
        logScale->setText("Logarithmic scale");
        cepstraScale->setText("Cepstra scale");

        linearScale->setCheckable( true );
        logScale->setCheckable( true );
        cepstraScale->setCheckable( true );

        connect(linearScale, SIGNAL(triggered()), SLOT(receiveLinearScale()));
        connect(logScale, SIGNAL(triggered()), SLOT(receiveLogScale()));
        connect(cepstraScale, SIGNAL(triggered()), SLOT(receiveCepstraScale()));

        zscale = new ComboBoxAction();
        zscale->addActionItem( linearScale );
        zscale->addActionItem( logScale );
        zscale->addActionItem( cepstraScale );
        zscale->decheckable( false );
        toolbar_render->addWidget( zscale );

        unsigned k=0;
        foreach( QAction* a, zscale->actions())
        {
            a->setShortcut(QString("Ctrl+") + ('1' + k++));
        }
        logScale->trigger();
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
        tf_resolution->setValue( 5000 );
        tf_resolution->setMaximum( 10000 );
        tf_resolution->setToolTip( "Time/frequency resolution." );
        toolbar_render->addWidget( tf_resolution );

        connect(tf_resolution, SIGNAL(valueChanged(int)), SLOT(receiveSetTimeFrequencyResolution(int)));
    }


    // Release cuda buffers and disconnect them from OpenGL before destroying
    // OpenGL rendering context. Just good housekeeping.
    connect(view, SIGNAL(destroying()), SLOT(clearCachedHeightmap()));
    connect(view, SIGNAL(transformChanged()), SLOT(emitTransformChanged()));

    // Create the OpenGL rendering context early. Because we want to create the
    // cuda context (in main.cpp) and bind it to an OpenGL context before the
    // context is required to be created by lazy initialization when painting
    // the widget
    view->glwidget = new QGLWidget( 0, Sawe::Application::shared_glwidget(), Qt::WindowFlags(0) );
    view->glwidget->makeCurrent();
    //view->glwidget->setMouseTracking(true);

    view->graphicsview = new GraphicsView(view);
    view->graphicsview->setViewport(view->glwidget);
    view->glwidget->makeCurrent(); // setViewport makes the glwidget loose context, take it back
    view->tool_selector = view->graphicsview->toolSelector(0);

    main->centralWidget()->layout()->setMargin(0);
    main->centralWidget()->layout()->addWidget(view->graphicsview);

    emitTransformChanged();
}


void RenderController::
        clearCachedHeightmap()
{
    // Stop worker from producing any more heightmaps by disconnecting
    // the collection callback from worker.
    if (model()->renderSignalTarget == model()->project()->worker.target())
        model()->project()->worker.target(Signal::pTarget());
    model()->renderSignalTarget.reset();


    // Assuming calling thread is the GUI thread.

    // Clear all cached blocks and release cuda memory befure destroying cuda
    // context
    foreach( const boost::shared_ptr<Heightmap::Collection>& collection, model()->collections )
        collection->reset();
}

} // namespace Tools

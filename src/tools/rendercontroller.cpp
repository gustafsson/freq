#include "rendercontroller.h"

// tools
#include "support/sinksignalproxy.h"
#include "support/toolbar.h"
#include "widgets/valueslider.h"

// ui
#include "ui/comboboxaction.h"
#include "ui_mainwindow.h" // Locate actions for binding
#include "ui/mainwindow.h"

// Sonic AWE, Setting different transforms for rendering
#include "filters/reassign.h"
#include "filters/ridge.h"
#include "heightmap/render/renderer.h"
#include "heightmap/update/updateproducer.h"
#include "heightmap/update/updateconsumer.h"
#include "heightmap/tfrmappings/stftblockfilter.h"
#include "heightmap/tfrmappings/cwtblockfilter.h"
#include "heightmap/tfrmappings/cepstrumblockfilter.h"
#include "tfr/cwt.h"
#include "tfr/stft.h"
#include "tfr/cepstrum.h"
#include "tfr/transformoperation.h"
#include "graphicsview.h"
#include "graphicsscene.h"
#include "sawe/application.h"
#include "signal/buffersource.h"
#include "signal/reroutechannels.h"
#include "tools/support/operation-composite.h"
#include "tools/support/renderoperation.h"
#include "tools/support/renderviewupdateadapter.h"
#include "tools/support/heightmapprocessingpublisher.h"
#include "sawe/configuration.h"

// gpumisc
#include "demangle.h"
#include "computationkernel.h"
#include "log.h"

// Qt
#include <QSlider>
#include <QGraphicsView>
#include <QResizeEvent>
#include <QGLContext>
#include <QSettings>

// boost
#include <boost/format.hpp>

// todo remove
#include "navigationcontroller.h"
#include <QTimer>

using namespace Ui;
using namespace boost;

#ifdef min
#undef min
#endif

namespace Tools
{

RenderController::
        RenderController( QPointer<RenderView> view, Sawe::Project* project )
            :
            tool_selector(0),
            graphicsview(0),
            transform(0),
            hz_scale(0),
            amplitude_scale(0),
            hzmarker(0),
            hz_scale_action(0),
            amplitude_scale_action(0),
            tf_resolution_action(0),
            linearScale(0),
            view(view),
            project(project),
            toolbar_render(0),
            logScale(0),
            cepstraScale(0),
            color(0)
{
    Support::RenderViewUpdateAdapter* rvup;
    Support::RenderOperationDesc::RenderTarget::ptr rvu(
                rvup = new Support::RenderViewUpdateAdapter);

    connect(rvup, SIGNAL(redraw()), view, SLOT(redraw()));

    model()->init(project->processing_chain (), rvu);

    // 'this' is parent
    auto hpp = new Support::HeightmapProcessingPublisher(
                view->model->target_marker ()->target_needs (),
                view->model->tfr_mapping (),
                &view->model->camera.q[0],
                this);
    connect(rvup, SIGNAL(setLastUpdatedInterval(Signal::Interval)), hpp, SLOT(setLastUpdatedInterval(Signal::Interval)));
    connect(view, SIGNAL(painting()), hpp, SLOT(update()));
    setupGui();

    {
        // Default values for rendercontroller
        ::Ui::MainWindow* ui = getItems();
#ifdef TARGET_hast
        tf_resolution->setValue( 1<<14 );
//        transform->actions().at(0)->trigger();
#else
//        transform->actions().at(1)->trigger();
#endif
        transform->actions().at(0)->trigger();

        ui->actionTransform_Stft->trigger();
        logScale->trigger();

//#ifndef USE_CUDA
//        ui->actionTransform_Stft->trigger();
//        linearScale->trigger();
//#else
//        ui->actionTransform_Cwt->trigger();
//        logScale->trigger();
//#endif

        model()->transform_descs ()->getParam<Tfr::StftDesc>().setWindow(Tfr::StftDesc::WindowType_Hann, 0.75f);
        model()->transform_descs ()->getParam<Tfr::StftDesc>().enable_inverse(false);

        ui->actionToggleTransformToolBox->setChecked( true );
    }
}


RenderController::
        ~RenderController()
{
    if (QGLContext::currentContext())
        deleteTarget();
}


void RenderController::
        stateChanged()
{
    view->redraw();

    project->setModified();
}


void RenderController::
        emitAxisChanged()
{
    view->emitAxisChanged();
}


void RenderController::
        receiveSetRainbowColors()
{
    model()->render_settings.color_mode = Heightmap::Render::RenderSettings::ColorMode_Rainbow;
    stateChanged();
}


void RenderController::
        receiveSetGrayscaleColors()
{
    model()->render_settings.color_mode = Heightmap::Render::RenderSettings::ColorMode_Grayscale;
    stateChanged();
}


void RenderController::
        receiveSetBlackGrayscaleColors()
{
    model()->render_settings.color_mode = Heightmap::Render::RenderSettings::ColorMode_BlackGrayscale;
    stateChanged();
}


void RenderController::
        receiveSetColorscaleColors()
{
    model()->render_settings.color_mode = Heightmap::Render::RenderSettings::ColorMode_FixedColor;
    stateChanged();
}


void RenderController::
        receiveSetGreenRedColors()
{
    model()->render_settings.color_mode = Heightmap::Render::RenderSettings::ColorMode_GreenRed;
    stateChanged();
}


void RenderController::
        receiveSetGreenWhiteColors()
{
    model()->render_settings.color_mode = Heightmap::Render::RenderSettings::ColorMode_GreenWhite;
    stateChanged();
}


void RenderController::
        receiveSetGreenColors()
{
    model()->render_settings.color_mode = Heightmap::Render::RenderSettings::ColorMode_Green;
    stateChanged();
}


void RenderController::
        receiveToogleHeightlines(bool value)
{
    model()->render_settings.draw_contour_plot = value;
    stateChanged();
}


void RenderController::
        receiveToggleOrientation(bool value)
{
    model()->render_settings.left_handed_axes = !value;

    graphicsview->setLayoutDirection( value
                                            ? QBoxLayout::RightToLeft
                                            : QBoxLayout::TopToBottom );

    stateChanged();
}


void RenderController::
        receiveTogglePiano(bool value)
{
    model()->render_settings.draw_piano = value;

    ::Ui::MainWindow* ui = getItems();

    ui->actionToggle_cursor_marker->setEnabled( ui->actionToggle_t_grid->isChecked() || ui->actionToggle_hz_grid->isChecked() || ui->actionToggle_piano_grid->isChecked() );

    stateChanged();
}


void RenderController::
        receiveToggleHz(bool value)
{
    model()->render_settings.draw_hz = value;

    ::Ui::MainWindow* ui = getItems();

    ui->actionToggle_cursor_marker->setEnabled( ui->actionToggle_t_grid->isChecked() || ui->actionToggle_hz_grid->isChecked() || ui->actionToggle_piano_grid->isChecked() );

    stateChanged();
}


void RenderController::
        receiveToggleTAxis(bool value)
{
    model()->render_settings.draw_t = value;

    ::Ui::MainWindow* ui = getItems();

    ui->actionToggle_cursor_marker->setEnabled( ui->actionToggle_t_grid->isChecked() || ui->actionToggle_hz_grid->isChecked() || ui->actionToggle_piano_grid->isChecked() );

    stateChanged();
}


void RenderController::
    receiveToggleCursorMarker(bool value)
{
    model()->render_settings.draw_cursor_marker = value;
    stateChanged();
}


void RenderController::
        transformChanged()
{
    bool isCwt = dynamic_cast<const Tfr::Cwt*>(currentTransform().get ());

    if (isCwt)
    {
        float scales_per_octave = model()->transform_descs ()->getParam<Tfr::Cwt>().scales_per_octave ();
        tf_resolution->setValue ( scales_per_octave );
    }
    else
    {
        int chunk_size = model()->transform_descs ()->getParam<Tfr::StftDesc>().chunk_size ();
        tf_resolution->setValue ( chunk_size );
    }

    this->yscale->setValue( model()->render_settings.y_scale );

    // keep buttons in sync
    ::Ui::MainWindow* ui = getItems();
    if (model()->render_settings.draw_piano)  hzmarker->setCheckedAction( ui->actionToggle_piano_grid );
    if (model()->render_settings.draw_hz)  hzmarker->setCheckedAction( ui->actionToggle_hz_grid );
    switch( model()->render_settings.color_mode )
    {
    case Heightmap::Render::RenderSettings::ColorMode_Rainbow: color->setCheckedAction(ui->actionSet_rainbow_colors); break;
    case Heightmap::Render::RenderSettings::ColorMode_Grayscale: color->setCheckedAction(ui->actionSet_grayscale); break;
    case Heightmap::Render::RenderSettings::ColorMode_BlackGrayscale: color->setCheckedAction(ui->actionSet_blackgrayscale); break;
    case Heightmap::Render::RenderSettings::ColorMode_FixedColor: color->setCheckedAction(ui->actionSet_colorscale); break;
    case Heightmap::Render::RenderSettings::ColorMode_GreenRed: color->setCheckedAction(ui->actionSet_greenred_colors); break;
    case Heightmap::Render::RenderSettings::ColorMode_GreenWhite: color->setCheckedAction(ui->actionSet_greenwhite_colors); break;
    case Heightmap::Render::RenderSettings::ColorMode_Green: color->setCheckedAction(ui->actionSet_green_colors); break;
    }
    ui->actionSet_contour_plot->setChecked(model()->render_settings.draw_contour_plot);
    ui->actionToggleOrientation->setChecked(!model()->render_settings.left_handed_axes);
}


void RenderController::
        receiveSetYScale( qreal value )
{
    // Keep in sync with transformChanged()
    //float f = 2.f * value / yscale->maximum() - 1.f;
    model()->render_settings.y_scale = value; //exp( 8.f*f*f * (f>0?1:-1));

    stateChanged();

    yscale->setToolTip(QString("Intensity level %1").arg(model()->render_settings.y_scale));
}


void RenderController::
        receiveSetYBottom( qreal value )
{
    model()->render_settings.y_offset = value;

    stateChanged();

    ybottom->setToolTip(QString("Offset %1").arg(model()->render_settings.y_offset));
}


void RenderController::
        receiveSetTimeFrequencyResolution( qreal value )
{
    bool isCwt = dynamic_cast<const Tfr::Cwt*>(currentTransform().get ());
    if (isCwt)
        model()->transform_descs ()->getParam<Tfr::Cwt>().scales_per_octave ( value );
    else
        model()->transform_descs ()->getParam<Tfr::StftDesc>().set_approximate_chunk_size( value );

    stateChanged();

    view->emitTransformChanged();

    // updateTransformDesc is called each frame
}


void RenderController::yscaleIncrease()
{    yscale->triggerAction( QAbstractSlider::SliderSingleStepAdd ); stateChanged(); }
void RenderController::yscaleDecrease()
{    yscale->triggerAction( QAbstractSlider::SliderSingleStepSub ); stateChanged(); }
void RenderController::tfresolutionIncrease()
{    tf_resolution->triggerAction( QAbstractSlider::SliderPageStepAdd ); stateChanged(); }
void RenderController::tfresolutionDecrease()
{    tf_resolution->triggerAction( QAbstractSlider::SliderPageStepSub ); stateChanged(); }


void RenderController::
        updateTransformDesc()
{
    {
        // don't bother about proper timesteps
        auto& log_scale = model()->render_settings.log_scale;
        log_scale.TimeStep (0.05f);
        if (log_scale != &log_scale)
            view->redraw ();
    }

    Tfr::TransformDesc::ptr t = currentTransform();
    Tfr::TransformDesc::ptr newuseroptions;

    if (!t)
        return;

    {
        auto tfr_map = model()->tfr_mapping ().write ();

        // If the transform currently in use differs from the transform settings
        // that should be used, change the transform.
        Tfr::TransformDesc::ptr useroptions = tfr_map->transform_desc();

        // If there is a transform but no tfr_map transform_desc it means that
        // there is a bug (or at least some continued refactoring todo). Update
        // tfr_map
        EXCEPTION_ASSERT (useroptions);

        newuseroptions = model()->transform_descs ().read ()->cloneType(typeid(*useroptions));
        EXCEPTION_ASSERT (newuseroptions);

        if (*newuseroptions != *useroptions)
          {
            model()->block_update_queue->clear();
            tfr_map->transform_desc( newuseroptions );
          }
    }

    if (*t != *newuseroptions)
        setCurrentFilterTransform(newuseroptions);
}


void RenderController::
        setCurrentFilterTransform( Tfr::TransformDesc::ptr t )
{
    {
        auto td = model()->transform_descs ().write ();
        Tfr::StftDesc& s = td->getParam<Tfr::StftDesc>();
        Tfr::Cwt& c = td->getParam<Tfr::Cwt>();

        // TODO add tfr resolution string to TransformDesc
        bool isCwt = dynamic_cast<const Tfr::Cwt*>(t.get ());
        if (isCwt)
            tf_resolution->setToolTip(QString("Time/frequency resolution\nMorlet std: %1\nScales per octave").arg(c.sigma(), 0, 'f', 1));
        else
            tf_resolution->setToolTip(QString("Time/frequency resolution\nSTFT window: %1 samples").arg(s.chunk_size()));
    }

    model()->set_transform_desc (t);
}


void RenderController::
        setBlockFilter(Heightmap::MergeChunkDesc::ptr mcdp, Tfr::TransformDesc::ptr transform_desc)
{
    // Wire it up to a FilterDesc
    Heightmap::Update::UpdateProducerDesc* cbfd;
    Tfr::ChunkFilterDesc::ptr kernel(cbfd
            = new Heightmap::Update::UpdateProducerDesc(model()->block_update_queue, model()->tfr_mapping ()));
    cbfd->setMergeChunkDesc( mcdp );
    kernel.write ()->transformDesc(transform_desc);
    setBlockFilter( kernel );
}


void RenderController::
        setBlockFilter(Tfr::ChunkFilterDesc::ptr kernel)
{
    Tfr::TransformOperationDesc::ptr adapter( new Tfr::TransformOperationDesc(kernel));
    // Ambiguity
    // Tfr::TransformOperationDesc defines a current transformDesc
    // VisualizationParams also defines a current transformDesc

    std::string oldTransform_name = currentTransform() ? vartype(*currentTransform()) : "(null)";
    bool wasCwt = dynamic_cast<const Tfr::Cwt*>(currentTransform().get ());

    model()->set_filter (adapter);

    EXCEPTION_ASSERT( currentTransform() );

    stateChanged();

    ::Ui::MainWindow* ui = getItems();

    ui->actionToggle_piano_grid->setVisible( true );
    hz_scale->setEnabled( true );

    bool isCwt = dynamic_cast<const Tfr::Cwt*>(currentTransform().get ());

    if (isCwt || wasCwt) {
        auto td = model()->transform_descs ().write ();
        Tfr::StftDesc& s = td->getParam<Tfr::StftDesc>();
        Tfr::Cwt& c = td->getParam<Tfr::Cwt>();

        float wavelet_default_time_support = c.wavelet_default_time_support();
        float wavelet_fast_time_support = c.wavelet_time_support();
        c.wavelet_time_support(wavelet_default_time_support);

        if (isCwt && !wasCwt)
        {
            tf_resolution->setRange (2, 40);
            tf_resolution->setDecimals (1);
            c.scales_per_octave (s.chunk_size ()/(c.wavelet_time_support_samples()/c.wavelet_time_support()/c.scales_per_octave ()));
            // transformChanged updates value accordingly
        }

        if (!isCwt && wasCwt)
        {
            tf_resolution->setRange (1<<5, 1<<20, Widgets::ValueSlider::Logaritmic);
            tf_resolution->setDecimals (0);
            s.set_approximate_chunk_size( c.wavelet_time_support_samples()/c.wavelet_time_support() );
            // transformChanged updates value accordingly
        }

        c.wavelet_fast_time_support( wavelet_fast_time_support );
    }

    // abort target needs
    auto needs = model ()->target_marker ()->target_needs ();
    auto step = needs->step ().lock (); // lock weak_ptr
    needs->updateNeeds (Signal::Intervals());
    int sleep_ms = 1000;
    Timer t;
    for (int i=0; i<sleep_ms && !Signal::Processing::Step::sleepWhileTasks (step.read(), 1); i++)
    {
        model ()->block_update_queue->clear ();
    }

    if (!Signal::Processing::Step::sleepWhileTasks (step.read(), 1))
        Log("%s didn't finish in %g ms, changing anyway to %s") % oldTransform_name % t.elapsed () % vartype(*currentTransform ());
    else
        Log("%s finished in %g ms, changing to %s") % oldTransform_name % t.elapsed () % vartype(*currentTransform ());

    // then change the tfr_mapping
    model()->tfr_mapping ().write ()->transform_desc( currentTransform()->copy() );

    view->emitTransformChanged();
    //return ps;
}


Tfr::TransformDesc::ptr RenderController::
        currentTransform()
{
    return model()->transform_desc ();
/*
//Use Signal::Processing namespace
    Tfr::Filter* f = currentFilter();
    return f?f->transform().get():0;
*/
}


float RenderController::
        headSampleRate()
{
    return project->extent ().sample_rate.get_value_or (1);
//    return project->head->head_source()->sample_rate();
}


float RenderController::
        currentTransformMinHz()
{
    Tfr::TransformDesc::ptr t = currentTransform();
    EXCEPTION_ASSERT(t);
    return t->freqAxis(headSampleRate()).min_hz;
}


::Ui::MainWindow* RenderController::
        getItems()
{
    ::Ui::SaweMainWindow* main = dynamic_cast< ::Ui::SaweMainWindow*>(project->mainWindow());
    return main->getItems();
}


void RenderController::
        receiveSetTransform_Cwt()
{
/*
    Heightmap::CwtToBlock* cwtblock = new Heightmap::CwtToBlock(model()->tfr_map (), model()->renderer.get());
    cwtblock->complex_info = Heightmap::ComplexInfo_Amplitude_Non_Weighted;

    setBlockFilter( cwtblock );
*/
    // Setup the kernel that will take the transform data and create an image
    Heightmap::MergeChunkDesc::ptr mcdp(new Heightmap::TfrMappings::CwtBlockFilterDesc(Heightmap::ComplexInfo_Amplitude_Non_Weighted));

    // Get a copy of the transform to use
    Tfr::TransformDesc::ptr transform_desc = model()->transform_descs ().write ()->getParam<Tfr::Cwt>().copy();

    setBlockFilter(mcdp, transform_desc);
}


void RenderController::
        receiveSetTransform_Stft()
{
    // Setup the kernel that will take the transform data and create an image
    Heightmap::MergeChunkDesc::ptr mcdp(new Heightmap::TfrMappings::StftBlockFilterDesc(model()->get_stft_block_filter_params ()));

    // Get a copy of the transform to use
    Tfr::TransformDesc::ptr transform_desc = model()->transform_descs ().write ()->getParam<Tfr::StftDesc>().copy();

    setBlockFilter(mcdp, transform_desc);
}


void RenderController::
        receiveSetTransform_Cwt_phase()
{
    // Setup the kernel that will take the transform data and create an image
    Heightmap::MergeChunkDesc::ptr mcdp(new Heightmap::TfrMappings::CwtBlockFilterDesc(Heightmap::ComplexInfo_Phase));

    // Get a copy of the transform to use
    Tfr::TransformDesc::ptr transform_desc = model()->transform_descs ()->getParam<Tfr::Cwt>().copy();

    setBlockFilter(mcdp, transform_desc);
}


void RenderController::
        receiveSetTransform_Cwt_weight()
{
    // Setup the kernel that will take the transform data and create an image
    Heightmap::MergeChunkDesc::ptr mcdp(new Heightmap::TfrMappings::CwtBlockFilterDesc(Heightmap::ComplexInfo_Amplitude_Weighted));

    // Get a copy of the transform to use
    Tfr::TransformDesc::ptr transform_desc = model()->transform_descs ()->getParam<Tfr::Cwt>().copy();

    setBlockFilter(mcdp, transform_desc);
}


void RenderController::
        receiveSetTransform_Cepstrum()
{
    // Setup the kernel that will take the transform data and create an image
    Heightmap::MergeChunkDesc::ptr mcdp(new Heightmap::TfrMappings::CepstrumBlockFilterDesc);

    // Get a copy of the transform to use
    Tfr::TransformDesc::ptr transform_desc = model()->transform_descs ()->getParam<Tfr::CepstrumDesc>().copy();

    setBlockFilter(mcdp, transform_desc);
}


void RenderController::
        receiveLinearScale()
{
    float fs = headSampleRate();

    Heightmap::FreqAxis fa;
    fa.setLinear( fs );

    if (currentTransform() && fa.min_hz < currentTransformMinHz())
    {
        fa.min_hz = currentTransformMinHz();
        fa.f_step = fs/2 - fa.min_hz;
    }

    model()->display_scale( fa );

    view->emitAxisChanged();
    stateChanged();
}


void RenderController::
        receiveWaveformScale()
{
    float maxvalue = 1;
    float minvalue = -1;

    Heightmap::FreqAxis fa;
    fa.setWaveform (minvalue, maxvalue);

    model()->display_scale( fa );

    view->emitAxisChanged();
    stateChanged();
}


void RenderController::
        receiveLogScale()
{
    float fs = headSampleRate();

    Heightmap::FreqAxis fa;

    {
        auto td = model()->transform_descs ().write ();
        Tfr::Cwt& cwt = td->getParam<Tfr::Cwt>();
        fa.setLogarithmic(
                cwt.get_wanted_min_hz (fs),
                cwt.get_max_hz (fs) );

        if (currentTransform() && fa.min_hz < currentTransformMinHz())
        {
            // Happens typically when currentTransform is a cepstrum transform with a short window size
            fa.setLogarithmic(
                    currentTransformMinHz(),
                    cwt.get_max_hz(fs) );
        }
    }

    model()->display_scale( fa );

    view->emitAxisChanged();
    stateChanged();
}


void RenderController::
        receiveCepstraScale()
{
    float fs = headSampleRate();

    Heightmap::FreqAxis fa;
    fa.setQuefrencyNormalized( fs, model()->transform_descs ()->getParam<Tfr::CepstrumDesc>().chunk_size() );

    if (currentTransform() && fa.min_hz < currentTransformMinHz())
    {
        // min_hz = 2*fs/window_size;
        // window_size = 2*fs/min_hz;
        fa.setQuefrencyNormalized( fs, 2*fs/currentTransformMinHz() );
    }

    model()->display_scale( fa );

    view->emitAxisChanged();
    stateChanged();
}


void RenderController::
        receiveLinearAmplitude()
{
    model()->render_settings.log_scale = 0;
//    model()->amplitude_axis( Heightmap::AmplitudeAxis_Linear );
//    view->emitAxisChanged();
    stateChanged();
}


void RenderController::
        receiveLogAmplitude()
{
    model()->render_settings.log_scale = 1;
//    model()->amplitude_axis( Heightmap::AmplitudeAxis_Logarithmic );
//    view->emitAxisChanged();
    stateChanged();
}


void RenderController::
        receiveFifthAmplitude()
{
    model()->amplitude_axis( Heightmap::AmplitudeAxis_5thRoot );
    view->emitAxisChanged();
    stateChanged();
}


RenderModel *RenderController::
        model()
{
    EXCEPTION_ASSERT( view );
    return view->model;
}


void RenderController::
        setupGui()
{
    ::Ui::SaweMainWindow* main = dynamic_cast< ::Ui::SaweMainWindow*>(project->mainWindow());
    toolbar_render = new Support::ToolBar(main);
    toolbar_render->setObjectName(QString::fromUtf8("toolBarRenderController"));
    toolbar_render->setWindowTitle("tool bar");
    toolbar_render->setEnabled(true);
    toolbar_render->setContextMenuPolicy(Qt::NoContextMenu);
    toolbar_render->setToolButtonStyle(Qt::ToolButtonIconOnly);
    main->addToolBar(Qt::BottomToolBarArea, toolbar_render);

    // Find Qt Creator managed actions
    ::Ui::MainWindow* ui = main->getItems();

    connect(ui->actionToggleTransformToolBox, SIGNAL(toggled(bool)), toolbar_render, SLOT(setVisible(bool)));
    connect((Support::ToolBar*)toolbar_render, SIGNAL(visibleChanged(bool)), ui->actionToggleTransformToolBox, SLOT(setChecked(bool)));

    main->installEventFilter( this );


    // ComboBoxAction* hzmarker
    {   hzmarker = new ComboBoxAction(toolbar_render);
        hzmarker->setObjectName("hzmarker");
        hzmarker->addActionItem( ui->actionToggle_hz_grid );
        hzmarker->addActionItem( ui->actionToggle_piano_grid );
        toolbar_render->addWidget( hzmarker );
        toolbar_render->addAction( ui->actionToggle_t_grid );
        toolbar_render->addAction( ui->actionToggle_cursor_marker );

        connect(ui->actionToggle_piano_grid, SIGNAL(toggled(bool)), SLOT(receiveTogglePiano(bool)));
        connect(ui->actionToggle_hz_grid, SIGNAL(toggled(bool)), SLOT(receiveToggleHz(bool)));
        connect(ui->actionToggle_t_grid, SIGNAL(toggled(bool)), SLOT(receiveToggleTAxis(bool)));
        connect(ui->actionToggle_cursor_marker, SIGNAL(toggled(bool)), SLOT(receiveToggleCursorMarker(bool)));
    }

    connect(ui->actionResetGraphics, SIGNAL(triggered()), view, SLOT(clearCaches()));
    connect( project->commandInvoker(), SIGNAL(projectChanged(const Command*)), view, SLOT(redraw()));
    connect( Sawe::Application::global_ptr(), SIGNAL(clearCachesSignal()), view, SLOT(clearCaches()) );
    view->model->render_settings.drawcrosseswhen0 = Sawe::Configuration::version().empty();


    // ComboBoxAction* color
    {   color = new ComboBoxAction(toolbar_render);
        color->setObjectName("ComboBoxActioncolor");
        color->decheckable( false );
#if !defined(TARGET_hast)
        color->addActionItem( ui->actionSet_colorscale );
        color->addActionItem( ui->actionSet_greenred_colors );
        color->addActionItem( ui->actionSet_rainbow_colors );
#endif
        color->addActionItem( ui->actionSet_greenwhite_colors );
        color->addActionItem( ui->actionSet_green_colors );
        color->addActionItem( ui->actionSet_grayscale );
        color->addActionItem( ui->actionSet_blackgrayscale );
        toolbar_render->addWidget( color );

        foreach(QAction*a,color->actions ())
            a->setChecked (false);

        connect(ui->actionSet_rainbow_colors, SIGNAL(triggered()), SLOT(receiveSetRainbowColors()));
        connect(ui->actionSet_grayscale, SIGNAL(triggered()), SLOT(receiveSetGrayscaleColors()));
        connect(ui->actionSet_blackgrayscale, SIGNAL(triggered()), SLOT(receiveSetBlackGrayscaleColors()));
        connect(ui->actionSet_colorscale, SIGNAL(triggered()), SLOT(receiveSetColorscaleColors()));
        connect(ui->actionSet_greenred_colors, SIGNAL(triggered()), SLOT(receiveSetGreenRedColors()));
        connect(ui->actionSet_greenwhite_colors, SIGNAL(triggered()), SLOT(receiveSetGreenWhiteColors()));
        connect(ui->actionSet_green_colors, SIGNAL(triggered()), SLOT(receiveSetGreenColors()));

#if defined(TARGET_hast)
        color->setCheckedAction(ui->actionSet_greenwhite_colors);
        ui->actionSet_greenwhite_colors->trigger();
#else
        color->setCheckedAction(ui->actionSet_colorscale);
        ui->actionSet_colorscale->trigger();
#endif
    }

    // ComboBoxAction* channels
    {   channelselector = new QToolButton(toolbar_render);
        channelselector->setVisible (false);
        channelselector->setObjectName("channelselector");
        channelselector->setCheckable( false );
        channelselector->setText("Channels");
        channelselector->setContextMenuPolicy( Qt::ActionsContextMenu );
        channelselector->setToolTip("Press to get a list of channels (or right click)");
        toolbar_render->addWidget( channelselector );
    }

    // QAction *actionSet_heightlines
    toolbar_render->addAction(ui->actionSet_contour_plot);
    connect(ui->actionSet_contour_plot, SIGNAL(toggled(bool)), SLOT(receiveToogleHeightlines(bool)));

    toolbar_render->addAction(ui->actionToggleOrientation);
    connect(ui->actionToggleOrientation, SIGNAL(toggled(bool)), SLOT(receiveToggleOrientation(bool)));

    // ComboBoxAction* transform
    {   connect(ui->actionTransform_Cwt, SIGNAL(triggered()), SLOT(receiveSetTransform_Cwt()));
        connect(ui->actionTransform_Stft, SIGNAL(triggered()), SLOT(receiveSetTransform_Stft()));
//        connect(ui->actionTransform_Cwt_phase, SIGNAL(triggered()), SLOT(receiveSetTransform_Cwt_phase()));
//        connect(ui->actionTransform_Cwt_weight, SIGNAL(triggered()), SLOT(receiveSetTransform_Cwt_weight()));
        connect(ui->actionTransform_Cepstrum, SIGNAL(triggered()), SLOT(receiveSetTransform_Cepstrum()));

        transform = new ComboBoxAction(toolbar_render);
        transform->setObjectName("ComboBoxActiontransform");
        transform->addActionItem( ui->actionTransform_Stft );
        transform->addActionItem( ui->actionTransform_Cwt );

        if (!Sawe::Configuration::feature("stable")) {
            transform->addActionItem( ui->actionTransform_Cepstrum );
        }

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


    // ComboBoxAction* hz-scale
    {   waveformScale = new QAction( toolbar_render );
        linearScale = new QAction( toolbar_render );
        logScale = new QAction( toolbar_render );
        cepstraScale = new QAction( toolbar_render );

        waveformScale->setText("Waveform");
        linearScale->setText("Linear scale");
        logScale->setText("Logarithmic scale");
        cepstraScale->setText("Cepstra scale");

        // for serialization
        waveformScale->setObjectName("waveformScale");
        linearScale->setObjectName("linearScale");
        logScale->setObjectName("logScale");
        cepstraScale->setObjectName("cepstraScale");

        waveformScale->setVisible (false);
        linearScale->setCheckable( true );
        logScale->setCheckable( true );
        cepstraScale->setCheckable( true );

        connect(waveformScale, SIGNAL(triggered()), SLOT(receiveWaveformScale()));
        connect(linearScale, SIGNAL(triggered()), SLOT(receiveLinearScale()));
        connect(logScale, SIGNAL(triggered()), SLOT(receiveLogScale()));
        connect(cepstraScale, SIGNAL(triggered()), SLOT(receiveCepstraScale()));

        hz_scale = new ComboBoxAction();
        hz_scale->setObjectName("hz_scale");
        //hz_scale->addActionItem( waveformScale );
        hz_scale->addActionItem( linearScale );
        hz_scale->addActionItem( logScale );
        if (!Sawe::Configuration::feature("stable")) {
            hz_scale->addActionItem( cepstraScale );
        }
        hz_scale->decheckable( false );
        hz_scale_action = toolbar_render->addWidget( hz_scale );

        unsigned k=0;
        foreach( QAction* a, hz_scale->actions())
        {
            a->setShortcut(QString("Ctrl+") + ('1' + k++));
        }
        logScale->trigger();
    }


    // ComboBoxAction* amplitude-scale
    {   QAction* linearAmplitude = new QAction( toolbar_render );
        QAction* logAmpltidue = new QAction( toolbar_render );
        QAction* fifthAmpltidue = new QAction( toolbar_render );

        linearAmplitude->setText("Linear amplitude");
        logAmpltidue->setText("Logarithmic amplitude");
        fifthAmpltidue->setText("|A|^(1/5) amplitude");

        // for serialization
        linearAmplitude->setObjectName("linearAmplitude");
        logAmpltidue->setObjectName("logAmpltidue");
        fifthAmpltidue->setObjectName("fifthAmpltidue");

        linearAmplitude->setCheckable( true );
        logAmpltidue->setCheckable( true );
        fifthAmpltidue->setCheckable( true );

        connect(linearAmplitude, SIGNAL(triggered()), SLOT(receiveLinearAmplitude()));
        connect(logAmpltidue, SIGNAL(triggered()), SLOT(receiveLogAmplitude()));
        connect(fifthAmpltidue, SIGNAL(triggered()), SLOT(receiveFifthAmplitude()));

        amplitude_scale = new ComboBoxAction();
        amplitude_scale->addActionItem( linearAmplitude );
        amplitude_scale->addActionItem( logAmpltidue );
//        amplitude_scale->addActionItem( fifthAmpltidue );
        amplitude_scale->decheckable( false );
        amplitude_scale->setDefaultAction (logAmpltidue);
        amplitude_scale_action = toolbar_render->addWidget( amplitude_scale );

        unsigned k=0;
        foreach( QAction* a, amplitude_scale->actions())
        {
            a->setShortcut(QString("Alt+") + ('1' + k++));
        }
        linearAmplitude->trigger();
    }


    // QSlider * yscale
    {   yscale = new Widgets::ValueSlider( toolbar_render );
        yscale->setObjectName("yscale");
        yscale->setOrientation( Qt::Horizontal );
        yscale->setRange (0.0003, 2000, Widgets::ValueSlider::LogaritmicZeroMin );
        yscale->setValue ( 1 );
        yscale->setDecimals (2);
        yscale->setToolTip( "Intensity level" );
        yscale->setSliderSize ( 300 );
        toolbar_render->addWidget( yscale );

        connect(yscale, SIGNAL(valueChanged(qreal)), SLOT(receiveSetYScale(qreal)));
        receiveSetYScale(yscale->value());

        QAction* yscaleIncrease = new QAction(yscale);
        QAction* yscaleDecrease = new QAction(yscale);

        yscaleIncrease->setShortcut(QString("Alt+Up"));
        yscaleDecrease->setShortcut(QString("Alt+Down"));

        connect(yscaleIncrease, SIGNAL(triggered()), SLOT(yscaleIncrease()));
        connect(yscaleDecrease, SIGNAL(triggered()), SLOT(yscaleDecrease()));

        yscale->addAction( yscaleIncrease );
        yscale->addAction( yscaleDecrease );
    }

    // QSlider * ybottom
    {   ybottom = new Widgets::ValueSlider( toolbar_render );
        ybottom->setObjectName("ybottom");
        ybottom->setOrientation( Qt::Horizontal );
        ybottom->setRange (-1, 1, Widgets::ValueSlider::Linear );
        ybottom->setValue (0);
        ybottom->setDecimals (2);
        ybottom->setToolTip( "Offset" );
        ybottom->setSliderSize ( 300 );
        toolbar_render->addWidget( ybottom );

        connect(ybottom, SIGNAL(valueChanged(qreal)), SLOT(receiveSetYBottom(qreal)));
        receiveSetYBottom(ybottom->value());
    }

    // QSlider * tf_resolution
    {   tf_resolution = new Widgets::ValueSlider( toolbar_render );
        tf_resolution->setObjectName ("tf_slider");
        tf_resolution->setRange (1<<5, 1<<20, Widgets::ValueSlider::Logaritmic);
        tf_resolution->setValue ( 4096 );
        tf_resolution->setToolTip ("Window size (time/frequency resolution) ");
        tf_resolution->setSliderSize ( 300 );
        tf_resolution->updateLineEditOnValueChanged (false); // RenderController does that instead
        tf_resolution_action = toolbar_render->addWidget (tf_resolution);

        connect(tf_resolution, SIGNAL(valueChanged(qreal)), SLOT(receiveSetTimeFrequencyResolution(qreal)));
        receiveSetTimeFrequencyResolution(tf_resolution->value());

        QAction* tfresolutionIncrease = new QAction(yscale);
        QAction* tfresolutionDecrease = new QAction(yscale);

        tfresolutionIncrease->setShortcut(QString("Alt+Right"));
        tfresolutionDecrease->setShortcut(QString("Alt+Left"));

        connect(tfresolutionIncrease, SIGNAL(triggered()), SLOT(tfresolutionIncrease()));
        connect(tfresolutionDecrease, SIGNAL(triggered()), SLOT(tfresolutionDecrease()));

        tf_resolution->addAction( tfresolutionIncrease );
        tf_resolution->addAction( tfresolutionDecrease );
    }

    connect(this->view.data(), SIGNAL(transformChanged()), SLOT(updateFreqAxis()), Qt::QueuedConnection);
    connect(this->view.data(), SIGNAL(transformChanged()), SLOT(updateAmplitudeAxis()), Qt::QueuedConnection);
    connect(this->view.data(), SIGNAL(transformChanged()), SLOT(updateChannels()), Qt::QueuedConnection);
    connect(this->view.data(), SIGNAL(transformChanged()), SLOT(transformChanged()), Qt::QueuedConnection);

    // Validate the current transform used for rendering before each rendering.
    // Lazy updating with easier and direct access to a transform description.
    connect(this->view.data(), SIGNAL(prePaint()), SLOT(updateTransformDesc()), Qt::DirectConnection);

    // Release cuda buffers and disconnect them from OpenGL before destroying
    // OpenGL rendering context. Just good housekeeping.
    connect(this->view.data(), SIGNAL(destroying()), SLOT(deleteTarget()), Qt::DirectConnection);
    connect(Sawe::Application::global_ptr(), SIGNAL(clearCachesSignal()), SLOT(clearCaches()), Qt::DirectConnection);

    // Create the OpenGL rendering context early. Because we want to create the
    // cuda context (in main.cpp) and bind it to an OpenGL context before the
    // context is required to be created by lazy initialization when painting
    // the widget
    view->glwidget = new QGLWidget( 0, Sawe::Application::shared_glwidget(), Qt::WindowFlags(0) );
    view->glwidget->setObjectName( QString("glwidget %1").arg((size_t)this));
    view->glwidget->makeCurrent();
    model()->render_settings.dpifactor = project->mainWindow ()->devicePixelRatio ();

    GraphicsScene* scene = new GraphicsScene(view);
    connect(this->view.data (), SIGNAL(redrawSignal()), scene, SLOT(redraw()));
    graphicsview = new GraphicsView(scene);
    graphicsview->setViewport(view->glwidget);
    view->glwidget->makeCurrent(); // setViewport makes the glwidget loose context, take it back
    this->tool_selector = graphicsview->toolSelector(0, project->commandInvoker());

    // UpdateConsumer takes view->glwidget as parent, could use multiple updateconsumers ...
    int n_update_consumers = 1;
    for (int i=0; i<n_update_consumers; i++)
    {
        auto uc = new Heightmap::Update::UpdateConsumer(view->glwidget, model()->block_update_queue);
        connect(uc, SIGNAL(didUpdate()), view.data (), SLOT(redraw()));
    }

    main->centralWidget()->layout()->setMargin(0);
    main->centralWidget()->layout()->addWidget(graphicsview);
    main->centralWidget()->setFocus ();

    view->emitTransformChanged();

#ifdef TARGET_hast
    toolbarWidgetVisible(channelselector, false);
    toolbarWidgetVisible(tf_resolution, false);
    toolbarWidgetVisible(amplitude_scale, false);
    toolbarWidgetVisible(hz_scale, false);
    toolbarWidgetVisible(transform, false);
    ui->actionToggleOrientation->setChecked(true);
    ui->actionToggleOrientation->setVisible(false);
#endif
}


void RenderController::
        deleteTarget()
{
//    model()->block_update_queue.reset ();
//    model()->renderer.reset();
//    clearCaches();
}


void RenderController::
        clearCaches()
{
    foreach( const Heightmap::Collection::ptr& collection, model()->collections() )
        collection.write ()->clear();
}


void RenderController::
        updateFreqAxis()
{
    switch(model()->display_scale().axis_scale)
    {
    case Tfr::AxisScale_Waveform:
        receiveWaveformScale();
        break;

    case Tfr::AxisScale_Linear:
        receiveLinearScale();
        break;

    case Tfr::AxisScale_Logarithmic:
        receiveLogScale();
        break;

    case Tfr::AxisScale_Quefrency:
        receiveCepstraScale();
        break;

    default:
        EXCEPTION_ASSERT(false);
        break;
    }
}


void RenderController::
        updateAmplitudeAxis()
{
//    Tfr::Filter* filter = model()->block_filter ();
//    Heightmap::StftToBlock* stftblock = dynamic_cast<Heightmap::StftToBlock*>( filter );

/*    switch (model()->amplitude_axis ())
    {
    case Heightmap::AmplitudeAxis_Real:
        break;
    default:
        stftblock->freqNormalization.reset ();
        break;
    }*/

    view->emitAxisChanged();
    stateChanged();

//    switch (model()->amplitude_axis ())
/*    switch (model()->amplitude_axis ())
    {
    case Heightmap::AmplitudeAxis_Linear:
        receiveLinearAmplitude();
        break;

    case Heightmap::AmplitudeAxis_Logarithmic:
        receiveLogAmplitude();
        break;

    case Heightmap::AmplitudeAxis_5thRoot:
        receiveFifthAmplitude();
        break;

    case Heightmap::AmplitudeAxis_Real:
        receiveLinearAmplitude();
        break;

    default:
        break;
    }*/
}


void RenderController::
        updateChannels()
{
/*
//Use Signal::Processing namespace
    Signal::RerouteChannels* channels = model()->renderSignalTarget->channels();
    unsigned N = channels->source()->num_channels();
*/
    unsigned  N = project->extent().number_of_channels.get_value_or (0);
    if (model()->tfr_mapping ().read ()->channels() != (int)N)
        model()->recompute_extent ();
    for (unsigned i=0; i<N; ++i)
    {
        foreach (QAction* a, channelselector->actions())
        {
            if (!a)
                continue;
            if (a->data().toUInt() >= N)
                delete a;
        }

        for (unsigned i = channelselector->actions().size(); i<N; ++i)
        {
            QAction* a = new QAction(channelselector);
            a->setText(QString("Channel %1").arg(i));
            a->setData( i );
            a->setCheckable( true );
            a->setChecked( false );
            connect(a, SIGNAL(toggled(bool)), SLOT( reroute() ));
            channelselector->addAction( a );

            a->setChecked( true ); // invokes reroute
        }
    }
}


void RenderController::
        reroute()
{
    //Signal::RerouteChannels* channels = model()->renderSignalTarget->channels();
    foreach (QAction* o, channelselector->actions())
    {
        unsigned c = o->data().toUInt();
        //channels->map(c, o->isChecked() ? c : Signal::RerouteChannels::NOTHING );
        if (model()->collections()[c].read ()->isVisible() != o->isChecked())
        {
            model()->collections()[c].write ()->setVisible( o->isChecked() );
            stateChanged();
        }
    }
}


bool RenderController::
        eventFilter(QObject *o, QEvent *e)
{
    //eventFilter is called a lot, do the most simple test possible to
    // determine if we're interested or not
    if (e->type() == QEvent::FocusIn || e->type() == QEvent::FocusOut)
    {
        if (project->mainWindow() == o)
        {
            if (e->type() == QEvent::FocusIn)
                windowGotFocus();
            else
                windowLostFocus();
        }
    }

    // this eventFilter doesn't block any events
    return false;
}


void RenderController::
        windowLostFocus()
{
/*
    // Use Signal::Processing namespace
    project->worker.min_fps( 20 );
*/
}


void RenderController::
        windowGotFocus()
{
/*
    // Use Signal::Processing namespace
    project->worker.min_fps( 4 );
*/
}


void RenderController::
        toolbarWidgetVisible(QWidget* w, bool v)
{
    toolbarWidgetVisible(toolbar_render, w, v);
}


void RenderController::
        toolbarWidgetVisible(QToolBar* t, QWidget* w, bool v)
{
    foreach(QAction*a, t->actions())
        if (t->widgetForAction(a) == w)
            a->setVisible(v);
}

} // namespace Tools

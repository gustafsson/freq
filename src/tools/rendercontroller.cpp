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
#include "heightmap/blockfilter.h"
#include "heightmap/renderer.h"
#include "heightmap/chunkblockfilter.h"
#include "heightmap/tfrmappings/stftblockfilter.h"
#include "heightmap/tfrmappings/cwtblockfilter.h"
#include "signal/postsink.h"
#include "tfr/cwt.h"
#include "tfr/stft.h"
#include "tfr/cepstrum.h"
#include "tfr/drawnwaveform.h"
#include "graphicsview.h"
#include "sawe/application.h"
#include "signal/buffersource.h"
#include "signal/reroutechannels.h"
#include "signal/oldoperationwrapper.h"
#include "tools/support/operation-composite.h"
#include "tools/support/renderoperation.h"
#include "tools/support/renderviewupdateadapter.h"
#include "sawe/configuration.h"

// gpumisc
#include "demangle.h"
#include "computationkernel.h"

// Qt
#include <QSlider>
#include <QGraphicsView>
#include <QResizeEvent>
#include <QMetaClassInfo>
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
        RenderController( QPointer<RenderView> view )
            :
            view(view),
            toolbar_render(0),
            hz_scale(0),
            linearScale(0),
            logScale(0),
            cepstraScale(0),
            amplitude_scale(0),
            hzmarker(0),
            color(0),
            transform(0)
{
    Support::RenderViewUpdateAdapter* rvup;
    Support::RenderOperationDesc::RenderTarget::Ptr rvu(
                rvup = new Support::RenderViewUpdateAdapter);

    connect(rvup, SIGNAL(userinput_update()), view, SLOT(userinput_update()));
    connect(rvup, SIGNAL(setLastUpdateSize(Signal::UnsignedIntervalType)), view, SLOT(setLastUpdateSize(Signal::UnsignedIntervalType)));

    model()->init(model()->project ()->processing_chain (), rvu);


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

        write1(model()->transform_descs ())->getParam<Tfr::StftDesc>().setWindow(Tfr::StftDesc::WindowType_Hann, 0.75f);
        write1(model()->transform_descs ())->getParam<Tfr::StftDesc>().enable_inverse(false);

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
    // Don't lock the UI, instead wait a moment before any change is made
    view->userinput_update();

    model()->project()->setModified();
}


void RenderController::
        receiveSetRainbowColors()
{
    model()->renderer->render_settings.color_mode = Heightmap::RenderSettings::ColorMode_Rainbow;
    stateChanged();
}


void RenderController::
        receiveSetGrayscaleColors()
{
    model()->renderer->render_settings.color_mode = Heightmap::RenderSettings::ColorMode_Grayscale;
    stateChanged();
}


void RenderController::
        receiveSetBlackGrayscaleColors()
{
    model()->renderer->render_settings.color_mode = Heightmap::RenderSettings::ColorMode_BlackGrayscale;
    stateChanged();
}


void RenderController::
        receiveSetColorscaleColors()
{
    model()->renderer->render_settings.color_mode = Heightmap::RenderSettings::ColorMode_FixedColor;
    stateChanged();
}


void RenderController::
        receiveSetGreenRedColors()
{
    model()->renderer->render_settings.color_mode = Heightmap::RenderSettings::ColorMode_GreenRed;
    stateChanged();
}


void RenderController::
        receiveSetGreenWhiteColors()
{
    model()->renderer->render_settings.color_mode = Heightmap::RenderSettings::ColorMode_GreenWhite;
    stateChanged();
}


void RenderController::
        receiveSetGreenColors()
{
    model()->renderer->render_settings.color_mode = Heightmap::RenderSettings::ColorMode_Green;
    stateChanged();
}


void RenderController::
        receiveToogleHeightlines(bool value)
{
    model()->renderer->render_settings.draw_contour_plot = value;
    stateChanged();
}


void RenderController::
        receiveToggleOrientation(bool value)
{
    model()->renderer->render_settings.left_handed_axes = !value;

    view->graphicsview->setLayoutDirection( value
                                            ? QBoxLayout::RightToLeft
                                            : QBoxLayout::TopToBottom );

    stateChanged();
}


void RenderController::
        receiveTogglePiano(bool value)
{
    model()->renderer->render_settings.draw_piano = value;

    ::Ui::MainWindow* ui = getItems();

    ui->actionToggle_cursor_marker->setEnabled( ui->actionToggle_t_grid->isChecked() || ui->actionToggle_hz_grid->isChecked() || ui->actionToggle_piano_grid->isChecked() );

    stateChanged();
}


void RenderController::
        receiveToggleHz(bool value)
{
    model()->renderer->render_settings.draw_hz = value;

    ::Ui::MainWindow* ui = getItems();

    ui->actionToggle_cursor_marker->setEnabled( ui->actionToggle_t_grid->isChecked() || ui->actionToggle_hz_grid->isChecked() || ui->actionToggle_piano_grid->isChecked() );

    stateChanged();
}


void RenderController::
        receiveToggleTAxis(bool value)
{
    model()->renderer->render_settings.draw_t = value;

    ::Ui::MainWindow* ui = getItems();

    ui->actionToggle_cursor_marker->setEnabled( ui->actionToggle_t_grid->isChecked() || ui->actionToggle_hz_grid->isChecked() || ui->actionToggle_piano_grid->isChecked() );

    stateChanged();
}


void RenderController::
    receiveToggleCursorMarker(bool value)
{
    model()->renderer->render_settings.draw_cursor_marker = value;
    stateChanged();
}


void RenderController::
        transformChanged()
{
    bool isCwt = dynamic_cast<const Tfr::Cwt*>(currentTransform().get ());

    if (isCwt)
    {
        // The Cwt TransformDesc need to know the current sample rate
        float fs = headSampleRate ();
        write1(model()->transform_descs ())->getParam<Tfr::Cwt>().set_fs(fs);

        float scales_per_octave = write1(model()->transform_descs ())->getParam<Tfr::Cwt>().scales_per_octave ();
        tf_resolution->setValue ( scales_per_octave );
    }
    else
    {
        int chunk_size = write1(model()->transform_descs ())->getParam<Tfr::StftDesc>().chunk_size ();
        tf_resolution->setValue ( chunk_size );
    }

    this->yscale->setValue( model()->renderer->render_settings.y_scale );

    // keep buttons in sync
    ::Ui::MainWindow* ui = getItems();
    if (model()->renderer->render_settings.draw_piano)  hzmarker->setCheckedAction( ui->actionToggle_piano_grid );
    if (model()->renderer->render_settings.draw_hz)  hzmarker->setCheckedAction( ui->actionToggle_hz_grid );
    switch( model()->renderer->render_settings.color_mode )
    {
    case Heightmap::RenderSettings::ColorMode_Rainbow: color->setCheckedAction(ui->actionSet_rainbow_colors); break;
    case Heightmap::RenderSettings::ColorMode_Grayscale: color->setCheckedAction(ui->actionSet_grayscale); break;
    case Heightmap::RenderSettings::ColorMode_BlackGrayscale: color->setCheckedAction(ui->actionSet_blackgrayscale); break;
    case Heightmap::RenderSettings::ColorMode_FixedColor: color->setCheckedAction(ui->actionSet_colorscale); break;
    case Heightmap::RenderSettings::ColorMode_GreenRed: color->setCheckedAction(ui->actionSet_greenred_colors); break;
    case Heightmap::RenderSettings::ColorMode_GreenWhite: color->setCheckedAction(ui->actionSet_greenwhite_colors); break;
    case Heightmap::RenderSettings::ColorMode_Green: color->setCheckedAction(ui->actionSet_green_colors); break;
    }
    ui->actionSet_contour_plot->setChecked(model()->renderer->render_settings.draw_contour_plot);
    ui->actionToggleOrientation->setChecked(!model()->renderer->render_settings.left_handed_axes);


    // Only CWT benefits a lot from larger chunks, keep a lower min-framerate than otherwise
/*
//Use Signal::Processing namespace
    if (dynamic_cast<const Tfr::Cwt*>(model()->transform()))
        model()->project()->worker.min_fps( 1 );
    else
        model()->project()->worker.min_fps( 4 );

    // clear worker assumptions of target
    //model()->project()->worker.target(model()->renderSignalTarget);
*/
}


void RenderController::
        receiveSetYScale( qreal value )
{
    // Keep in sync with transformChanged()
    //float f = 2.f * value / yscale->maximum() - 1.f;
    model()->renderer->render_settings.y_scale = value; //exp( 8.f*f*f * (f>0?1:-1));

    stateChanged();

    yscale->setToolTip(QString("Intensity level %1").arg(model()->renderer->render_settings.y_scale));
}


void RenderController::
        receiveSetTimeFrequencyResolution( qreal value )
{
    bool isCwt = dynamic_cast<const Tfr::Cwt*>(currentTransform().get ());
    if (isCwt)
        write1(model()->transform_descs ())->getParam<Tfr::Cwt>().scales_per_octave ( value );
    else
        write1(model()->transform_descs ())->getParam<Tfr::StftDesc>().set_approximate_chunk_size( value );

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
    Tfr::TransformDesc::Ptr t = currentTransform();
    Tfr::TransformDesc::Ptr newuseroptions;

    if (!t)
        return;

    {
        Heightmap::TfrMapping::WritePtr tfr_map(model()->tfr_mapping ());

        // If the transform currently in use differs from the transform settings
        // that should be used, change the transform.
        Tfr::TransformDesc::Ptr useroptions = tfr_map->transform_desc();

        // If there is a transform but no tfr_map transform_desc it means that
        // there is a bug (or at least some continued refactoring todo). Update
        // tfr_map
        EXCEPTION_ASSERT (useroptions);

        newuseroptions = read1(model()->transform_descs ())->cloneType(typeid(*useroptions));
        EXCEPTION_ASSERT (newuseroptions);

        if (*newuseroptions != *useroptions)
            tfr_map->transform_desc( newuseroptions );
    }

    if (*t != *newuseroptions)
        setCurrentFilterTransform(newuseroptions);
}


void RenderController::
        setCurrentFilterTransform( Tfr::TransformDesc::Ptr t )
{
    {
        Tools::Support::TransformDescs::WritePtr td(model()->transform_descs ());
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
        setBlockFilter(Signal::DeprecatedOperation* blockfilter)
{
    Signal::OperationDesc::Ptr adapter(new Signal::OldOperationDescWrapper(Signal::pOperation (blockfilter)));
    setBlockFilter(adapter);
}


void RenderController::
        setBlockFilter(Heightmap::MergeChunkDesc::Ptr mcdp, Tfr::TransformDesc::Ptr transform_desc)
{
    // Wire it up to a FilterDesc
    Heightmap::ChunkBlockFilterDesc* cbfd;
    Tfr::FilterKernelDesc::Ptr kernel(cbfd
            = new Heightmap::ChunkBlockFilterDesc(model()->tfr_mapping ()));
    cbfd->setMergeChunkDesc( mcdp );
    Tfr::FilterDesc::Ptr desc( new Tfr::FilterDesc(transform_desc, kernel));
    setBlockFilter( desc );
}


void RenderController::
        setBlockFilter(Signal::OperationDesc::Ptr adapter)
{
    bool wasCwt = dynamic_cast<const Tfr::Cwt*>(currentTransform().get ());

    model()->set_filter (adapter);

    stateChanged();

    ::Ui::MainWindow* ui = getItems();

    ui->actionToggle_piano_grid->setVisible( true );
    hz_scale->setEnabled( true );

    bool isCwt = dynamic_cast<const Tfr::Cwt*>(currentTransform().get ());

    if (isCwt || wasCwt) {
        Tools::Support::TransformDescs::WritePtr td(model()->transform_descs ());
        Tfr::StftDesc& s = td->getParam<Tfr::StftDesc>();
        Tfr::Cwt& c = td->getParam<Tfr::Cwt>();
        float FS = headSampleRate();

        float wavelet_default_time_support = c.wavelet_default_time_support();
        float wavelet_fast_time_support = c.wavelet_time_support();
        c.wavelet_time_support(wavelet_default_time_support);

        if (isCwt && !wasCwt)
        {
            tf_resolution->setRange (2, 40);
            tf_resolution->setDecimals (1);
            c.scales_per_octave (s.chunk_size ()/(c.wavelet_time_support_samples(FS)/c.wavelet_time_support()/c.scales_per_octave ()));
            // transformChanged updates value accordingly
        }

        if (!isCwt && wasCwt)
        {
            tf_resolution->setRange (1<<5, 1<<20, Widgets::ValueSlider::Logaritmic);
            tf_resolution->setDecimals (0);
            s.set_approximate_chunk_size( c.wavelet_time_support_samples(FS)/c.wavelet_time_support() );
            // transformChanged updates value accordingly
        }

        c.wavelet_fast_time_support( wavelet_fast_time_support );
    }

    write1(model()->tfr_mapping ())->transform_desc( currentTransform()->copy() );

    view->emitTransformChanged();
    //return ps;
}


Tfr::TransformDesc::Ptr RenderController::
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
    return model()->project ()->extent ().sample_rate.get_value_or (1);
//    return model()->project()->head->head_source()->sample_rate();
}


float RenderController::
        currentTransformMinHz()
{
    Tfr::TransformDesc::Ptr t = currentTransform();
    EXCEPTION_ASSERT(t);
    return t->freqAxis(headSampleRate()).min_hz;
}


::Ui::MainWindow* RenderController::
        getItems()
{
    ::Ui::SaweMainWindow* main = dynamic_cast< ::Ui::SaweMainWindow*>(model()->project()->mainWindow());
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
    Heightmap::MergeChunkDesc::Ptr mcdp(new Heightmap::TfrMappings::CwtBlockFilterDesc(Heightmap::ComplexInfo_Amplitude_Non_Weighted));

    // Cwt needs fs
    float fs = headSampleRate ();
    write1(model()->transform_descs ())->getParam<Tfr::Cwt>().set_fs(fs);

    // Get a copy of the transform to use
    Tfr::TransformDesc::Ptr transform_desc = write1(model()->transform_descs ())->getParam<Tfr::Cwt>().copy();

    setBlockFilter(mcdp, transform_desc);
}


void RenderController::
        receiveSetTransform_Stft()
{
    // Setup the kernel that will take the transform data and create an image
    Heightmap::MergeChunkDesc::Ptr mcdp(new Heightmap::TfrMappings::StftBlockFilterDesc(model()->get_stft_block_filter_params ()));

    // Get a copy of the transform to use
    Tfr::TransformDesc::Ptr transform_desc = write1(model()->transform_descs ())->getParam<Tfr::StftDesc>().copy();

    setBlockFilter(mcdp, transform_desc);
}


void RenderController::
        receiveSetTransform_Cwt_phase()
{
    Heightmap::CwtToBlock* cwtblock = new Heightmap::CwtToBlock(model()->tfr_mapping (), model()->renderer.get());
    cwtblock->complex_info = Heightmap::ComplexInfo_Phase;

    setBlockFilter( cwtblock );
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
    EXCEPTION_ASSERTX(false, "Use Signal::Processing namespace");
/*
    Heightmap::CwtToBlock* cwtblock = new Heightmap::CwtToBlock(model()->tfr_map (), model()->renderer.get());
    cwtblock->complex_info = Heightmap::ComplexInfo_Amplitude_Non_Weighted;

    Signal::PostSink* ps = setBlockFilter( cwtblock );

    ps->filter( Signal::pOperation(new Filters::Ridge()));
*/
}


void RenderController::
        receiveSetTransform_Cwt_weight()
{
    Heightmap::CwtToBlock* cwtblock = new Heightmap::CwtToBlock(model()->tfr_mapping (), model()->renderer.get());
    cwtblock->complex_info = Heightmap::ComplexInfo_Amplitude_Weighted;

    setBlockFilter( cwtblock );
}


void RenderController::
        receiveSetTransform_Cepstrum()
{
    Heightmap::CepstrumToBlock* cepstrumblock = new Heightmap::CepstrumToBlock(model()->tfr_mapping ());

    setBlockFilter( cepstrumblock );
}


void RenderController::
        receiveSetTransform_DrawnWaveform()
{
    Heightmap::DrawnWaveformToBlock* drawnwaveformblock = new Heightmap::DrawnWaveformToBlock(model()->tfr_mapping ());

    setBlockFilter( drawnwaveformblock );

    ::Ui::MainWindow* ui = getItems();

    hz_scale->setEnabled( false );
    if (ui->actionToggle_piano_grid->isChecked())
        hzmarker->setChecked( false );
    ui->actionToggle_piano_grid->setVisible( false );

    // blockfilter sets the proper "frequency" axis
    linearScale->trigger();
}


void RenderController::
        receiveLinearScale()
{
    float fs = headSampleRate();

    Tfr::FreqAxis fa;
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
        receiveLogScale()
{
    float fs = headSampleRate();

    Tfr::FreqAxis fa;

    {
        Support::TransformDescs::WritePtr td(model()->transform_descs ());
        Tfr::Cwt& cwt = td->getParam<Tfr::Cwt>();
        fa.setLogarithmic(
                cwt.wanted_min_hz(),
                cwt.get_max_hz(fs) );

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

    Tfr::FreqAxis fa;
    fa.setQuefrencyNormalized( fs, write1(model()->transform_descs ())->getParam<Tfr::CepstrumDesc>().chunk_size() );

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
    model()->amplitude_axis( Heightmap::AmplitudeAxis_Linear );
    view->emitAxisChanged();
    stateChanged();
}


void RenderController::
        receiveLogAmplitude()
{
    model()->amplitude_axis( Heightmap::AmplitudeAxis_Logarithmic );
    view->emitAxisChanged();
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
    ::Ui::SaweMainWindow* main = dynamic_cast< ::Ui::SaweMainWindow*>(model()->project()->mainWindow());
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
//        connect(ui->actionTransform_Cwt_reassign, SIGNAL(triggered()), SLOT(receiveSetTransform_Cwt_reassign()));
//        connect(ui->actionTransform_Cwt_ridge, SIGNAL(triggered()), SLOT(receiveSetTransform_Cwt_ridge()));
//        connect(ui->actionTransform_Cwt_weight, SIGNAL(triggered()), SLOT(receiveSetTransform_Cwt_weight()));
        connect(ui->actionTransform_Cepstrum, SIGNAL(triggered()), SLOT(receiveSetTransform_Cepstrum()));
        connect(ui->actionTransform_Waveform, SIGNAL(triggered()), SLOT(receiveSetTransform_DrawnWaveform()));

        transform = new ComboBoxAction(toolbar_render);
        transform->setObjectName("ComboBoxActiontransform");
        transform->addActionItem( ui->actionTransform_Stft );
        transform->addActionItem( ui->actionTransform_Cwt );

        if (!Sawe::Configuration::feature("stable")) {
            transform->addActionItem( ui->actionTransform_Cepstrum );
            transform->addActionItem( ui->actionTransform_Waveform );
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
    {   linearScale = new QAction( toolbar_render );
        logScale = new QAction( toolbar_render );
        cepstraScale = new QAction( toolbar_render );

        linearScale->setText("Linear scale");
        logScale->setText("Logarithmic scale");
        cepstraScale->setText("Cepstra scale");

        // for serialization
        linearScale->setObjectName("linearScale");
        logScale->setObjectName("logScale");
        cepstraScale->setObjectName("cepstraScale");

        linearScale->setCheckable( true );
        logScale->setCheckable( true );
        cepstraScale->setCheckable( true );

        connect(linearScale, SIGNAL(triggered()), SLOT(receiveLinearScale()));
        connect(logScale, SIGNAL(triggered()), SLOT(receiveLogScale()));
        connect(cepstraScale, SIGNAL(triggered()), SLOT(receiveCepstraScale()));

        hz_scale = new ComboBoxAction();
        hz_scale->setObjectName("hz_scale");
        hz_scale->addActionItem( linearScale );
        hz_scale->addActionItem( logScale );
        if (!Sawe::Configuration::feature("stable")) {
            hz_scale->addActionItem( cepstraScale );
        }
        hz_scale->decheckable( false );
        toolbar_render->addWidget( hz_scale );

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
        amplitude_scale->addActionItem( fifthAmpltidue );
        amplitude_scale->decheckable( false );
        toolbar_render->addWidget( amplitude_scale );

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


    // QSlider * tf_resolution
    {   tf_resolution = new Widgets::ValueSlider( toolbar_render );
        tf_resolution->setObjectName ("tf_slider");
        tf_resolution->setRange (1<<5, 1<<20, Widgets::ValueSlider::Logaritmic);
        tf_resolution->setValue ( 4096 );
        tf_resolution->setToolTip ("Window size (time/frequency resolution) ");
        tf_resolution->setSliderSize ( 300 );
        tf_resolution->updateLineEditOnValueChanged (false); // RenderController does that instead
        toolbar_render->addWidget (tf_resolution);

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

    {
/*
//Use Signal::Processing namespace
        Signal::DeprecatedOperation* first_source = model()->project()->head->chain()->root_source().get();

        view->glwidget->setObjectName( QString("glwidget %1").arg(first_source->name().c_str()));
*/
        view->glwidget->setObjectName( QString("glwidget %1").arg((size_t)this));
    }

    view->glwidget->makeCurrent();

    view->graphicsview = new GraphicsView(view);
    view->graphicsview->setViewport(view->glwidget);
    view->glwidget->makeCurrent(); // setViewport makes the glwidget loose context, take it back
    view->tool_selector = view->graphicsview->toolSelector(0, model()->project()->commandInvoker());

    main->centralWidget()->layout()->setMargin(0);
    main->centralWidget()->layout()->addWidget(view->graphicsview);
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
    clearCaches();

/*
Use Signal::Processing namespace
    model()->renderSignalTarget.reset();
*/
}


void RenderController::
        clearCaches()
{
    // Stop worker from producing any more heightmaps by disconnecting
    // the collection callback from worker.
/*
Use Signal::Processing namespace
    if (model()->renderSignalTarget == model()->project()->worker.target())
        model()->project()->worker.target(Signal::pTarget());
*/

    // Assuming calling thread is the GUI thread.

    // Clear all cached blocks and release cuda memory befure destroying cuda
    // context
    foreach( const Heightmap::Collection::Ptr& collection, model()->collections() )
    {
        write1(collection)->reset();
    }


    // TODO clear stuff from FftImplementations somewhere not here
}


void RenderController::
        updateFreqAxis()
{
    switch(model()->display_scale().axis_scale)
    {
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
    unsigned  N = model()->project ()->extent().number_of_channels.get_value_or (0);
    if (read1(model()->tfr_mapping ())->channels() != (int)N)
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
        if (read1(model()->collections()[c])->isVisible() != o->isChecked())
        {
            write1(model()->collections()[c])->setVisible( o->isChecked() );
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
        if (model()->project()->mainWindow() == o)
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
    model()->project()->worker.min_fps( 20 );
*/
}


void RenderController::
        windowGotFocus()
{
/*
    // Use Signal::Processing namespace
    model()->project()->worker.min_fps( 4 );
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

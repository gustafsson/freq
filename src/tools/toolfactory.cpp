#include "toolfactory.h"

// Tools
#include "navigationcontroller.h"
#include "rendercontroller.h"
#include "renderview.h"
#include "timelinecontroller.h"
#include "timelineview.h"
#include "playbackcontroller.h"
#include "playbackview.h"
#include "recordmodel.h"
#include "recordcontroller.h"
#include "recordview.h"
#include "commentcontroller.h"
#include "tooltipcontroller.h"
#include "aboutdialog.h"
#include "playbackmarkerscontroller.h"
#include "transforminfoform.h"
#include "harmonicsinfoform.h"
#include "workercontroller.h"
#include "fantrackerview.h"
#include "fantrackermodel.h"
#include "settingscontroller.h"
#include "clickableimageview.h"
#include "dropnotifyform.h"
#include "sendfeedback.h"
#include "checkupdates.h"
#include "undoredo.h"
#include "commands/commandhistory.h"
#include "splashscreen.h"
#include "widgets/widgetoverlaycontroller.h"
#include "filtercontroller.h"
#include "printscreencontroller.h"

#include "selectioncontroller.h"
//#include "brushcontroller.h"
//#include "brushview.h"
//#include "matlabcontroller.h"
//#include "graphcontroller.h"
//#include "exportaudiodialog.h"
//#include "fantrackercontroller.h"
//#include "selectionviewinfo.h"
//#include "openandcomparecontroller.h"

// Sonic AWE
#include "sawe/project.h"
#include "sawe/configuration.h"
#include "ui/mainwindow.h"

// gpumisc
#include "TaskTimer.h"

// Qt
#include <QHBoxLayout>

namespace Tools
{

ToolFactory::
        ToolFactory(Sawe::Project* p)
:   ToolRepo(p),
    render_model( p ),
    selection_model( p ),
    playback_model( p )
{
    try
    {

    _render_view = new RenderView(&render_model);

    _objects.push_back( QPointer<QObject>( new RenderController(_render_view)));

    _timeline_view = new TimelineView(p, _render_view);
    _timeline_controller = new TimelineController(_timeline_view);


//Use Signal::Processing namespace
    //_selection_controller = new SelectionController(&selection_model, _render_view );


    //_navigation_controller = new NavigationController(_render_view);

/*
//Use Signal::Processing namespace
    playback_model.selection = &selection_model;
    _playback_view.reset( new PlaybackView(&playback_model, _render_view) );
    _playback_controller = new PlaybackController(p, _playback_view.data(), _render_view);
*/

#ifndef TARGET_hast
    // No brushes for Sjostridsskolan, the Swedish Naval Academy
/*
//Use Signal::Processing namespace
        _brush_model.reset( new BrushModel(p, &render_model) );
        _brush_view.reset( new BrushView(_brush_model.data() ));
        _brush_controller = new BrushController( _brush_view.data(), _render_view );
*/
#endif

//    if (RecordModel::canCreateRecordModel(p))
//        addRecording ();

/*
//Use Signal::Processing namespace
    _comment_controller = new CommentController( _render_view );
*/
    tool_controllers_.push_back( _comment_controller );

#if !defined(TARGET_sd) && !defined(TARGET_reader) && !defined(TARGET_hast)
/*
//Use Signal::Processing namespace
    // no matlab for sound design version, nor reader
    _matlab_controller = new MatlabController( p, _render_view );
*/
#endif

#ifndef TARGET_hast
/*
//Use Signal::Processing namespace
    _graph_controller = new GraphController( _render_view );
*/
#endif

    _tooltip_controller = new TooltipController(
            _render_view, dynamic_cast<CommentController*>(_comment_controller.data()) );
    tool_controllers_.push_back( _tooltip_controller );

#ifndef TARGET_hast
/*
//Use Signal::Processing namespace
    _fantracker_model.reset( new FanTrackerModel( &render_model ) );
    _fantracker_view.reset(new FanTrackerView( _fantracker_model.data() ,_render_view ));
    _fantracker_controller = new FanTrackerController(_fantracker_view.data(), _render_view );
*/
#endif

    _about_dialog = new AboutDialog( p );

/*
//Use Signal::Processing namespace
    _playbackmarkers_model.reset( new PlaybackMarkersModel() );
    _playbackmarkers_view.reset( new PlaybackMarkersView( _playbackmarkers_model.data(), p ));
    _playbackmarkers_controller = new PlaybackMarkersController(
            _playbackmarkers_view.data(), _render_view );
    playback_model.markers = _playbackmarkers_model.data();

    _transform_info_form = new TransformInfoForm(p, _render_view );

#ifndef TARGET_hast
    _export_audio_dialog = new ExportAudioDialog(p, &selection_model, _render_view);
#endif

    _harmonics_info_form = new HarmonicsInfoForm(
            p,
            dynamic_cast<TooltipController*>(_tooltip_controller.data()),
            _render_view
            );

    _selection_view_info = new SelectionViewInfo(p, &selection_model );
*/
//    _objects.push_back( QPointer<QObject>( new OpenAndCompareController( p ) ));

    _objects.push_back( QPointer<QObject>( new SettingsController( p )));

    // Promotion
    // _objects.push_back( QPointer<QObject>( new ClickableImageView( _render_view )));

#if !defined(USE_CUDA) && !defined(USE_OPENCL)
//    _objects.push_back( QPointer<QObject>( new DropNotifyForm( p->mainWindow()->centralWidget(), _render_view )));
#endif

    _objects.push_back( QPointer<QObject>( new SendFeedback( p->mainWindow() )));

    _objects.push_back( QPointer<QObject>( new CheckUpdates( p->mainWindow() )));

    _objects.push_back( QPointer<QObject>( new UndoRedo( p )));

    _objects.push_back( QPointer<QObject>( new Commands::CommandHistory( p->commandInvoker() )));

    if (Sawe::Configuration::feature("splash_screen"))
        _objects.push_back( QPointer<QObject>( new SplashScreen() ));

    if (Sawe::Configuration::feature("overlay_navigation"))
        _objects.push_back( QPointer<QObject>( new Widgets::WidgetOverlayController( _render_view ) ));

    _objects.push_back( QPointer<QObject>( new FilterController( p )));

    _objects.push_back( QPointer<QObject>( new PrintScreenController( p )));

    //
    // Insert new tools here, and delete things in the destructor in the
    // opposite order that they were created
    //


    _worker_view.reset( new WorkerView(p));
    _worker_controller.reset( new WorkerController( _worker_view.data(), _render_view, _timeline_view ) );

    } catch (const std::exception& x) {
        TaskInfo(boost::format("ToolFactory() caught exception\n%s") % boost::diagnostic_information(x));
        QMessageBox::critical(0, "Init error", QString("Init error. See logfile for details"));
    }
}


ToolFactory::
        ~ToolFactory()
{
    TaskInfo ti(__FUNCTION__);
    // Try to clear things in the opposite order that they were created

    // 'delete 0' is a valid operation and does nothing

    while(!_objects.empty())
    {
        delete _objects.back();
        _objects.pop_back();
    }

//    delete _selection_view_info;

    delete _harmonics_info_form;

//    delete _export_audio_dialog;

    delete _transform_info_form;

    delete _playbackmarkers_controller;

    delete _about_dialog;

//    delete _selection_controller;

    delete _navigation_controller;

    delete _playback_controller;

//    delete _brush_controller;

    delete _record_controller;

    delete _comment_controller;

//    delete _matlab_controller;

//    delete _graph_controller;

    delete _tooltip_controller;

//    delete _fantracker_controller;

    EXCEPTION_ASSERT( _timeline_controller );
	delete _timeline_controller;

    EXCEPTION_ASSERT( _timeline_view );
    delete _timeline_view;

    EXCEPTION_ASSERT( _render_view );
    delete _render_view;
}


void ToolFactory::
        addRecording (Adapters::Recorder* recorder)
{
    Sawe::Project*p = render_model.project ();

    _record_model.reset( RecordModel::createRecorder(
                p->processing_chain (),
                p->default_target (),
                recorder,
                p, _render_view ));

    _record_view.reset( new RecordView(_record_model.data() ));
    _record_controller = new RecordController( _record_view.data() );
}


ToolFactory::
        ToolFactory()
            :
            render_model( 0 ),
            selection_model( 0 ),
            playback_model( 0 )
{
    EXCEPTION_ASSERT( false );
}


} // namespace Tools

#ifndef TOOLFACTORY_H
#define TOOLFACTORY_H

namespace Sawe {
    class Project;
}

// TODO remove
#include "rendermodel.h"
#include "selectionmodel.h"
#include "playbackmodel.h"
#include "renderview.h"
#include "sawe/toolmodel.h"
#include "adapters/recorder.h"

#include <boost/serialization/version.hpp>

#include <typeinfo>
#include <QScopedPointer>

namespace Tools
{
    /**
      Find a better name...
      */
    class ToolFactory: public ToolRepo
    {
    public:
        ToolFactory(Sawe::Project* p);
        ~ToolFactory();

        void addRecording(Adapters::Recorder::ptr recorder);

        /**
         * TODO render_model should not be public. And a session could have more than one render model.
         * TODO add the concept of sessions.
         */
        Sawe::Project* project;
        RenderModel render_model;
        SelectionModel selection_model;
        PlaybackModel playback_model;
        class RecordModel* record_model() { return _record_model.data (); }

        //virtual ToolMainLoop* mainloop() { return render_view(); }
        virtual RenderView* render_view() { return _render_view; }


        template<typename T>
        T* getObject()
        {
            for (int i=0; i<_objects.size(); ++i)
                if (T* t = dynamic_cast<T*>(_objects.at(i).data()))
                    return t;
            return 0;
        }

    private:
        friend class boost::serialization::access;
        ToolFactory(); // required by serialization, should never be called
        template<class Archive> void serialize(Archive& ar, const unsigned int version) {
            TaskInfo ti("ToolFactory::serialize");
            ar
                    & BOOST_SERIALIZATION_NVP(render_model);
            if (version == 0)
            {
                unsigned playback_device = -1;
                ar
                    & boost::serialization::make_nvp("playback_model.playback_device", playback_device);
            }
            ar
                    & BOOST_SERIALIZATION_NVP(playback_model.selection_filename)
                    & BOOST_SERIALIZATION_BASE_OBJECT_NVP(ToolRepo);
        }

        // QPointer only to tools that are owned by ToolFactory but may be
        // removed by their parent QObject during destruction.
        // QScopedPointer objects are owned by tool factory.
        // Other objects are stored as raw pointers and are guaranteed to be
        // removed by destorying the main window. ToolFactory will also be
        // destroyed shortly after the destruction of the main window.
        //
        // The currently active tool is already deleted prior to ~ToolFactory
        // when the main window is closed.

        QPointer<RenderView> _render_view; // owned by centralwidget

        QPointer<class TimelineView> _timeline_view; // owned by timelinedock which is owned by mainwindow
        QPointer<class TimelineController> _timeline_controller; // owned by _timeline_view

        QPointer<class SelectionController> _selection_controller; // might be deleted by _render_view

        QPointer<class NavigationController> _navigation_controller; // might be deleted by _render_view

        QScopedPointer<class PlaybackView> _playback_view;
        QPointer<class PlaybackController> _playback_controller;

//        QScopedPointer<class BrushModel> _brush_model;
//        QScopedPointer<class BrushView> _brush_view;
//        QPointer<class BrushController> _brush_controller; // might be deleted by _render_view

        QScopedPointer<class RecordModel> _record_model;
        QScopedPointer<class RecordView> _record_view;
        QPointer<class RecordController> _record_controller; // might be deleted by _render_view

        QPointer<class ToolController> _comment_controller;

//        QPointer<class MatlabController> _matlab_controller;

//        QPointer<class GraphController> _graph_controller;

        QPointer<class ToolController> _tooltip_controller;

        QScopedPointer<class FanTrackerModel> _fantracker_model;
        QScopedPointer<class FanTrackerView> _fantracker_view;
//        QPointer<class FanTrackerController> _fantracker_controller;

        QPointer<class AboutDialog> _about_dialog;

        QScopedPointer<class PlaybackMarkersModel> _playbackmarkers_model;
        QScopedPointer<class PlaybackMarkersView> _playbackmarkers_view;
        QPointer<class PlaybackMarkersController> _playbackmarkers_controller;

        QPointer<class TransformInfoForm> _transform_info_form;

//        QPointer<class ExportAudioDialog> _export_audio_dialog;

        QPointer<class HarmonicsInfoForm> _harmonics_info_form;

//        QPointer<class SelectionViewInfo> _selection_view_info;

        QList<QPointer<QObject> > _objects;

        QScopedPointer<class WorkerView> _worker_view;
        QScopedPointer<class WorkerController> _worker_controller;
    };
} // namespace Tools

BOOST_CLASS_VERSION(Tools::ToolFactory, 1)

#endif // TOOLFACTORY_H

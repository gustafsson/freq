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
#include "toolmodel.h"

#include <typeinfo>
#include <QScopedPointer>
#include <QPointer>
#include <boost/serialization/set.hpp>

namespace Tools
{
    /**
      Find a better name...
      */
    class ToolFactory
    {
    public:
        ToolFactory(Sawe::Project* p);
        ~ToolFactory();

        RenderModel render_model;
        SelectionModel selection_model;
        PlaybackModel playback_model;

        template<class Archive> void save_tools(Archive& ar, const unsigned int version) {
            serialize_tools( ar, version );
/*			ar
                    & BOOST_SERIALIZATION_NVP(_brush_model.data())
                    & BOOST_SERIALIZATION_NVP(_record_model.data())*/
        }

        template<class Archive> void load_tools(Archive& ar, const unsigned int version) {
            serialize_tools( ar, version );
/*			class BrushModel* brush = 0;
			ar
                    & BOOST_SERIALIZATION_NVP(_brush_model.data())
                    & BOOST_SERIALIZATION_NVP(_record_model)
*/

            foreach( const boost::shared_ptr<ToolModel>& model, toolModels)
            {
                _comment_controller->createView( model.get(), _project, _render_view);
            }
        }


        std::set<boost::shared_ptr<ToolModel> > toolModels;

    private:
        template<class Archive> void serialize_tools(Archive& ar, const unsigned int /*version*/) {
            ar
                    & BOOST_SERIALIZATION_NVP(render_model._qx)
                    & BOOST_SERIALIZATION_NVP(render_model._qy)
                    & BOOST_SERIALIZATION_NVP(render_model._qz)
                    & BOOST_SERIALIZATION_NVP(render_model._px)
                    & BOOST_SERIALIZATION_NVP(render_model._py)
                    & BOOST_SERIALIZATION_NVP(render_model._pz)
                    & BOOST_SERIALIZATION_NVP(render_model._rx)
                    & BOOST_SERIALIZATION_NVP(render_model._ry)
                    & BOOST_SERIALIZATION_NVP(render_model._rz)
                    & BOOST_SERIALIZATION_NVP(render_model.xscale)
                    & BOOST_SERIALIZATION_NVP(playback_model.playback_device)
                    & BOOST_SERIALIZATION_NVP(playback_model.selection_filename)
                    & BOOST_SERIALIZATION_NVP(toolModels);
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
        QScopedPointer<class RenderController> _render_controller;

        QPointer<class TimelineView> _timeline_view; // owned by timelinedock which is owned by mainwindow
        QPointer<class TimelineController> _timeline_controller; // owned by _timeline_view

        QPointer<class SelectionController> _selection_controller; // might be deleted by _render_view

        QPointer<class NavigationController> _navigation_controller; // might be deleted by _render_view

        QScopedPointer<class PlaybackView> _playback_view;
        QPointer<class PlaybackController> _playback_controller;

        QScopedPointer<class BrushModel> _brush_model;
        QScopedPointer<class BrushView> _brush_view;
        QPointer<class BrushController> _brush_controller; // might be deleted by _render_view

        QScopedPointer<class RecordModel> _record_model;
        QScopedPointer<class RecordView> _record_view;
        QPointer<class RecordController> _record_controller; // might be deleted by _render_view

        QPointer<class ToolController> _comment_controller;

        QPointer<class MatlabController> _matlab_controller;

        QPointer<class GraphController> _graph_controller;

        QPointer<class TooltipController> _tooltip_controller;

        QPointer<class AboutDialog> _about_dialog;

        QScopedPointer<class PlaybackMarkersModel> _playbackmarkers_model;
        QScopedPointer<class PlaybackMarkersView> _playbackmarkers_view;
        QPointer<class PlaybackMarkersController> _playbackmarkers_controller;

        Sawe::Project* _project;
    };
} // namespace Tools

#endif // TOOLFACTORY_H

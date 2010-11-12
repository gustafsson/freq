#ifndef TOOLFACTORY_H
#define TOOLFACTORY_H

namespace Sawe {
    class Project;
}

// TODO remove
#include "rendermodel.h"
#include "selectionmodel.h"
#include "playbackmodel.h"

#include <typeinfo>
#include <QScopedPointer>
#include <QPointer>

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

    private:
        // QPointer only to tools that are owned by ToolFactory but may be
        // removed by their parent QObject during destruction.
        // QScopedPointer objects are owned by tool factory.
        // Other objects are stored as raw pointers and are guaranteed to be
        // removed by destorying the main window. ToolFactory will also be
        // destroyed shortly after the destruction of the main window.
        //
        // The currently active tool is already deleted prior to ~ToolFactory
        // when the main window is closed.

        class RenderView* _render_view; // owned by centralwidget
        QScopedPointer<class RenderController> _render_controller;

        class TimelineView* _timeline_view; // owned by timelinedock which is owned by mainwindow
        class TimelineController* _timeline_controller; // owned by _timeline_view

        QPointer<class SelectionController> _selection_controller; // might be deleted by _render_view

        QPointer<class NavigationController> _navigation_controller; // might be deleted by _render_view

        QScopedPointer<class PlaybackView> _playback_view;
        QPointer<class PlaybackController> _playback_controller;

        QScopedPointer<class BrushModel> _brush_model;
        QScopedPointer<class BrushView> _brush_view;
        QPointer<class BrushController> _brush_controller; // might be deleted by _render_view

        Sawe::Project* _project;
    };
} // namespace Tools

#endif // TOOLFACTORY_H

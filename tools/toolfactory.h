#ifndef TOOLFACTORY_H
#define TOOLFACTORY_H

namespace Sawe {
    class Project;
}

// TODO remove
#include "rendermodel.h"
#include "selectionmodel.h"

#include "renderview.h"
#include "selectionview.h"

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

        // TODO remove
        RenderModel render_model;
        SelectionModel selection_model;

        RenderView render_view;
        SelectionView selection_view;
        // PlaybackView playback_view;
        // DiskwriterView diskwriter_view;

        // map<string, QWidget*> SelectionView selection_widget;

    private:
        class Timeline* t;

        Sawe::Project* _project;
    };
} // namespace Tools

#endif // TOOLFACTORY_H

#ifndef TOOLFACTORY_H
#define TOOLFACTORY_H

#include "sawe/project.h"
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

        RenderModel render_model;
        SelectionModel selection_model;

        RenderView render_view;
        SelectionView selection_view;
        // PlaybackView playback_view;
        // DiskwriterView diskwriter_view;

    private:
        Sawe::Project* _project;
    };
} // namespace Tools

#endif // TOOLFACTORY_H

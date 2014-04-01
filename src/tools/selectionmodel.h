#ifndef SELECTIONMODEL_H
#define SELECTIONMODEL_H

#include "signal/operation.h"
#include <vector>

#include <QObject>

namespace Sawe {
    class Project;
}


namespace Tools
{
    class SelectionModel: public QObject
    {
        Q_OBJECT
    public:
        SelectionModel(Sawe::Project* p);
        virtual ~SelectionModel();

        enum SaveInside
        {
            SaveInside_FALSE,
            SaveInside_TRUE,
            SaveInside_UNCHANGED
        };

        Signal::OperationDesc::ptr current_selection_copy(SaveInside si = SaveInside_UNCHANGED);
        void               set_current_selection(Signal::OperationDesc::ptr o);
        void               try_set_current_selection(Signal::OperationDesc::ptr o);
        const Signal::OperationDesc::ptr& current_selection() { return current_selection_; }

        Sawe::Project* project() { return project_; }

        // TODO How should 'all_selections' be used?
        // Should SelectionModel only describe one selection?
        std::vector<Signal::OperationDesc::ptr> all_selections;

    signals:
        void selectionChanged();

    private:
        Sawe::Project* project_;
        Signal::OperationDesc::ptr current_selection_;

        static Signal::OperationDesc::ptr copy_selection(Signal::OperationDesc::ptr, SaveInside si = SaveInside_UNCHANGED);
    };
} // namespace Tools

#endif // SELECTIONMODEL_H

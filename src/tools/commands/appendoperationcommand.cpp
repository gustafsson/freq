#include "appendoperationcommand.h"

#include "sawe/project.h"
#include "tools/support/operation-composite.h"
#include "signal/operationcache.h"

#include <boost/foreach.hpp>

namespace Tools {
namespace Commands {

AppendOperationCommand::
        AppendOperationCommand(Sawe::Project*p, Signal::pOperation s)
            :
            s(s),
            p(p)
{
}


void AppendOperationCommand::
        execute()
{
    QMutexLocker l(&p->head->chain()->mutex);

    prevHead = p->head->head_source();

    Tools::SelectionModel& m = p->tools().selection_model;

    if (m.current_selection() && m.current_selection()!=s)
    {
        Signal::pOperation onselectionOnly(new Tools::Support::OperationOnSelection(
                p->head->head_source(),
                Signal::pOperation( new Signal::OperationCachedSub(
                    m.current_selection_copy( Tools::SelectionModel::SaveInside_TRUE ))),
                Signal::pOperation( new Signal::OperationCachedSub(
                    m.current_selection_copy( Tools::SelectionModel::SaveInside_FALSE ))),
                s
                ));

        s = onselectionOnly;
    }

    p->head->appendOperation( s );

    newHead = p->head->head_source();

    p->tools().render_model.renderSignalTarget->findHead( p->head->chain() )->head_source( p->head->head_source() );
    p->tools().playback_model.playbackTarget->findHead( p->head->chain() )->head_source( p->head->head_source() );

    p->setModified();
}


void AppendOperationCommand::
        undo()
{
    if (p->head->chain()->tip_source() == newHead)
        p->head->chain()->tip_source( prevHead );
    else
    {
        BOOST_FOREACH( Signal::Operation*o, newHead->outputs())
        {
            o->source( prevHead );
        }

        Signal::Intervals affected = Signal::Operation::affectedDiff(newHead, prevHead);
        prevHead->invalidate_samples(affected);
    }
}


std::string AppendOperationCommand::
        toString()
{
    return s->name();
}


} // namespace Commands
} // namespace Tools

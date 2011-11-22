#include "changeselectioncommand.h"
#include "tools/selectioncontroller.h"

namespace Tools {
namespace Commands {

ChangeSelectionCommand::ChangeSelectionCommand(Tools::SelectionController*p, Signal::pOperation s)
    :
    s(s),
    p(p)
{
}


void ChangeSelectionCommand::
        execute()
{
    prevSelection = p->model()->current_selection_copy();
    p->setCurrentSelectionCommand( s );
}


void ChangeSelectionCommand::
        undo()
{
    p->setCurrentSelectionCommand( prevSelection );
}


std::string ChangeSelectionCommand::
        toString()
{
    return s ? s->name() : "No selection";
}


} // namespace Commands
} // namespace Tools

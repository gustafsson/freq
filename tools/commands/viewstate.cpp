#include "viewstate.h"
#include "viewcommand.h"

namespace Tools {
namespace Commands {

ViewState::
        ViewState(CommandInvoker* state)
            :
    QObject(state)
{
    connect(state,SIGNAL(projectChanged(const Command*)), SLOT(emitViewChanged(const Command*)));
}


void ViewState::
        emitViewChanged(const Command *c)
{
    if (const ViewCommand * t = dynamic_cast<const ViewCommand*>(c))
        emit viewChanged(t);
}

} // namespace Commands
} // namespace Tools

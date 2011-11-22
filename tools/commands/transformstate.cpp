#include "transformstate.h"
#include "transformcommand.h"

namespace Tools {
namespace Commands {

TransformState::
        TransformState(ProjectState* state)
            :
    QObject(state)
{
    connect(state,SIGNAL(projectChanged(const Command*)), SLOT(emitTransformChanged(const Command*)));
}


void TransformState::
        emitTransformChanged(const Command *c)
{
    if (const TransformCommand * t = dynamic_cast<const TransformCommand*>(c))
        emit transformChanged(t);
}

} // namespace Commands
} // namespace Tools

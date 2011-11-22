#ifndef TRANSFORMSTATE_H
#define TRANSFORMSTATE_H

#include "projectstate.h"

namespace Tools {
namespace Commands {

class TransformCommand;

class TransformState : public QObject
{
    Q_OBJECT
public:
    explicit TransformState(ProjectState* state);

signals:
    void transformChanged(const TransformCommand*);

private slots:
    void emitTransformChanged(const Command*);

};

} // namespace Commands
} // namespace Tools

#endif // TRANSFORMSTATE_H

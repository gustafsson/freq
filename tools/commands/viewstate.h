#ifndef VIEWSTATE_H
#define VIEWSTATE_H

#include "projectstate.h"

namespace Tools {
namespace Commands {

class ViewCommand;

class ViewState : public QObject
{
    Q_OBJECT
public:
    explicit ViewState(ProjectState* state);

signals:
    void viewChanged(const ViewCommand*);

private slots:
    void emitViewChanged(const Command*);

};

} // namespace Commands
} // namespace Tools

#endif // VIEWSTATE_H

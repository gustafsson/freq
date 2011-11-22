#ifndef PROJECTSTATE_H
#define PROJECTSTATE_H

#include <QObject>
#include <boost/shared_ptr.hpp>

#include "commandlist.h"

namespace Sawe { class Project; }

namespace Tools {
namespace Commands {

class ProjectState: public QObject
{
    Q_OBJECT
public:
    ProjectState(Sawe::Project * project);
    virtual ~ProjectState();

    void executeCommand(CommandP cmd);
    void redo();
    void undo();

    const CommandList& commandList() const;

    Sawe::Project * project() const;
signals:
    void projectChanged(const Command*);

private:
    CommandList commandList_;
    Sawe::Project * project_;
};

} // namespace Commands
} // namespace Tools

#endif // PROJECTSTATE_H

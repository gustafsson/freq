#ifndef COMMANDINVOKER_H
#define COMMANDINVOKER_H

#include <QObject>
#include <boost/shared_ptr.hpp>

#include "commandlist.h"

namespace Sawe { class Project; }

namespace Tools {
namespace Commands {

class CommandInvoker: public QObject
{
    Q_OBJECT
public:
    CommandInvoker(Sawe::Project * project);
    virtual ~CommandInvoker();

    void invokeCommand(CommandP cmd);
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

#endif // COMMANDINVOKER_H

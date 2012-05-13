#ifndef UNDOREDO_H
#define UNDOREDO_H

#include <QObject>

namespace Sawe
{
    class Project;
}

namespace Tools {

class UndoRedo : public QObject
{
    Q_OBJECT
public:
    explicit UndoRedo(::Sawe::Project *project);

private slots:
    void updateNames();
    void undo();
    void redo();

private:
    ::Sawe::Project *project_;
};

} // namespace Tools

#endif // UNDOREDO_H

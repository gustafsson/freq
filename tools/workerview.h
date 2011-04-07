#ifndef WORKERVIEW_H
#define WORKERVIEW_H

#include <QObject>

namespace Sawe { class Project; }

namespace Tools {

class WorkerView : public QObject
{
    Q_OBJECT
public:
    WorkerView(Sawe::Project* project);

    Sawe::Project* project();

signals:

public slots:
    virtual void draw();

private:
    Sawe::Project* project_;
};

} // namespace Tools

#endif // WORKERVIEW_H

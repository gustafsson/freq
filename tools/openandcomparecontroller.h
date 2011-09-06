#ifndef OPENANDCOMPARECONTROLLER_H
#define OPENANDCOMPARECONTROLLER_H

#include "sawe/project.h"
#include <QObject>

namespace Tools {

class OpenAndCompareController: public QObject
{
    Q_OBJECT
public:
    OpenAndCompareController(Sawe::Project* project);
    virtual ~OpenAndCompareController();

private:
    Sawe::Project* _project;

    void setupGui();

private slots:
    void slotOpenAndCompare();
};

} // namespace Tools

#endif // OPENANDCOMPARECONTROLLER_H

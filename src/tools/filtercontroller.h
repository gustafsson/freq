#ifndef TOOLS_FILTERCONTROLLER_H
#define TOOLS_FILTERCONTROLLER_H

#include <QObject>

namespace Sawe { class Project; }

namespace Tools {

class FilterController: public QObject
{
    Q_OBJECT
public:
    FilterController(Sawe::Project* p);

private slots:
    void addAbsolute();
    void addEnvelope();

private:
    Sawe::Project* project_;
};

} // namespace Tools

#endif // TOOLS_FILTERCONTROLLER_H

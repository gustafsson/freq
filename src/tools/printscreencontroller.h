#ifndef TOOLS_PRINTSCREENCONTROLLER_H
#define TOOLS_PRINTSCREENCONTROLLER_H

#include <QObject>

namespace Sawe { class Project; }

namespace Tools {

class PrintScreenController : public QObject
{
    Q_OBJECT
public:
    explicit PrintScreenController(Sawe::Project* project);
    
public slots:
    void takePrintScreen();

private:
    Sawe::Project* project_;
};

} // namespace Tools

#endif // TOOLS_PRINTSCREENCONTROLLER_H

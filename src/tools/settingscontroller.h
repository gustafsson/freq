#ifndef SETTINGSCONTROLLER_H
#define SETTINGSCONTROLLER_H

#include <QObject>

class QAction;

namespace Sawe
{
    class Project;
}

namespace Tools {
class SettingsDialog;

class SettingsController: public QObject
{
    Q_OBJECT
public:
    SettingsController(Sawe::Project*project);
    ~SettingsController();

private slots:
    void showSettings();
    void dialogFinished(int);

private:
    QAction* showSettingsAction;

    Sawe::Project* project_;
    bool initialized_;
};

} // namespace Tools

#endif // SETTINGSCONTROLLER_H

#ifndef SETTINGSDIALOG_H
#define SETTINGSDIALOG_H

#include <QDialog>

class QAbstractButton;

namespace Sawe {
    class Project;
}

namespace Tools {

namespace Ui {
    class SettingsDialog;
}

class SettingsDialog : public QDialog
{
    Q_OBJECT

public:
    explicit SettingsDialog(Sawe::Project* project, QWidget *parent);
    ~SettingsDialog();

private slots:
    void inputDeviceChanged(int);
    void outputDeviceChanged(int);
    void selectMatlabPath();
    void selectOctavePath();
    void radioButtonMatlab(bool);
    void radioButtonOctave(bool);
    void octavePathChanged(QString text);
    void matlabPathChanged(QString text);
    void resolutionChanged(int);
    void abstractButtonClicked(QAbstractButton*);

private:
    Ui::SettingsDialog *ui;
    Sawe::Project* project;

    void setupGui();
    void updateResolutionSlider();
    void clearSettings();
};

} // namespace Tools

#endif // SETTINGSDIALOG_H

#include "settingsdialog.h"
#include "ui_settingsdialog.h"

#include "sawe/project.h"
#include "adapters/playback.h"
#include "adapters/microphonerecorder.h"

#include <QSettings>
#include <QFileDialog>

namespace Tools {

SettingsDialog::
        SettingsDialog(Sawe::Project* project, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::SettingsDialog),
    project(project)
{
    setAttribute(Qt::WA_DeleteOnClose);

    ui->setupUi(this);

    this->setWindowModality( Qt::WindowModal );

    setupGui();
}


SettingsDialog::~SettingsDialog()
{
    delete ui;
}


void SettingsDialog::
        setupGui()
{
    std::list<Adapters::Playback::DeviceInfo> devices =
            Adapters::Playback::get_devices();

    int default_input = -1;
    int default_output = -1;
    foreach (const Adapters::Playback::DeviceInfo& di, devices)
    {
        if (di.inputChannels)
            ui->comboBoxAudioIn->addItem((di.name + " " + di.name2).c_str(), di.index);
        if (di.outputChannels)
            ui->comboBoxAudioOut->addItem((di.name + " " + di.name2).c_str(), di.index);
        if (di.isDefaultIn)
            default_input = di.index;
        if (di.isDefaultOut)
            default_output = di.index;
    }

    int currentInput =          QSettings().value("inputdevice", -1).toInt();
    int currentOutput =         QSettings().value("outputdevice", -1).toInt();
    if (0 > currentInput)       currentInput = default_input;
    if (0 > currentOutput)      currentOutput = default_output;
    currentInput =              ui->comboBoxAudioIn->findData( currentInput );
    currentOutput =             ui->comboBoxAudioOut->findData( currentOutput );
    if (0 <= currentInput)      ui->comboBoxAudioIn->setCurrentIndex( currentInput );
    if (0 <= currentOutput)     ui->comboBoxAudioOut->setCurrentIndex( currentOutput );

    connect(ui->comboBoxAudioIn, SIGNAL(currentIndexChanged(int)), SLOT(inputDeviceChanged(int)));
    connect(ui->comboBoxAudioOut, SIGNAL(currentIndexChanged(int)), SLOT(outputDeviceChanged(int)));

    connect(ui->pushButtonMatlab, SIGNAL(clicked()), SLOT(selectMatlabPath()));
    connect(ui->pushButtonOctave, SIGNAL(clicked()), SLOT(selectOctavePath()));

    ui->lineEditOctave->setText( QSettings().value("octavepath").toString() );
    ui->lineEditMatlab->setText( QSettings().value("matlabpath").toString() );
    connect(ui->lineEditMatlab, SIGNAL(textChanged(QString)), SLOT(matlabPathChanged(QString)));
    connect(ui->lineEditOctave, SIGNAL(textChanged(QString)), SLOT(octavePathChanged(QString)));

    QString defaultscript = QSettings().value("defaultscript", "matlab").toString();
    if (defaultscript=="matlab")
        ui->radioButtonMatlab->setChecked(true);
    else
        ui->radioButtonOctave->setChecked(true);

    connect(ui->radioButtonMatlab, SIGNAL(toggled(bool)), SLOT(radioButtonMatlab(bool)));
    connect(ui->radioButtonOctave, SIGNAL(toggled(bool)), SLOT(radioButtonOctave(bool)));

    connect(ui->buttonBox, SIGNAL(accepted()), SLOT(accept()));
    connect(ui->buttonBox, SIGNAL(rejected()), SLOT(reject()));
}


void SettingsDialog::
        inputDeviceChanged(int i)
{
    int inputDevice = ui->comboBoxAudioIn->itemData( i ).toInt();
    QSettings().setValue("inputdevice", inputDevice);

    Adapters::MicrophoneRecorder* mr = dynamic_cast<Adapters::MicrophoneRecorder*>(project->head->head_source()->root());
    if (mr)
        mr->changeInputDevice( inputDevice );
}


void SettingsDialog::
        outputDeviceChanged(int i)
{
    QSettings().setValue("outputdevice", ui->comboBoxAudioOut->itemData( i ).toInt());
}


void SettingsDialog::
        selectMatlabPath()
{
    QFileDialog dialog(project->mainWindowWidget(),
                       "Select path to Matlab",
                        QSettings().value("matlabpath", "").toString());

    //dialog.setFilter( QDir::Executable | QDir::Files | QDir::AllDirs | QDir::AllEntries | QDir::NoDotAndDotDot | QDir::Readable);

    if (dialog.exec() == QDialog::Accepted)
    {
        QString dir = dialog.selectedFiles().first();
        ui->lineEditMatlab->setText( dir );
    }
}


void SettingsDialog::
        selectOctavePath()
{
    QFileDialog dialog(project->mainWindowWidget(),
                       "Select path to Octave",
                        QSettings().value("octavepath", "").toString());

    //dialog.setFilter( QDir::Executable | QDir::Files | QDir::AllDirs | QDir::AllEntries | QDir::NoDotAndDotDot | QDir::Readable);

    if (dialog.exec() == QDialog::Accepted)
    {
        QString dir = dialog.selectedFiles().first();
        ui->lineEditOctave->setText( dir );
    }
}


void SettingsDialog::
        radioButtonMatlab(bool checked)
{
    if (checked)
        QSettings().setValue("defaultscript", "matlab");
}


void SettingsDialog::
        radioButtonOctave(bool checked)
{
    if (checked)
        QSettings().setValue("defaultscript", "octave");
}


void SettingsDialog::
        octavePathChanged(QString text)
{
    QSettings().setValue("octavepath", text);
    ui->radioButtonOctave->setChecked(true);
}


void SettingsDialog::
        matlabPathChanged(QString text)
{
    QSettings().setValue("matlabpath", text);
    ui->radioButtonMatlab->setChecked(true);
}


} // namespace Tools

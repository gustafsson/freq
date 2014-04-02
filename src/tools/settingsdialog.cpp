#include "settingsdialog.h"
#include "ui_settingsdialog.h"

#include "sawe/project.h"
#include "sawe/application.h"
#include "adapters/playback.h"
#include "adapters/microphonerecorder.h"
#include "tools/recordmodel.h"

#include "tfr/cwt.h"
#include "heightmap/collection.h"

#include "ui/mainwindow.h"

#include <QSettings>
#include <QFileDialog>
#include <QMessageBox>

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

    ui->lineEditLogFiles->setText(Sawe::Application::log_directory());

    updateResolutionSlider();
    resolutionChanged(ui->horizontalSliderResolution->value());
    connect(ui->horizontalSliderResolution, SIGNAL(valueChanged(int)), SLOT(resolutionChanged(int)));

    connect(ui->buttonBox, SIGNAL(accepted()), SLOT(accept()));
    connect(ui->buttonBox, SIGNAL(rejected()), SLOT(reject()));
    connect(ui->buttonBox, SIGNAL(clicked(QAbstractButton*)), SLOT(abstractButtonClicked(QAbstractButton*)));

#if !defined(TARGET_sd) && !defined(TARGET_reader) && !defined(TARGET_hast)
    ui->radioButtonMatlab->setVisible(false);
    ui->radioButtonOctave->setVisible(false);
    ui->lineEditMatlab->setVisible(false);
    ui->lineEditOctave->setVisible(false);
    ui->pushButtonMatlab->setVisible(false);
    ui->pushButtonOctave->setVisible(false);
#endif
}


void SettingsDialog::
        inputDeviceChanged(int i)
{
    int inputDevice = ui->comboBoxAudioIn->itemData( i ).toInt();
    QSettings().setValue("inputdevice", inputDevice);

    Tools::RecordModel* record_model = project->tools().record_model();
    auto rec = record_model->recording.write ();
    Adapters::MicrophoneRecorder* mr = dynamic_cast<Adapters::MicrophoneRecorder*>(&*rec);
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


void SettingsDialog::
        resolutionChanged(int v)
{
    float p = 1 - (v-ui->horizontalSliderResolution->minimum())/(float)(ui->horizontalSliderResolution->maximum()-ui->horizontalSliderResolution->minimum());
    // keep in sync with updateResolutionSlider
    float resolution = 1 + p*5;
    int fraction = 1 << (int)(resolution*0.5);

    QSettings().setValue("resolution", resolution);


    float prevRedundancy = project->tools().render_view()->model->renderer->redundancy();
    project->tools().render_view()->model->renderer->redundancy(resolution);
    project->tools().render_view()->model->renderer->setFractionSize(fraction, fraction);

    bool isCwt = dynamic_cast<const Tfr::Cwt*>(project->tools().render_model.transform_desc().get ());
    bool subtexelAggregationChanged = isCwt && (prevRedundancy == 1.f) != (resolution == 1.f);

#ifndef CWT_SUBTEXEL_AGGREGATION
    // subtexel aggregation is way to slow
    subtexelAggregationChanged = false;
#endif

    if (subtexelAggregationChanged)
    {
        Tools::RenderModel* rendermodel = &project->tools ().render_model;
        Signal::Processing::TargetNeeds::ptr needs = rendermodel->target_marker()->target_needs();
        needs.write ()->deprecateCache(Signal::Intervals::Intervals_ALL);
        needs.write ()->updateNeeds(
                    Signal::Intervals(),
                    Signal::Interval::IntervalType_MIN,
                    Signal::Interval::IntervalType_MAX);
    }

    project->tools().render_view()->redraw();
}


void SettingsDialog::
        abstractButtonClicked(QAbstractButton* b)
{
    if (QDialogButtonBox::ResetRole == ui->buttonBox->buttonRole(b))
    {
        clearSettings();
    }
}


void SettingsDialog::
        updateResolutionSlider()
{
    float resolution = project->tools().render_view()->model->renderer->redundancy();
    if (!project->isSaweProject())
        resolution = QSettings().value("resolution", resolution).toFloat();

    // keep in sync with resolutionChanged
    float p = (resolution - 1)/5;
    int value = (1 - p)*(ui->horizontalSliderResolution->maximum()-ui->horizontalSliderResolution->minimum()) + ui->horizontalSliderResolution->minimum();
    ui->horizontalSliderResolution->setValue( value );
}


void SettingsDialog::
        clearSettings()
{
    if (QMessageBox::Yes == QMessageBox::question(this, "Sonic AWE", "Clear all user defined settings?", QMessageBox::Yes | QMessageBox::No, QMessageBox::No))
    {
        QMessageBox::information(this, "Sonic AWE", "Restart Sonic AWE to reload default settings");
        QSettings().setValue("reset on next startup", true);
    }
}

} // namespace Tools

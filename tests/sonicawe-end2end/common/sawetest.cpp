#include "sawetest.h"

#include "sawe/application.h"
#include "tools/dropnotifyform.h"

#include "ui/mainwindow.h"
#include "ui_mainwindow.h"
#include "TaskTimer.h"


SaweTestClass::
        SaweTestClass()
            : work_sections_( 0 )
{}


Sawe::pProject SaweTestClass::
        project()
{
    return Sawe::pProject(project_);
}


void SaweTestClass::
        project(Sawe::pProject p)
{
    if (!project_.expired())
    {
        Sawe::pProject oldp = project();
        if (oldp)
        {
            disconnect( oldp->tools().render_view(), SIGNAL(postPaint()), this, SLOT(postPaint()));
            disconnect( oldp->tools().render_view(), SIGNAL(finishedWorkSection()), this, SLOT(renderViewFinishedWorkSection()));
        }
    }

    project_ = p;

    connect( p->tools().render_view(), SIGNAL(postPaint()), this, SLOT(postPaint()));
    connect( p->tools().render_view(), SIGNAL(finishedWorkSection()), this, SLOT(renderViewFinishedWorkSection()));
}


void SaweTestClass::
        exec()
{
    TaskTimer ti("%s exec", vartype(*this).c_str());
    Sawe::Application::global_ptr()->exec();
}


void SaweTestClass::
        closeDropNotifications()
{
    foreach (QObject* o, project()->mainWindow()->centralWidget()->children())
    {
        if (Tools::DropNotifyForm* dnf = dynamic_cast<Tools::DropNotifyForm*>(o))
        {
            dnf->close();
        }
    }
}


void SaweTestClass::
        timeLineVisibility(bool visible)
{
    QAction* timelineAction = project()->mainWindow()->getItems()->actionToggleTimelineWindow;
    if (visible != timelineAction->isChecked())
        timelineAction->trigger();
}


void SaweTestClass::
        postPaint()
{
    TaskTimer tt("SaweTestClass::postPaint");
    disconnect( sender(), SIGNAL(postPaint()), this, SLOT(postPaint()));

    projectOpened();
}


void SaweTestClass::
        projectOpened()
{
    closeDropNotifications();
    timeLineVisibility(true);
}


void SaweTestClass::
        renderViewFinishedWorkSection()
{
    finishedWorkSection(work_sections_++);
}

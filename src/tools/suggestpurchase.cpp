#include "suggestpurchase.h"
#include "dropnotifyform.h"
#include "sawe/reader.h"
#include "sawe/application.h"

#include <QSettings>
#include <QDate>

namespace Tools {

SuggestPurchase::SuggestPurchase(QWidget *centralWidget) :
    QObject(centralWidget)
{
    // wait for reader to finish
    connect( Sawe::Application::global_ptr(), SIGNAL(licenseChecked()), SLOT(suggest()) );
}


void SuggestPurchase::
        suggest()
{
    if ("not"!=Sawe::Reader::reader_text().substr(0,3))
        return;

    QSettings settings;
    QDate last_purchase_reminder = settings.value ("last purchase reminder", QDate::currentDate ()).toDate ();
    int starts_until_purchase_reminder = settings.value ("starts until purchase reminder", 10).toInt ();

    bool remind = false;
    int days_since_reminder = last_purchase_reminder.daysTo (QDate::currentDate ());
    if (days_since_reminder > starts_until_purchase_reminder)
        remind = true;
    if (starts_until_purchase_reminder == 0)
        remind = true;

    if (remind)
    {
        srand(time(0));
        starts_until_purchase_reminder = 2 + (rand()%10);

        settings.setValue ("last purchase reminder", QDate::currentDate ());
        settings.setValue ("starts until purchase reminder", starts_until_purchase_reminder);

        dropnotify();
    }
    else
    {
        starts_until_purchase_reminder--;
        settings.setValue ("starts until purchase reminder", starts_until_purchase_reminder);
    }
}


void SuggestPurchase::
        dropnotify ()
{
    new DropNotifyForm(
            dynamic_cast<QWidget*>(parent()),
            0,
            "You are using an evaluation version of " + QApplication::instance ()->applicationName () + ". Please purchase a license to continue using the program.",
            "https://gumroad.com/l/sonicawe",
            "Purchase");
}

} // namespace Tools

#include "nonblockingmessagebox.h"

namespace Sawe {

void NonblockingMessageBox::
        show( QMessageBox::Icon icon, QString title, QString msg, QString details )
{
    QMessageBox* message = new QMessageBox(
            icon,
            title,
            msg);

    message->setDetailedText(details);
    message->setAttribute( Qt::WA_DeleteOnClose );
    message->show();
}


} // namespace Sawe

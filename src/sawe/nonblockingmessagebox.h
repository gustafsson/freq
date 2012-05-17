#ifndef NONBLOCKINGMESSAGEBOX_H
#define NONBLOCKINGMESSAGEBOX_H

#include <QMessageBox>

namespace Sawe {

class NonblockingMessageBox
{
public:
    static void show( QMessageBox::Icon icon, QString title, QString msg, QString details = "" );
};

} // namespace Sawe

#endif // NONBLOCKINGMESSAGEBOX_H

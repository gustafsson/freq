#include "commandedit.h"

#include <QKeyEvent>

namespace Tools {
namespace Support {

CommandEdit::
        CommandEdit(QWidget *parent)
            :
    QLineEdit(parent)
{
    connect(this,SIGNAL(returnPressed()), SLOT(returnPressedSlot()));
    command_index = 0;
    commands.resize(command_index+1);
}


void CommandEdit::
        keyPressEvent ( QKeyEvent * event )
{
    switch (event->key())
    {
    case Qt::Key_Up:
        if (command_index>0)
        {
            if (command_index==commands.size()-1)
                commands[command_index] = text();

            command_index--;
            setText( commands[command_index] );
        }
        break;

    case Qt::Key_Down:
        if (command_index+1<commands.size())
        {
            command_index++;
            setText( commands[command_index] );
        }
        break;

    default:
        QLineEdit::keyPressEvent(event);
    }
}


void CommandEdit::
        returnPressedSlot()
{
    command_index = commands.size()-1;
    commands[command_index] = text();
    command_index++;
    commands.resize(command_index+1);
}


} // namespace Support
} // namespace Tools

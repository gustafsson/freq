import QtQuick 2.0
import QtQuick.Controls 1.2

ApplicationWindow {
    //    statusBar: StatusBar {
    //            visible: false
    //            RowLayout { Label { text: heightmap.displayedTransformDetails } }
    //        }

    id: appwindow
    width: 320
    height: 480
    title: main.title + " - Freq"

    menuBar: MenuBar {
        Menu {
            title: "File"
            MenuItem {
                text: "New window"
                shortcut: 6 // QKeySequence::New
                onTriggered: {
                    var component = Qt.createComponent("app.qml")
                    var window    = component.createObject()
                    window.show()
                }
            }
            MenuItem {
                text: "Close " + main.title
                shortcut: 4 // QKeySequence::Close
                onTriggered: appwindow.close()
            }
        }
    }

    Main {
        id: main
        anchors.fill: parent;
    }
}

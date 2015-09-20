import QtQuick 2.2
import QtQuick.Layouts 1.1

Item {
    id: greeting

    RowLayout {
        anchors.fill: parent
        spacing: 0

        Item {
            Layout.fillWidth: true
        }

        ColumnLayout {
            Layout.alignment: Qt.AlignTop
            Layout.topMargin: greeting.height*0.1

            id: infoBox

            Text {
                Layout.maximumWidth: greeting.width*0.7
                Layout.margins: 8

                text: qsTr("This view shows the frequencies that your microphone is currently recording. Scroll around by dragging, rotate with two fingers held together and zoom by moving two fingers in different directions. Try to whistle a bit, or clap your hands")
                wrapMode: Text.Wrap
            }

            Text {
                Layout.alignment: Qt.AlignRight
                Layout.margins: 13

                text: "Got it"

                MouseArea {
                    anchors.fill: parent
                    anchors.margins: -10 // make it easier to hit
                    onClicked: {visible = false; textAnimation.start();}
                    z: 10
                }

                Rectangle {
                    color: Qt.rgba(0.675, 0.675, 0.975, 0.8)
                    anchors.margins: -6
                    anchors.fill: parent
                    z: -1
                }
            }
        }

        Item {
            Layout.fillWidth: true
        }

        Rectangle {
            Layout.maximumWidth: 0
            color: Qt.rgba(0.975, 0.975, 0.975, 0.8)
            Layout.margins: -8
            anchors.fill: infoBox
            z: -1
        }
    }

    SequentialAnimation on opacity {
        id: textAnimation
        running: false
        NumberAnimation { to: 0; duration: 500; easing.type: Easing.OutQuart }
        ScriptAction { script: greeting.visible=false; }
    }
}

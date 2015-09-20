import QtQuick 2.0
import QtQuick.Controls 1.2
import OpenGLUnderQML 1.0
import QtQuick.Layouts 1.1

Item {
    property Squircle heightmap
    property bool displayDetails: false

    Layout.minimumHeight: transformSettingsFlow.height

    MouseArea {
        Layout.maximumHeight: 0
        Layout.maximumWidth: 0
        anchors.fill: transformSettingsFlow

        hoverEnabled: true
        onPressed: triggerMe(mouse)

        function triggerMe(mouse) {
            displayDetails = true;
            opacity_animation.restart();
            mouse.accepted = true;
        }
    }

    Component.onCompleted: heightmap.touchNavigation.connect(touchNavigation)

    signal touchNavigation()
    onTouchNavigation: opacity_animation.restart()

    onOpacityChanged: {
        visible: opacity!=0.0;
    }

    SequentialAnimation on opacity {
        id: opacity_animation
        NumberAnimation { to: 1; duration: 100; easing.type: Easing.InQuad }
        PauseAnimation { duration: 5000 }
        NumberAnimation { to: 0; duration: 2000; easing.type: Easing.OutQuad }
        ScriptAction { script: displayDetails = false; }
    }

    Flow {
        id: transformSettingsFlow
        anchors.left: parent.left
        anchors.right: parent.right

        spacing: 10

        Text {
            wrapMode: Text.WordWrap
            text: heightmap.displayedTransformDetails
        }

        ComboBox {
            visible: displayDetails && !heightmap.isIOS
            width: 150
            currentIndex: 0
            model: ListModel {
                ListElement { text: "Spectrogram"; name: "stft" }
                ListElement { text: "Wavelet"; name: "wavelet" }
            }
            onCurrentIndexChanged: {
                if (heightmap)
                    heightmap.displayedTransform = model.get(currentIndex).name
            }
        }

        ComboBox {
            visible: displayDetails
            width: 150
            currentIndex: 1
            model: ListModel {
                ListElement { text: "Linear height"; name: "linear" }
                ListElement { text: "Logarithmic height"; name: "log" }
            }
            onCurrentIndexChanged: if (heightmap) heightmap.displayedHeight = model.get(currentIndex).name
        }

        ComboBox {
            visible: displayDetails
            width: 150
            currentIndex: 1
            model: ListModel {
                ListElement { text: "Linear frequency"; name: "linear" }
                ListElement { text: "Logarithmic frequency"; name: "log" }
            }

            onCurrentIndexChanged: if (heightmap) heightmap.freqAxis = model.get(currentIndex).name
        }

        CheckBox {
            visible: displayDetails && !heightmap.isIOS
            text: qsTr("Equalize colors")
            checked: false
            onCheckedChanged: { if (checked) settrue.start(); else setfalse.start(); }

            property real smoothValue: 0
            onSmoothValueChanged: heightmap.equalizeColors = smoothValue

            SequentialAnimation on smoothValue {
                running: false
                id: setfalse
                NumberAnimation { to: 0; duration: 1000; easing.type: Easing.InOutCubic }
            }

            SequentialAnimation on smoothValue {
                running: false
                id: settrue
                NumberAnimation { to: 1; duration: 1000; easing.type: Easing.InOutCubic }
            }
        }

        CheckBox {
            visible: displayDetails && !heightmap.isIOS

            text: "Freeze T/F ratio"
            checked: false
            onCheckedChanged: {
                optimalres.paused = checked;
                parent.displayDetails = checked;
                opacity_animation.stop();
            }
        }

//    Slider {
//        visible: displayDetails
//        Layout.fillWidth: true
//        onValueChanged: heightmap.y_min = value
//    }

//    Slider {
//        visible: displayDetails
//        Layout.fillWidth: true
//        onValueChanged: heightmap.y_max = value
//    }
    }
}

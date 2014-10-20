import QtQuick 2.0
import QtQuick.Controls 1.2
import OpenGLUnderQML 1.0
import QtQuick.Layouts 1.1

ColumnLayout {
    property Squircle heightmap

    Layout.fillWidth: true

    Component.onCompleted: {
        heightmap.touchNavigation.connect(touchNavigation)
    }

    signal touchNavigation()
    onTouchNavigation: opacity_animation.restart()

    Text {
        Layout.fillWidth: true

        CheckBox {
            id: transformCheckbox
            anchors.left: parent.left
            anchors.top: parent.top
            anchors.bottom: parent.bottom

            text: qsTr("")
            checked: false
            onCheckedChanged: {
                optimalres.paused = checked;
                opacity_animation.stop();
            }
        }

        wrapMode: Text.WordWrap
        text: "     " + heightmap.displayedTransformDetails
    }

    onOpacityChanged: {
        if (opacity==0.0) visible=false;
        if (opacity==1.0) visible=true;
    }

    SequentialAnimation on opacity {
        id: opacity_animation
        NumberAnimation { to: 1; duration: 100; easing.type: Easing.InQuad }
        PauseAnimation { duration: 5000 }
        NumberAnimation { to: 0; duration: 1000; easing.type: Easing.OutQuad }
    }

    Rectangle {
        color: Qt.rgba(1, 1, 1, 1)
        radius: 5
        border.width: 1
        border.color: "black"
        anchors.fill: parent
        anchors.margins: -5
        z: -1
        Layout.maximumHeight: 0
    }

    ComboBox {
        visible: transformCheckbox.checked
        Layout.fillWidth: true
        currentIndex: 1
        model: ListModel {
            ListElement { text: "Waveform"; name: "waveform" }
            ListElement { text: "Spectrogram"; name: "stft" }
            ListElement { text: "Wavelet"; name: "wavelet" }
        }
        onCurrentIndexChanged: if (heightmap) heightmap.displayedTransform = model.get(currentIndex).name
    }

    ComboBox {
        visible: transformCheckbox.checked
        Layout.fillWidth: true
        currentIndex: 1
        model: ListModel {
            ListElement { text: "Linear height"; name: "linear" }
            ListElement { text: "Logarithmic height"; name: "log" }
        }
        onCurrentIndexChanged: if (heightmap) heightmap.displayedHeight = model.get(currentIndex).name
    }

    CheckBox {
        visible: transformCheckbox.checked
        Layout.fillWidth: true
        text: qsTr("Equalize colors")
        checked: true
        onCheckedChanged: { if (checked) settrue.start(); else setfalse.start(); }

        property real smoothValue: 1
        onSmoothValueChanged: heightmap.equalizeColors = smoothValue

        SequentialAnimation on smoothValue {
            running: false
            id: setfalse
            NumberAnimation { to: 0; duration: 1000; easing.type: Easing.InQuad }
        }

        SequentialAnimation on smoothValue {
            running: false
            id: settrue
            NumberAnimation { to: 1; duration: 1000; easing.type: Easing.InQuad }
        }
    }

//    Slider {
//        visible: transformCheckbox.checked
//        Layout.fillWidth: true
//        onValueChanged: heightmap.y_min = value
//    }

//    Slider {
//        visible: transformCheckbox.checked
//        Layout.fillWidth: true
//        onValueChanged: heightmap.y_max = value
//    }
}

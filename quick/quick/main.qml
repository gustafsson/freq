import QtQuick 2.2
import QtQuick.Controls 1.2
import OpenGLUnderQML 1.0
import QtQuick.Layouts 1.1

ApplicationWindow {
    objectName: "root item "

    width: 320
    height: 480

    Chain {
        id: chain
        // common chain, the app only opens one source file at a time
        //
        // property: Chain::ptr chain
        // property: Target::ptr target, this is where modifications common for all targets is applied, such
        // as cropping

        // There might be multiple render targets (Squircle) but all showing the same point in time, open the
        // same file twice to shoe different points in time simultaneously

        // The filters selected in one squircle can be applied on another squircle
    }

    ColumnLayout {
        objectName: "row layout"
        anchors.fill: parent

        Heightmap {
            id: heightmap1
            objectName: "heightmap1"
            chain: chain
            timepos: heightmap2.timepos
            Layout.fillWidth: true
            Layout.fillHeight: true
            height: 5
        }
        Heightmap {
            id: heightmap2
            objectName: "heightmap2"
            chain: chain
            timepos: heightmap1.timepos
            displayedTransform: "waveform"
            Layout.fillWidth: true
            Layout.fillHeight: true
            height: 1
        }
    }
    /*RowLayout {
        spacing: 2
        anchors.fill: parent
        Rectangle { Layout.fillWidth: true; Layout.fillHeight: true; color: "red" }
        Rectangle { Layout.fillWidth: true; Layout.fillHeight: true; color: "green" }
        Rectangle { color: "blue"; width: 50; height: 20 }
    }*/

    DropArea {
        id: dropArea
        anchors.fill: parent

        onEntered: {
            drag.accept (Qt.CopyAction);
        }
        onDropped: {
            if (drop.hasUrls && 1===drop.urls.length)
            {
                openurl.openUrl(drop.urls[0]);
                drop.accept (Qt.CopyAction);
            }
        }
        onExited: {
        }

        OpenUrl {
            id: openurl
            chain: chain
            signal openUrl(url p)
        }
    }

    Text {
        visible: true
        id: text
        color: "black"
        wrapMode: Text.WordWrap
        text: "Scroll by dragging, rotate with two fingers together, zoom with two fingers in different directions. http://freq.consulting"
        anchors.right: parent.right
        anchors.left: parent.left
        anchors.bottom: parent.bottom
        anchors.margins: 20

        SequentialAnimation on opacity {
            running: true
            NumberAnimation { to: 1; duration: 15000; easing.type: Easing.InQuad }
            NumberAnimation { to: 0; duration: 5000; easing.type: Easing.OutQuad }
        }

        Rectangle {
            color: Qt.rgba(1, 1, 1, 1)
            radius: 10
            border.width: 1
            border.color: "black"
            anchors.fill: parent
            anchors.margins: -10
            z: -1

            SequentialAnimation on radius {
                running: false // super annoying
                NumberAnimation { to: 20; duration: 1000; easing.type: Easing.InQuad }
                NumberAnimation { to: 10; duration: 1000; easing.type: Easing.OutQuad }
                loops: Animation.Infinite
            }
        }
    }
}

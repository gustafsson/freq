import QtQuick 2.2
import QtQuick.Controls 1.2
import OpenGLUnderQML 1.0
import QtQuick.Layouts 1.1

Item {
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

    Item {
        id: sharedCamera
        property real timepos: 0
        property real timezoom: 1
    }

    ColumnLayout {
        objectName: "row layout"
        anchors.fill: parent
        spacing: 0

        Heightmap {
            id: heightmap1
            objectName: "heightmap1"
            chain: chain
            selection: selection
            timepos: sharedCamera.timepos
            timezoom: sharedCamera.timezoom
            Layout.fillWidth: true
            Layout.fillHeight: true
            height: 5

            onTouchNavigation: {
                sharedCamera.timepos = timepos;
                sharedCamera.timezoom = timezoom;
            }
        }

        LayoutSplitter {}

        Heightmap {
            id: heightmap2
            objectName: "heightmap2"
            chain: chain
            scalepos: 0.5
            xangle: 90.0
            yangle: 180.0
            timepos: sharedCamera.timepos
            timezoom: sharedCamera.timezoom
            displayedTransform: "waveform"
            Layout.fillWidth: true
            Layout.fillHeight: true
            height: 1

            onTouchNavigation: {
                // force orthogonal view of waveform
                scalepos = 0.5;
                xangle = 90.0;
                yangle = 180.0;

                sharedCamera.timepos = timepos;
                sharedCamera.timezoom = timezoom;
            }
        }
    }

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

    ColumnLayout {
        anchors.left: parent.left
        anchors.top : parent.top
        anchors.right : parent.right
        anchors.margins: 20

        spacing: 30
        opacity: 0.75

        Text {
            Layout.fillWidth: true

            id: text
            color: "black"
            wrapMode: Text.WordWrap
            text: "Scroll by dragging, rotate with two fingers together, zoom with two fingers in different directions. http://freq.consulting"

            SequentialAnimation on opacity {
                PauseAnimation { duration: 15000 }
                NumberAnimation { to: 0; duration: 5000; easing.type: Easing.InQuart }
                ScriptAction { script: text.visible=false; }
            }

            Rectangle {
                color: Qt.rgba(1, 1, 1, 1)
                radius: 5
                border.width: 1
                border.color: "black"
                anchors.fill: parent
                anchors.margins: -5
                z: -1

                SequentialAnimation on radius {
                    running: false // super annoying
                    NumberAnimation { to: 20; duration: 1000; easing.type: Easing.InQuad }
                    NumberAnimation { to: 10; duration: 1000; easing.type: Easing.OutQuad }
                    loops: Animation.Infinite
                }
            }
        }

        TransformSettings {
            Layout.fillWidth: true
            id: transformsettings
            heightmap: heightmap1
        }

        Rectangle {
            color: Qt.rgba(1, 1, 1, 1)
            radius: 5
            border.width: 1
            border.color: "black"
            anchors.fill: transformsettings
            anchors.margins: -5
            z: -1
            Layout.maximumHeight: 0
            opacity: transformsettings.opacity
            visible: transformsettings.visible
        }
    }

    Text {
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottom: parent.bottom

        wrapMode: Text.WordWrap
        text: "  " + chain.title + "  "

        Component.onCompleted: {
            heightmap1.touchNavigation.connect(touchNavigation)
        }

        signal touchNavigation()
        onTouchNavigation: opacity_animation.restart()

        SequentialAnimation on opacity {
            id: opacity_animation
            NumberAnimation { to: 1; duration: 100; easing.type: Easing.InQuad }
            PauseAnimation { duration: 5000 }
            NumberAnimation { to: 0; duration: 1000; easing.type: Easing.OutQuad }
        }

        Rectangle {
            color: Qt.rgba(1, 1, 1, 1)
            radius: 10
            anchors.fill: parent
            anchors.margins: -3
            z: -1
        }
    }

    Selection {
        id: selection
        filteredHeightmap: heightmap2
        renderOnHeightmap: heightmap1
    }

    OptimalTimeFrequencyResolution {
        id: optimalres
        squircle: heightmap1
        paused: false
    }
}

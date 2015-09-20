import QtQuick 2.2
import QtQuick.Controls 1.2
import OpenGLUnderQML 1.0
import QtQuick.Layouts 1.1
import QtGraphicalEffects 1.0

Item {
    objectName: "root item "

    width: 320
    height: 480

    property string title: chain.title

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

            onOpenFileInfo: {
                text.text = infoText;
                textAnimation.restart();
            }
        }
    }

    ColumnLayout {
        objectName: "row layout"
        anchors.fill: parent
        spacing: 0


        Heightmap {
            id: waveform
            objectName: "waveform"
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
            visibleTimeAxis: 0

            onTouchNavigation: {
                // force orthogonal view of waveform
                scalepos = 0.5;
                xangle = 90.0;
                yangle = 180.0;

                sharedCamera.timepos = timepos;
                sharedCamera.timezoom = timezoom;
            }
        }


        LayoutSplitter {
            prevSibbling: waveform
            nextSibbling: heightmap
        }


        Heightmap {
            id: heightmap
            objectName: "heightmap"
            chain: chain
            selection: selection
            timepos: sharedCamera.timepos
            timezoom: sharedCamera.timezoom
            Layout.fillWidth: true
            Layout.fillHeight: true
            height: 6

            onTouchNavigation: {
                sharedCamera.timepos = timepos;
                sharedCamera.timezoom = timezoom;
            }


            ColumnLayout {
                anchors.fill: heightmap
                anchors.margins: 20

                spacing: 15

                Item {
                    Layout.fillHeight : true
                }

                Rectangle {
                    color: Qt.rgba(0.975, 0.975, 0.975, 0.8)
                    anchors.margins: -8
                    anchors.fill: transformsettings
                    z: -1
                    Layout.maximumHeight: 0
                    opacity: transformsettings.opacity
                    visible: transformsettings.visible
                }

                TransformSettings {
                    Layout.fillWidth: true
                    id: transformsettings
                    heightmap: heightmap
                }

                /*MyComponent {
                    Layout.fillWidth: true
                }*/
            }
        }
    }


    Text {
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottom: parent.bottom

        wrapMode: Text.WordWrap
        text: "  " + chain.title + "  "

        Component.onCompleted: {
            heightmap.touchNavigation.connect(touchNavigation)
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
        filteredHeightmap: waveform.isIOS ? null : waveform
        renderOnHeightmap: heightmap
    }

    OptimalTimeFrequencyResolution {
        id: optimalres
        squircle: heightmap
        paused: false

        focus: true
        Keys.onPressed: {
            if (event.key === Qt.Key_Space ) {
                showAll();
                event.accepted = true;
            }
        }

        onUpdateSharedCamera: heightmap.touchNavigation()
    }

    Greeting {
        anchors.fill: parent
    }
}

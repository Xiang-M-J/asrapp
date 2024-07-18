// recorder_isolate.dart
import 'dart:isolate';
import 'package:flutter/services.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:path_provider/path_provider.dart';
import 'dart:io';
import 'dart:async';
import 'dart:isolate';
import 'package:flutter/material.dart';

// 消息类型
enum RecorderMessage {
  start,
  stop,
}

// 录音函数
void recorderManager(SendPort sendPort) async {
  // BackgroundIsolateBinaryMessenger.ensureInitialized();
  FlutterSoundRecorder recorder = FlutterSoundRecorder();
  Directory tempDir = await getTemporaryDirectory();
  String filePath = '${tempDir.path}/recording.aac';
  ReceivePort receivePort = ReceivePort();

  sendPort.send(receivePort.sendPort);

  await recorder.openRecorder();

  await for (var message in receivePort) {
    if (message == RecorderMessage.start) {
      await recorder.startRecorder(
        toFile: filePath,
        codec: Codec.aacADTS,
      );
      sendPort.send('Recording started');
    } else if (message == RecorderMessage.stop) {
      await recorder.stopRecorder();
      sendPort.send(filePath);
    }
  }
}



void main() {
  WidgetsFlutterBinding.ensureInitialized();

  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: RecorderScreen(),
    );
  }
}

class RecorderScreen extends StatefulWidget {
  @override
  _RecorderScreenState createState() => _RecorderScreenState();
}

class _RecorderScreenState extends State<RecorderScreen> {
  String _recordingFilePath = '';
  Isolate? _recorderIsolate;
  SendPort? _sendPort;
  bool _isRecording = false;

  Future<void> _initIsolate() async {
    ReceivePort receivePort = ReceivePort();
    _recorderIsolate = await Isolate.spawn(recorderManager, receivePort.sendPort);

    receivePort.listen((message) {
      if (message is SendPort) {
        _sendPort = message;
      } else if (message is String) {
        setState(() {
          if (message == 'Recording started') {
            _isRecording = true;
          } else {
            _recordingFilePath = message;
            _isRecording = false;
          }
        });
      }
    });
  }

  Future<void> _toggleRecording() async {
    if (_sendPort != null) {
      if (_isRecording) {
        _sendPort?.send(RecorderMessage.stop);
      } else {
        _sendPort?.send(RecorderMessage.start);
      }
    }
  }

  @override
  void initState() {
    super.initState();
    _initIsolate();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Recorder'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            ElevatedButton(
              onPressed: _toggleRecording,
              child: Text(_isRecording ? 'Stop Recording' : 'Start Recording'),
            ),
            Text('Recording saved at: $_recordingFilePath'),
          ],
        ),
      ),
    );
  }
}

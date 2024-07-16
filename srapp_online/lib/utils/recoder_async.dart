import 'dart:isolate';
import 'package:audio_session/audio_session.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:path_provider/path_provider.dart';
import 'dart:io';

import 'package:permission_handler/permission_handler.dart';

// 消息类型
enum RecorderMessage {
  start,
  stop,
}

class RecorderAsync{
  FlutterSoundRecorder? mRecorder = FlutterSoundRecorder();


}
// 录音函数
void recorderManager(SendPort sendPort) async {
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

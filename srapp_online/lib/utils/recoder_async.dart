import 'dart:async';
import 'dart:isolate';
import 'package:audio_session/audio_session.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:path_provider/path_provider.dart';
import 'dart:io';

import 'package:permission_handler/permission_handler.dart';
import 'package:srapp_online/utils/type_converter.dart';

// 消息类型
enum RecorderMessage {
  start,
  stop,
}

class RecorderAsync{
  FlutterSoundRecorder? mRecorder = FlutterSoundRecorder();
  StreamSubscription? recordingDataSubscription;
  ReceivePort receivePort = ReceivePort();


  void recordManager(SendPort sendPort) async {

    List<int> waveform = List<int>.empty(growable: true);

    await mRecorder?.openRecorder();

    sendPort.send(receivePort.sendPort);
    await for (var message in receivePort) {
      if (message == 1) {
        waveform.clear();
        var recordingDataController = StreamController<Food>();
        recordingDataSubscription =
            recordingDataController.stream.listen((buffer) {
              if (buffer is FoodData) {
                waveform.addAll(uint8LtoInt16List(buffer.data!));
              }
            });
        await mRecorder?.startRecorder(
          toStream: recordingDataController.sink,
          codec: Codec.pcm16,
          numChannels: 1,
          sampleRate: 16000,
          enableVoiceProcessing: false,
          bufferSize: 8192,
        );
        sendPort.send('start');
      } else if (message == 0) {
        await mRecorder?.stopRecorder();
        if (recordingDataSubscription != null) {
          await recordingDataSubscription?.cancel();
          recordingDataSubscription = null;
        }
        sendPort.send("stop");
      } else if (message == -1) {
        waveform.clear();

        mRecorder?.closeRecorder();
        if (recordingDataSubscription != null) {
          await recordingDataSubscription?.cancel();
          recordingDataSubscription = null;
        }
        mRecorder = null;
      }
    }
  }
}
// 录音函数

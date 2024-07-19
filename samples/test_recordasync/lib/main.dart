import 'dart:async';
import 'dart:isolate';
import 'dart:math';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:audio_session/audio_session.dart';
import 'package:permission_handler/permission_handler.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MyApp());
}

List<int> uint8LtoInt16List(Uint8List rawData) {
  List<int> intArray = List.empty(growable: true);
  ByteData byteData = ByteData.sublistView(rawData);
  for (var i = 0; i < byteData.lengthInBytes; i += 2) {
    intArray.add(byteData.getInt16(i, Endian.little).toInt());
  }
  return intArray;
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: const Text('Flutter Isolate Example'),
        ),
        body: const Center(
          child: IsolateDemo(),
        ),
      ),
    );
  }
}

class IsolateDemo extends StatefulWidget {
  const IsolateDemo({super.key});

  @override
  IsolateDemoState createState() => IsolateDemoState();
}


class IsolateDemoState extends State<IsolateDemo> {
  final String _output = "Press the button to start record";
  SendPort? _sendPort;
  Isolate? isolate;
  ReceivePort receivePort = ReceivePort();
  // RootIsolateToken rootIsolateToken = RootIsolateToken.instance!;
  FlutterSoundRecorder mRecorder = FlutterSoundRecorder();
  StreamSubscription? recordingDataSubscription;
  List<int> waveform = [];
  @override
  void initState() {
    super.initState();
    initializeAudio();
    _startIsolate();
  }



  Future<void> initializeAudio() async {

    var status = await Permission.microphone.request();
    if (status != PermissionStatus.granted) {
      throw RecordingPermissionException("Microphone permission not granted");
    }
    await mRecorder.openRecorder();
    final session = await AudioSession.instance;
    await session.configure(AudioSessionConfiguration(
      avAudioSessionCategory: AVAudioSessionCategory.playAndRecord,
      avAudioSessionCategoryOptions:
          AVAudioSessionCategoryOptions.allowBluetooth | AVAudioSessionCategoryOptions.defaultToSpeaker,
      avAudioSessionMode: AVAudioSessionMode.spokenAudio,
      avAudioSessionRouteSharingPolicy: AVAudioSessionRouteSharingPolicy.defaultPolicy,
      avAudioSessionSetActiveOptions: AVAudioSessionSetActiveOptions.none,
      androidAudioAttributes: const AndroidAudioAttributes(
        contentType: AndroidAudioContentType.speech,
        flags: AndroidAudioFlags.none,
        usage: AndroidAudioUsage.voiceCommunication,
      ),
      androidAudioFocusGainType: AndroidAudioFocusGainType.gain,
      androidWillPauseWhenDucked: true,
    ));
  }

  Future<void> start() async {
    var recordingDataController = StreamController<Food>();
    waveform.clear();
    recordingDataSubscription = recordingDataController.stream.listen((buffer) {
      print(buffer.runtimeType);
      if (buffer is FoodData) {
        waveform.addAll(uint8LtoInt16List(buffer.data!));
        if(waveform.length > 9600){
          _sendPort?.send(waveform.sublist(0, 9600)); // 发送信息
          waveform.removeRange(0, 9600);
        }
      }
    });
    await mRecorder.startRecorder(
      toStream: recordingDataController.sink,
      codec: Codec.pcm16,
      numChannels: 1,
      sampleRate: 16000,
      enableVoiceProcessing: false,
      bufferSize: 8192,
    );
  }

  Future<void> stop() async {
    if(waveform.length > 16 * 60){
      _sendPort?.send(waveform);
    }
    await mRecorder.closeRecorder();
    if (recordingDataSubscription != null) {
      print("取消流");
      await recordingDataSubscription!.cancel();
      recordingDataSubscription = null;
    }

    // if (isolate != null) {
    //   isolate!.kill(priority: Isolate.immediate);
    //   isolate = null;
    //   receivePort.close();
    // }
  }

  Future<void> _startIsolate() async {
    // final ReceivePort receivePort = ReceivePort();

    isolate = await Isolate.spawn(processAudioTask, {
      "port": receivePort.sendPort,
      // "token": rootIsolateToken,
      // "recorder": mRecorder,
      // "controller": recordingDataController,
      // "subscription": recordingDataSubscription
    });
    // 这里发送了主线程的 receivePort 的 sendPort，使得isolate 可以通过该 sendPort 向主线程返回信息

    receivePort.listen((m) {
      // 在这里监听 isolate 返回的信息
      if (m is SendPort) {
        //
        _sendPort = m; // 捕获 isolate 发出的 sendPort，后面可以通过 _sendPort 向 isolate 传入信息
      } else {
        print(m);
      }
    });
  }

  static void processAudioTask(Map inputs) async {
    SendPort sendPort = inputs["port"];
    ReceivePort receivePort = ReceivePort(); // 定义 isolate 的 receivePort
    sendPort.send(receivePort.sendPort); // 发送该 isolate 的 sendPort，外部可以通过该 sendPort 向 isolate 中传入数据

    receivePort.listen((param) async {
      if(param is List<int>){
        print(param.length);
        double sum = 0.1;
        for (var i = 0; i < param.length; i++){
          sum += param[i] * param[i];
          Future.delayed(const Duration(microseconds: 40));
        }
        sum = 10 * log(sum);
        sendPort.send(sum);
        print("length: ${param.length}  power: ${sum}");
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: <Widget>[
        Text(_output),
        const SizedBox(height: 20),
        ElevatedButton(
          onPressed: start,
          child: const Text('Start'),
        ),
        const SizedBox(
          height: 40,
        ),
        ElevatedButton(
          onPressed: stop,
          child: const Text('stop'),
        ),
      ],
    );
  }
}

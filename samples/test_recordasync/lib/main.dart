import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:flutter/services.dart';
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
  final String _output = "Press the button to start computation";
  SendPort? _sendPort;
  RootIsolateToken rootIsolateToken = RootIsolateToken.instance!;

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
    final session = await AudioSession.instance;
    await session.configure(AudioSessionConfiguration(
      avAudioSessionCategory: AVAudioSessionCategory.playAndRecord,
      avAudioSessionCategoryOptions:
          AVAudioSessionCategoryOptions.allowBluetooth |
              AVAudioSessionCategoryOptions.defaultToSpeaker,
      avAudioSessionMode: AVAudioSessionMode.spokenAudio,
      avAudioSessionRouteSharingPolicy:
          AVAudioSessionRouteSharingPolicy.defaultPolicy,
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

  Future<void> _startIsolate() async {
    final ReceivePort receivePort = ReceivePort();
    

    Isolate isolate = await Isolate.spawn(_isolateTask, {"port": receivePort.sendPort, "token": rootIsolateToken});  
    // 这里发送了主线程的 receivePort 的 sendPort，使得isolate 可以通过该 sendPort 向主线程返回信息

    receivePort.listen((m) {    // 在这里监听 isolate 返回的信息
      if (m is SendPort) {    // 
        _sendPort = m;        // 捕获 isolate 发出的 sendPort，后面可以通过 _sendPort 向 isolate 传入信息
      }else{
        print(m);
      }
    });
  }

  start(){
    _sendPort?.send("start");
  }
  stop(){
    _sendPort?.send("stop");
  }

  static void _isolateTask(Map inputs) async {
    SendPort sendPort = inputs["port"];
    BackgroundIsolateBinaryMessenger.ensureInitialized(inputs["token"]);
    // SendPort sendPort = inputs["port"]s
    ReceivePort receivePort = ReceivePort();  // 定义 isolate 的 receivePort
    sendPort.send(receivePort.sendPort);    // 发送该 isolate 的 sendPort，外部可以通过该 sendPort 向 isolate 中传入数据

    List<int> waveform = [];

    FlutterSoundRecorder mRecorder = FlutterSoundRecorder();
    await mRecorder.openRecorder();

    var recordingDataController = StreamController<Food>();
    StreamSubscription? recordingDataSubscription;

    receivePort.listen((param) async {    // 这里监听外部传入的信息
      if (param == "start") {
        recordingDataSubscription =
            recordingDataController.stream.listen((buffer) {
          if (buffer is FoodData) {
            waveform.addAll(uint8LtoInt16List(buffer.data!));
            sendPort.send(waveform.length);   // 发送信息
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
      } else if (param == "stop") {
        await mRecorder.stopRecorder();
        if (recordingDataSubscription != null) {
          await recordingDataSubscription!.cancel();
          recordingDataSubscription = null;
        }
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
        const SizedBox(height: 40,),
        ElevatedButton(
          onPressed: stop,
          child: const Text('stop'),
        ),
      ],
    );
  }
}

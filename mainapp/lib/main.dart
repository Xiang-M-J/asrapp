import 'dart:io';
import 'dart:async';
import 'dart:typed_data';
import 'package:audio_session/audio_session.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:mainapp/feature.dart';
import 'package:mainapp/speech_recognizer.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:uuid/uuid.dart';
import 'package:file_selector/file_selector.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '语音识别',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: const AsrScreen(),
    );
  }
}

class AsrScreen extends StatefulWidget {
  const AsrScreen({super.key});

  @override
  AsrScreenState createState() => AsrScreenState();
}

class AsrScreenState extends State<AsrScreen> {
  // String _recordFilePath;
  final TextEditingController _textController = TextEditingController();
  // 倒计时总时长
  double starty = 0.0;
  double offset = 0.0;
  bool isUp = false;
  String textShow = "按住说话";

  ///默认隐藏状态
  bool voiceState = true;
  Timer? _timer;
  int _count = 0;
  int maxRecordTime = 300;

  FlutterSoundRecorder? mRecorder = FlutterSoundRecorder();
  List<Uint8List> data = List<Uint8List>.empty(growable: true);
  String? audioPath;
  StreamSubscription? recordingDataSubscription;
  bool mRecorderIsInited = false;
  var uuid = const Uuid();
  List<String> tempNames = List.empty(growable: true);

  SpeechRecognizer? speechRecognizer;
  static const sampleRate = 16000;

  String? recognizeResult;

  @override
  void initState() {
    super.initState();
    // _player = FlutterSoundPlayer();
    // _recorder = FlutterSoundRecorder();
    initializeAudio();
    speechRecognizer = SpeechRecognizer(sampleRate);
    speechRecognizer?.initModel();
  }

  Future<void> initializeAudio() async {
    var status = await Permission.microphone.request();
    if (status != PermissionStatus.granted) {
      throw RecordingPermissionException("Microphone permission not granted");
    }
    await mRecorder!.openRecorder();
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
    setState(() {
      mRecorderIsInited = true;
    });
  }

  Future<void> deleteFiles() async {
    for (var tempName in tempNames) {
      var file = File(tempName);
      if (file.existsSync()) {
        file.deleteSync();
      }
    }
  }

  @override
  void dispose() {
    // _player.closePlayer();
    // _recorder.closeRecorder();
    _textController.dispose();
    _timer?.cancel();
    mRecorder!.closeRecorder();
    mRecorder = null;
    deleteFiles();
    super.dispose();
  }

  Future<IOSink> createFile() async {
    var tempDir = await getTemporaryDirectory();

    String tempName = uuid.v4();
    audioPath = '${tempDir.path}/$tempName.pcm';
    tempNames.add(audioPath!);
    var outputFile = File(audioPath!);
    if (outputFile.existsSync()) {
      await outputFile.delete();
    }
    return outputFile.openWrite();
  }

  ///开始语音录制的方法
  void start() async {
    assert(mRecorderIsInited);
    var sink = await createFile();
    data.clear();
    var recordingDataController = StreamController<Food>();
    recordingDataSubscription = recordingDataController.stream.listen((buffer) {
      if (buffer is FoodData) {
        // print(buffer.data!);
        data.add(buffer.data!);
        sink.add(buffer.data!);
      }
    });
    await mRecorder!.startRecorder(
      toStream: recordingDataController.sink,
      codec: Codec.pcm16,
      numChannels: 1,
      sampleRate: 16000,
      enableVoiceProcessing: false,
      bufferSize: 8192,
    );
    setState(() {});
  }

  List<int> toInt16(List<Uint8List> rawData){
    List<int> intArray = List.empty(growable: true);

    for (var i = 0; i<rawData.length; i++){
      ByteData byteData = ByteData.sublistView(rawData[i]);
      for(var i = 0; i < byteData.lengthInBytes; i += 2){
        intArray.add(byteData.getInt16(i, Endian.little).toInt());
      }
    }
    return intArray;
  }
  ///停止语音录制的方法
  Future<void> stop() async {
    await mRecorder!.stopRecorder();
    if (recordingDataSubscription != null) {
      await recordingDataSubscription!.cancel();
      recordingDataSubscription = null;
    }
    final intData = toInt16(data);
    recognizeResult = await speechRecognizer?.predict(processInput(intData), true);
    // processInput(data);
    _textController.text = recognizeResult ?? "未识别到结果";
    speechRecognizer?.reset();
    setState(() {
      
    });
  }

  processInput(List<int> raw){
    List<int> waveform = List.empty(growable: true);
    for (var i = 0; i < raw.length; i++) {
         waveform.add((raw[i]));
    }
    List<List<double>> fbank = extractFbank(waveform);
    return fbank;
  }

  showVoiceView() {
    setState(() {
      textShow = "松开结束";
      voiceState = false;
    });
    start();
  }

  hideVoiceView() {
    if (_timer!.isActive) {
      if (_count < 1) {
        print("too short");
      }
      _timer?.cancel();
    }

    setState(() {
      textShow = "按住说话";
      voiceState = true;
    });

    stop();
  }

  int2time(int count) {
    String time = "";
    int hours = (count ~/ 60);
    int minutes = count % 60;

    time =
        "${hours.toString().padLeft(2, '0')}:${minutes.toString().padLeft(2, '0')}";
    return time;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('语音识别'),
      ),
      body: Center(
        child: Column(
          // mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            const SizedBox(
              height: 40,
            ),
            Text("录音时间：${int2time(_count)}"),
            const SizedBox(
              height: 10,
            ),
            GestureDetector(
              onLongPressStart: (details) {
                _count = 0;
                showVoiceView();
                _timer =
                    Timer.periodic(const Duration(milliseconds: 1000), (t) {
                  setState(() {
                    _count++;
                  });
                  if (_count == maxRecordTime) {
                    print("最多录制300s");
                    hideVoiceView();
                  }
                });
              },
              onLongPressEnd: (details) {
                hideVoiceView();
              },
              child: Container(
                height: 100,
                width: 100,

                decoration: BoxDecoration(
                  color: Colors.blueAccent,
                  borderRadius: BorderRadius.circular(50),
                  border: Border.all(width: 5.0, color: Colors.grey.shade200),
                ),

                // margin: const EdgeInsets.fromLTRB(50, 0, 50, 20),
                child: Center(
                  child: Text(
                    textShow,
                    style: const TextStyle(color: Colors.white),
                  ),
                ),
              ),
            ),
            const SizedBox(
              height: 20,
            ),
            ElevatedButton(
                onPressed: () async {
                  // const XTypeGroup typeGroup = XTypeGroup(
                  //   label: 'images',
                  //   extensions: <String>['m4a', 'pcm', ".wav"],
                  // );
                  // final XFile? file = await openFile(acceptedTypeGroups: <XTypeGroup>[typeGroup]);
                  // // var binData = await rootBundle.load(file!.path);
                  // if (file == null) {
                  //   return;
                  // }
                  try{
                    final rawData = await rootBundle.load("assets/audio/asr_example.wav");
                    List<int> intData = List.empty(growable: true);
                    for(var i = 78; i < rawData.lengthInBytes; i+=2){
                      intData.add(rawData.getInt16(i, Endian.little).toInt());
                    }

                    recognizeResult = await speechRecognizer?.predict(processInput(intData), true);
                    // processInput(data);
                    _textController.text = recognizeResult ?? "未识别到结果";
                  }
                  catch(e){
                    print(e.toString());
                  }


                  speechRecognizer?.reset();

                },
                child: const Text("打开文件")),
            const SizedBox(
              height: 20,
            ),
            TextFormField(
              controller: _textController,
              decoration: const InputDecoration(
                hintText: '识别结果',
                border: OutlineInputBorder(),
              ),
              readOnly: true,
              
            ),
          ],
        ),
      ),
    );
  }
}

import 'dart:ffi';
import 'dart:io';
import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';
import 'package:audio_session/audio_session.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:mainapp/pages/show_toasts.dart';
import 'package:mainapp/utils/feature_utils.dart';
import 'package:mainapp/utils/fsmnvad_dector.dart';
import 'package:mainapp/utils/ort_env_utils.dart';
import 'package:mainapp/utils/sound_utils.dart';
import 'package:mainapp/utils/speech_recognizer.dart';
import 'package:mainapp/utils/voice_activity_detector.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:uuid/uuid.dart';
import 'package:file_selector/file_selector.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
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
  final TextEditingController resultController = TextEditingController();
  final TextEditingController statusController = TextEditingController();

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
  FlutterSoundPlayer? mPlayer = FlutterSoundPlayer();

  List<Uint8List> data = List<Uint8List>.empty(growable: true);
  String? audioPath;
  StreamSubscription? recordingDataSubscription;
  bool mRecorderIsInitialed = false;
  var uuid = const Uuid();
  List<String> tempNames = List.empty(growable: true);

  SpeechRecognizer? speechRecognizer;
  FsmnVaDetector? vaDetector;

  static const sampleRate = 16000;

  bool isSpeechRecognizeModelInitialed = true;
  bool isRecording = false;
  bool isRecognizing = false;

  String? recognizeResult;
  bool useVAD = false;

  @override
  void initState() {
    super.initState();
    initializeAudio();

    if (!isSpeechRecognizeModelInitialed) {
      statusController.text = "语音识别模型正在加载";
    }
    initOrtEnv();
    speechRecognizer = SpeechRecognizer();
    vaDetector = FsmnVaDetector();
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
      mRecorderIsInitialed = true;
    });
    mPlayer!.openPlayer();
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
    resultController.dispose();
    statusController.dispose();
    _timer?.cancel();
    mRecorder!.closeRecorder();
    mRecorder = null;
    mPlayer!.closePlayer();
    mPlayer = null;
    deleteFiles();
    vaDetector?.release();
    speechRecognizer?.release();
    releaseOrtEnv();
    super.dispose();
  }

  void playRemindSound() async{
    await mPlayer!.startPlayer(
      fromDataBuffer: remindSound,
      codec: Codec.pcm16WAV
    );
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
    assert(mRecorderIsInitialed);
    var sink = await createFile();
    data.clear();
    var recordingDataController = StreamController<Food>();
    recordingDataSubscription = recordingDataController.stream.listen((buffer) {
      if (buffer is FoodData) {
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

  List<int> toInt16(List<Uint8List> rawData) {
    List<int> intArray = List.empty(growable: true);

    for (var i = 0; i < rawData.length; i++) {
      ByteData byteData = ByteData.sublistView(rawData[i]);
      for (var i = 0; i < byteData.lengthInBytes; i += 2) {
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

    setState(() {
      isRecognizing = true;
      statusController.text = "正在识别...";
    });
    final intData = toInt16(data);
    recognizeResult = await speechRecognizer
        ?.predictWrapper(processInput(intData));
    resultController.text = recognizeResult ?? "未识别到结果";
    speechRecognizer?.reset();
    setState(() {
      isRecognizing = false;
      statusController.text = "识别完成";
    });
  }

  processInput(List<int> raw) {
    List<int> waveform = List.empty(growable: true);
    for (var i = 0; i < raw.length; i++) {
      waveform.add((raw[i]));
    }
    List<List<double>> fbank = extractFbank(waveform);
    return fbank;
  }

  Float32List int2Float(List<int> raw){
    List<double> rawD = raw.map((e) => e / 32768).toList();
    Float32List input = Float32List.fromList(rawD);
    return input;
  }

  showVoiceView() {
    setState(() {
      textShow = "松开结束";
      voiceState = false;
    });
    start();
  }

  hideVoiceView() {
    if (isRecognizing) return;
    if (_timer!.isActive) {
      if (_count < 1) {
        showToastWrapper("说话时间太短了");
      }
      _timer?.cancel();
    }

    setState(() {
      textShow = "按住说话";
      voiceState = true;
      isRecording = false;
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
            Padding(
              padding:
                  const EdgeInsets.only(left: 10, top: 0, right: 10, bottom: 0),
              child: TextField(
                controller: statusController,
                style: const TextStyle(color: Colors.grey),
                decoration: const InputDecoration(
                  border: OutlineInputBorder(
                    borderSide: BorderSide(color: Colors.grey, width: 2),
                  ),
                ),
                maxLines: 1,
                readOnly: true,
              ),
            ),
            const SizedBox(
              height: 40,
            ),
            Text.rich(
              TextSpan(
                children: [
                  WidgetSpan(
                      child: Icon(
                    Icons.mic,
                    color: isRecording ? Colors.green : Colors.grey,
                    size: 24,
                  )),
                  TextSpan(
                      text: " 录音时间：${int2time(_count)}",
                      style: const TextStyle(fontSize: 18)),
                ],
              ),
            ),
            const SizedBox(
              height: 40,
            ),
            GestureDetector(
              onLongPressStart: isSpeechRecognizeModelInitialed
                  ? (details) {
                      if (isRecognizing) {
                        showToastWrapper("正在识别，请稍等");
                        return;
                      }
                      _count = 0;
                      setState(() {
                        isRecording = true;
                        statusController.text = "正在录音...";
                      });
                      showVoiceView();
                      _timer = Timer.periodic(
                          const Duration(milliseconds: 1000), (t) {
                        setState(() {
                          _count++;
                        });
                        if (_count == maxRecordTime) {
                          print("最多录制300s");
                          hideVoiceView();
                        }
                      });
                    }
                  : null,
              onLongPressEnd: isSpeechRecognizeModelInitialed
                  ? (details) {
                      hideVoiceView();
                    }
                  : null,
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
                onPressed: isSpeechRecognizeModelInitialed
                    ? () async {
                        if (isRecognizing) {
                          showToastWrapper("正在识别,请稍等");
                          return;
                        }
                        setState(() {
                          isRecognizing = true;
                          statusController.text = "正在识别...";
                        });
                        // playRemindSound();
                        try {
                          final rawData = await rootBundle
                              .load("assets/audio/asr_example.wav");
                          List<int> intData = List.empty(growable: true);
                          for (var i = 78; i < rawData.lengthInBytes; i += 2) {
                            intData.add(
                                rawData.getInt16(i, Endian.little).toInt());
                          }
                          if (useVAD && vaDetector!.isInitialed){
                            setState(() {
                              statusController.text = "正在获取VAD结果";
                            });
                            // Float32List input = int2Float(intData);
                            // TODO 修改
                             List<List<int>>? segments = await vaDetector?.predict(intData);
                             print(segments);
                          }
                          recognizeResult = await speechRecognizer
                              ?.predictWrapper(processInput(intData));

                          resultController.text = recognizeResult ?? "未识别到结果";
                        } catch (e) {
                          print(e.toString());
                        }

                        speechRecognizer?.reset();
                        setState(() {
                          isRecognizing = false;
                          statusController.text = "识别完成";
                        });
                      }
                    : null,
                child: const Text("打开文件")),
            const SizedBox(
              height: 20,
            ),
            Padding(
              padding:
                  const EdgeInsets.only(left: 10, top: 0, right: 10, bottom: 0),
              child: TextFormField(
                controller: resultController,
                decoration: const InputDecoration(
                  hintText: '识别结果',
                  border: OutlineInputBorder(),
                ),
                readOnly: true,
                minLines: 4,
                maxLines: 10,
              ),
            ),
            
            Row(
              children: [
                const Padding(padding: EdgeInsets.only(left: 10, top: 10, right: 10,  bottom: 0),
                  child: Text("是否使用VAD", style: TextStyle(
                      fontSize: 16
                  ),),

                ),
                Padding(padding: const EdgeInsets.only(left: 0, top: 10, right: 0,  bottom: 0),
                  child: Switch(
                      value: useVAD,
                      activeColor: Colors.blue,
                      // inactiveThumbColor: Colors.black,
                      onChanged: (value) {
                        setState(() {
                          useVAD = value;
                        });
                        if (!vaDetector!.isInitialed){
                          setState(() {
                            statusController.text = "正在加载VAD模型";
                          });
                          vaDetector?.initModel("assets/models/fsmn_vad.onnx");
                          setState(() {
                            statusController.text = "已加载VAD模型";
                          });
                        }
                      })
                ),
              ],
            )
          ],
        ),
      ),
    );
  }
}

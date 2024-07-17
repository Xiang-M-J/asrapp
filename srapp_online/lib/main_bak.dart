import 'dart:async';
import 'dart:isolate';
import 'dart:math';
import 'dart:typed_data';
import 'package:audio_session/audio_session.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:srapp_online/pages/show_toasts.dart';
import 'package:srapp_online/utils/audio_loader.dart';
import 'package:srapp_online/utils/ernie_punctuation.dart';
import 'package:srapp_online/utils/fsmnvad_dector.dart';
import 'package:srapp_online/utils/ort_env_utils.dart';
import 'package:srapp_online/utils/sound_utils.dart';
import 'package:srapp_online/utils/speech_recognizer.dart';
import 'package:srapp_online/utils/type_converter.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:logger/logger.dart';

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

List<int> waveform = List<int>.empty(growable: true);

void recordManager(SendPort sendPort) async {

  StreamSubscription? recordingDataSubscription;
  FlutterSoundRecorder? mRecorder = FlutterSoundRecorder();

  await mRecorder.openRecorder();
  ReceivePort receivePort = ReceivePort();

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
        await recordingDataSubscription.cancel();
        recordingDataSubscription = null;
      }
      sendPort.send("stop");
    } else if (message == -1) {
      waveform.clear();

      mRecorder?.closeRecorder();
      if (recordingDataSubscription != null) {
        await recordingDataSubscription.cancel();
        recordingDataSubscription = null;
      }
      mRecorder = null;
    }
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
  String textShow = "开始录音";

  ///默认隐藏状态
  bool voiceState = true;
  Timer? _timer;
  int _count = 0;
  int maxRecordTime = 3600;

  int startIdx = 0;
  int step = 9600;
  int lastWaveformLength = 0;
  List<List<int>> voice = List<List<int>>.empty(growable: true);

  FlutterSoundPlayer? mPlayer = FlutterSoundPlayer();

  Isolate? _recorderIsolate;
  SendPort? _sendPort;

  // List<Uint8List> data = List<Uint8List>.empty(growable: true);
  List<int> waveform = List<int>.empty(growable: true);

  bool mRecorderIsInitialed = false;

  List<String> tempAudioPaths = List.empty(growable: true);

  SpeechRecognizer? speechRecognizer;
  FsmnVaDetector? vaDetector;
  ErniePunctuation? erniePunctuation;

  static const sampleRate = 16000;

  bool isSRModelInitialed = false;
  bool isRecording = false;
  bool isRecognizing = false;

  String? recognizeResult;
  bool useVAD = false;
  bool usePunc = false;

  Timer? vadTimer;
  Timer? srTimer;

  var logger = Logger(
    filter: null, // Use the default LogFilter (-> only log in debug mode)
    printer: PrettyPrinter(), // Use the PrettyPrinter to format and print log
    output: null, // Use the default LogOutput (-> send everything to console)
  );

  @override
  void initState() {
    super.initState();
    initializeAudio();
    _initIsolate();

    if (!isSRModelInitialed) {
      statusController.text = "语音识别模型正在加载";
    }
    initOrtEnv();
    speechRecognizer = SpeechRecognizer();
    vaDetector = FsmnVaDetector();
    erniePunctuation = ErniePunctuation();
    initModel();
  }

  void initModel() async {
    await speechRecognizer?.initModel();
    await vaDetector?.initModel("assets/models/fsmn_vad.onnx");
    setState(() {
      statusController.text = "语音识别模型已加载";
      isSRModelInitialed = true;
    });
  }

  Future<void> initializeAudio() async {
    var status = await Permission.microphone.request();
    if (status != PermissionStatus.granted) {
      throw RecordingPermissionException("Microphone permission not granted");
    }
    // await mRecorder!.openRecorder();
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

  @override
  void dispose() {
    resultController.dispose();
    statusController.dispose();
    _timer?.cancel();
    vadTimer?.cancel();
    srTimer?.cancel();
    mPlayer!.closePlayer();
    mPlayer = null;
    _sendPort?.send(-1);
    vaDetector?.release();
    erniePunctuation?.release();
    speechRecognizer?.release();
    releaseOrtEnv();
    super.dispose();
  }

  void playRemindSound() async {
    await mPlayer!
        .startPlayer(fromDataBuffer: remindSound, codec: Codec.pcm16WAV);
  }

  Future<void> _initIsolate() async {
    ReceivePort receivePort = ReceivePort();
    _recorderIsolate = await Isolate.spawn(recordManager, receivePort.sendPort);

    receivePort.listen((message) {
      if (message is SendPort) {
        _sendPort = message;
      } else if (message is String) {
        setState(() {
          if (message == 'start') {
            isRecording = true;
          } else {
            isRecording = false;
          }
        });
      }
    });
  }

  // 进行推理，包括 VAD 和语音识别
  inference(List<int> intData) async {
    try {
      List<List<int>>? segments;
      if (useVAD && vaDetector!.isInitialed) {
        setState(() {
          statusController.text = "正在获取VAD结果";
        });
        segments = vaDetector?.predict(intData);
      }
      if ((useVAD && segments == null) || (useVAD && segments!.isEmpty)) {
        showToastWrapper("似乎没有检测到语音");
        setState(() {
          statusController.text = "识别完成";
          recognizeResult = "";
        });
      } else {
        setState(() {
          statusController.text = "开始语音识别...";
        });
        Map<String, List<dynamic>>? result;
        if (useVAD && segments!.isNotEmpty) {
          result =
          await speechRecognizer?.predictWithVADAsync(intData, segments);
        } else {
          result = await speechRecognizer?.predictAsync(intData);
        }

        if (result != null) {
          if (usePunc && erniePunctuation!.isInitialed) {
            recognizeResult = await erniePunctuation
                ?.predictAsync(result["char"] as List<String>);
          } else if (useVAD && !usePunc) {
            recognizeResult = speechRecognizer?.puncByVAD(segments!, result);
          } else {
            recognizeResult = "${result["char"]?.join(" ")}。";
          }
        } else {
          recognizeResult = null;
        }
      }
      resultController.text = recognizeResult ?? "未识别到结果";
      speechRecognizer?.reset();
    } catch (e) {
      resultController.text = e.toString();
    }
  }

  vadSegment(List<int> intData) async {
    List<List<int>>? segments;
    segments = await vaDetector?.predictASync(intData);
    return segments;
  }

  speechRecognize(List<int> intData) async {
    Map<String, List<dynamic>>? result;
    result = await speechRecognizer?.predictAsync(intData);
    return result;
  }

  hideVoiceView() {
    if (isRecognizing) return;
    if (_timer!.isActive) {
      _timer?.cancel();
      logger.i("stop count timer");
    }
  }

  int2time(int count) {
    String time = "";
    int hours = (count ~/ 60);
    int minutes = count % 60;

    time =
    "${hours.toString().padLeft(2, '0')}:${minutes.toString().padLeft(2, '0')}";
    return time;
  }

  vadSegmentWrapper(int step) async {
    if (step < 800) return;

    print(step);

    List<List<int>>? segments = await vadSegment(waveform.sublist(0, step));

    logger.i(segments);
    if (segments == null) {
      waveform = waveform.sublist(step);
      // startIdx += step;
    } else {
      if (segments.isEmpty) {
        waveform = waveform.sublist(step);
      } else {
        int lastStartIdx = 0;
        for (var segment in segments) {
          if (segment[1] * 16 < step) {
            voice.add(waveform.sublist(segment[0] * 16, segment[1] * 16));
            lastStartIdx = segment[1] * 16;
          } else {
            lastStartIdx = segment[0] * 16;
          }
        }
        waveform = waveform.sublist(lastStartIdx);
      }
    }
    if (waveform.length > 64000) {
      voice.add(waveform.sublist(0, 64000));
      waveform = waveform.sublist(
        64000,
      );
      logger.i("输入过长，直接截断");
    }
    logger.i(waveform.length);
  }

  srWrapper() async {
    if (voice.isEmpty) {
      logger.i("无语音段");
      return;
    }
    Map? result = await speechRecognize(voice.removeAt(0));
    resultController.text += result?["char"].join(" ") + "，";
  }

  setTimer() {
    _timer = Timer.periodic(const Duration(milliseconds: 1000), (t) {
      setState(() {
        _count++;
      });
    });
    startIdx = 0;
    srTimer = Timer.periodic(const Duration(seconds: 1), (t) async {
      if (!vadTimer!.isActive && voice.isEmpty) {
        if (srTimer!.isActive) srTimer?.cancel();
        logger.i("srTimer is stop");
      }
      await srWrapper();
    });
    // for(var i = 0; ; i++){
    //   Future.delayed(Duration(seconds: 2));
    //   print(waveform.length);
    // }
    vadTimer = Timer.periodic(const Duration(seconds: 1), (t) {
      if (!isRecording && waveform.length < 800) {
        if (vadTimer!.isActive) vadTimer?.cancel();
        logger.i("stop vad timer");
      }
      step = min(waveform.length, 16000);
      vadSegmentWrapper(step);
    });
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
                    borderRadius: BorderRadius.all(Radius.circular(8.0)),
                    borderSide: BorderSide(color: Colors.grey, width: 8),
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
              onTap: isSRModelInitialed
                  ? () {
                if (_sendPort != null) {
                  if (isRecording) {
                    _sendPort?.send(0);
                  } else {
                    _sendPort?.send(1);
                  }
                }
                // if (isRecording) {
                //   setState(() {
                //     isRecording = false;
                //     statusController.text = "";
                //     textShow = "开始录音";
                //   });
                //   hideVoiceView();
                // } else {
                //   _count = 0;
                //   setState(() {
                //     isRecording = true;
                //     statusController.text = "正在录音...";
                //     textShow = "停止录音";
                //   });
                //   start();
                //   setTimer();
                // }
              }
                  : null,
              child: Container(
                height: 128,
                width: 128,

                decoration: BoxDecoration(
                  color: isRecording ? Colors.green : Colors.grey,
                  borderRadius: BorderRadius.circular(64),
                  border: Border.all(width: 8.0, color: Colors.blueGrey),
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
                onPressed: isSRModelInitialed
                    ? () async {
                  if (isRecognizing) {
                    showToastWrapper("正在识别,请稍等");
                    return;
                  }

                  setState(() {
                    isRecognizing = true;
                    statusController.text = "正在识别...";
                  });
                  // File newFile = File(newPath!);
                  try {
                    WavLoader wavLoader = WavLoader();
                    final rawData = await rootBundle
                        .load("assets/audio/asr_example.wav");
                    List<int> intData = List.empty(growable: true);
                    List<int> wavInfo =
                    await wavLoader.loadByteData(rawData);
                    for (var i = wavInfo[0];
                    i < wavInfo[0] + wavInfo[1];
                    i += 2) {
                      intData.add(
                          rawData.getInt16(i, Endian.little).toInt());
                    }

                    await inference(intData);
                  } catch (e) {
                    print(e.toString());
                    speechRecognizer?.reset();
                  }
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
            const SizedBox(
              height: 20,
            ),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                Row(
                  children: [
                    const Padding(
                      padding: EdgeInsets.only(
                          left: 10, top: 10, right: 10, bottom: 0),
                      child: Text(
                        "是否使用VAD",
                        style: TextStyle(fontSize: 16),
                      ),
                    ),
                    Padding(
                        padding: const EdgeInsets.only(
                            left: 0, top: 10, right: 0, bottom: 0),
                        child: Switch(
                            value: useVAD,
                            activeColor: Colors.blue,
                            // inactiveThumbColor: Colors.black,
                            onChanged: (value) {
                              if (!isSRModelInitialed) {
                                showToastWrapper("正在初始化语音识别模型");
                                return;
                              }
                              setState(() {
                                useVAD = value;
                              });
                              if (!vaDetector!.isInitialed) {
                                setState(() {
                                  statusController.text = "正在加载VAD模型";
                                });
                                vaDetector
                                    ?.initModel("assets/models/fsmn_vad.onnx");
                                setState(() {
                                  statusController.text = "已加载VAD模型";
                                });
                              }
                            })),
                  ],
                ),
                Row(
                  children: [
                    const Padding(
                      padding: EdgeInsets.only(
                          left: 10, top: 10, right: 10, bottom: 0),
                      child: Text(
                        "使用标点模型",
                        style: TextStyle(fontSize: 16),
                      ),
                    ),
                    Padding(
                        padding: const EdgeInsets.only(
                            left: 0, top: 10, right: 0, bottom: 0),
                        child: Switch(
                            value: usePunc,
                            activeColor: Colors.blue,
                            // inactiveThumbColor: Colors.black,
                            onChanged: (value) async {
                              setState(() {
                                usePunc = value;
                              });

                              if (!erniePunctuation!.isInitialed) {
                                setState(() {
                                  statusController.text = "正在加载Punc模型";
                                });
                                await erniePunctuation?.initVocab();
                                await erniePunctuation?.initModel();
                                setState(() {
                                  statusController.text = "已加载Punc模型";
                                });
                              }
                            })),
                  ],
                )
              ],
            )
          ],
        ),
      ),
    );
  }
}

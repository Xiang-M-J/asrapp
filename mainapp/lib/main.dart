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
import 'package:mainapp/utils/audio_loader.dart';
import 'package:mainapp/utils/ernie_punctuation.dart';
import 'package:mainapp/utils/fsmnvad_dector.dart';
import 'package:mainapp/utils/ort_env_utils.dart';
import 'package:mainapp/utils/sound_utils.dart';
import 'package:mainapp/utils/speech_recognizer.dart';
import 'package:mainapp/utils/type_converter.dart';
import 'package:permission_handler/permission_handler.dart';
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

class AsrScreenState extends State<AsrScreen>
    with SingleTickerProviderStateMixin {
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

  int green = 50;
  int blue = 100;

  Timer? _colorTimer;

  @override
  void initState() {
    super.initState();
    initializeAudio();

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
    for (var tempName in tempAudioPaths) {
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
    _colorTimer?.cancel();
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

  void playRemindSound() async {
    await mPlayer!
        .startPlayer(fromDataBuffer: remindSound, codec: Codec.pcm16WAV);
  }

  Future<IOSink> createFile() async {
    String audioPath = await getTemporaryAudioPath("pcm");
    tempAudioPaths.add(audioPath);
    var outputFile = File(audioPath);
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

  ///停止语音录制的方法
  Future<void> stop() async {
    await mRecorder!.stopRecorder();
    if (recordingDataSubscription != null) {
      await recordingDataSubscription!.cancel();
      recordingDataSubscription = null;
    }
    if (_count < 1) {
      showToastWrapper("说话时间太短了");
      setState(() {
        statusController.text = "";
      });
      return;
    }

    setState(() {
      isRecognizing = true;
      statusController.text = "正在识别...";
    });
    final intData = uin8toInt16(data);

    await inference(intData);

    setState(() {
      isRecognizing = false;
      statusController.text = "识别完成";
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
        segments = await vaDetector?.predict(intData);
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

          if(usePunc && erniePunctuation!.isInitialed){
            recognizeResult = await erniePunctuation?.predictAsync(result["char"] as List<String>);
          }else if (useVAD && !usePunc){
            recognizeResult = speechRecognizer?.puncByVAD(segments!, result);
          }
          else  {
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
      _timer?.cancel();
    }
    if (_colorTimer!.isActive) {
      _colorTimer?.cancel();
    }

    setState(() {
      blue = 100;
      green = 50;
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
                    borderRadius: const BorderRadius.all(Radius.circular(8.0)),
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
              onLongPressStart: isSRModelInitialed
                  ? (details) {
                      if (!isSRModelInitialed) {
                        showToastWrapper("正在初始化模型");
                        return;
                      }
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
                      _colorTimer =
                          Timer.periodic(const Duration(milliseconds: 25), (t) {
                        setState(() {
                          blue += 2;
                          green += 2;
                          if (blue >= 200) {
                            blue = 100;
                          }
                          if (green >= 200) {
                            green = 50;
                          }
                        });
                      });
                    }
                  : null,
              onLongPressEnd: isSRModelInitialed
                  ? (details) {
                      hideVoiceView();
                    }
                  : null,
              child: Container(
                height: 128,
                width: 128,

                decoration: BoxDecoration(
                  color: Colors.blueAccent,
                  borderRadius: BorderRadius.circular(64),
                  border: Border.all(
                      width: 12.0, color: Color.fromARGB(200, 20, green, blue)),
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
                        var status = await Permission.storage.request();
                        if (status != PermissionStatus.granted) {
                          showToastWrapper("未授予访问本地文件的权限");
                          return;
                        }
                        const XTypeGroup typeGroup = XTypeGroup(
                          label: 'images',
                          extensions: <String>['m4a', 'mp3', ".wav"],
                        );
                        final XFile? file = await openFile(
                            acceptedTypeGroups: <XTypeGroup>[typeGroup]);
                        if (file == null) {
                          return;
                        }
                        String? newPath;
                        if (!file.path.endsWith("wav")) {
                          newPath = await audioTransformUtils(file.path);
                          if (newPath == null) {
                            return;
                          } else {
                            tempAudioPaths.add(newPath);
                          }
                        }
                        else{
                          newPath = file.path;
                        }
                        setState(() {
                          isRecognizing = true;
                          statusController.text = "正在识别...";
                        });
                        // File newFile = File(newPath!);
                        try {
                          WavLoader wavLoader = WavLoader();
                          // final rawData = await rootBundle
                          //     .load("assets/audio/asr_example.wav");
                          // List<int> intData = List.empty(growable: true);
                          // List<int> wavInfo =
                          //     await wavLoader.loadByteData(rawData);
                          // for (var i = wavInfo[0];
                          //     i < wavInfo[0] + wavInfo[1];
                          //     i += 2) {
                          //   intData.add(
                          //       rawData.getInt16(i, Endian.little).toInt());
                          // }

                          ByteData rawData = await wavLoader.load(newPath);
                          List<int> intData = List.empty(growable: true);
                          for (var i = 0; i < rawData.lengthInBytes; i += 2) {
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

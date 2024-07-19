import 'dart:collection';
import 'dart:ffi';
import 'dart:io';
import 'dart:async';
import 'dart:isolate';
import 'dart:math';
import 'dart:typed_data';
import 'package:audio_session/audio_session.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:srapp_online/pages/scrollable_text_field.dart';
import 'package:srapp_online/pages/show_toasts.dart';
import 'package:srapp_online/utils/audio_loader.dart';
import 'package:srapp_online/utils/ernie_punctuation.dart';
import 'package:srapp_online/utils/fsmnvad_dector.dart';
import 'package:srapp_online/utils/ort_env_utils.dart';
import 'package:srapp_online/utils/paraformer_online.dart';
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

class AsrScreen extends StatefulWidget {
  const AsrScreen({super.key});
  @override
  AsrScreenState createState() => AsrScreenState();
}

class AsrScreenState extends State<AsrScreen> with SingleTickerProviderStateMixin {
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
  List<Map<int, List<int>>> voice = List<Map<int, List<int>>>.empty(growable: true);

  FlutterSoundRecorder? mRecorder = FlutterSoundRecorder();
  FlutterSoundPlayer? mPlayer = FlutterSoundPlayer();

  // List<Uint8List> data = List<Uint8List>.empty(growable: true);
  List<int> waveform = List<int>.empty(growable: true);

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

  // bool dirty = false;
  int counter = 0;
  int lastVoiceLength = 0;
  String lastStepResult = "";

  String? recognizeResult;
  bool useVAD = false;
  bool usePunc = false;

  Timer? vadTimer;
  Timer? srTimer;

  String keyword = "开始识别";

  List<int> cacheWave = List<int>.empty(growable: true);

  ParaformerOnline paraformerOnline = ParaformerOnline();

  var logger = Logger(
    filter: null, // Use the default LogFilter (-> only log in debug mode)
    printer: PrettyPrinter(), // Use the PrettyPrinter to format and print log
    output: null, // Use the default LogOutput (-> send everything to console)
  );

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
    await vaDetector?.initModel("assets/models/fsmn_vad.onnx");
    await paraformerOnline.initModel();
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
    setState(() {
      mRecorderIsInitialed = true;
    });
    mPlayer!.openPlayer();
  }

  @override
  void dispose() {
    resultController.dispose();
    statusController.dispose();
    _timer?.cancel();
    vadTimer?.cancel();
    srTimer?.cancel();
    mRecorder!.closeRecorder();
    mRecorder = null;
    mPlayer!.closePlayer();
    mPlayer = null;
    vaDetector?.release();
    erniePunctuation?.release();
    speechRecognizer?.release();
    paraformerOnline.release();
    releaseOrtEnv();
    super.dispose();
  }

  void playRemindSound() async {
    await mPlayer!.startPlayer(fromDataBuffer: remindSound, codec: Codec.pcm16WAV);
  }

  ///开始语音录制的方法
  void start() async {
    assert(mRecorderIsInitialed);
    // data.clear();
    waveform.clear();
    voice.clear();
    setState(() {
      resultController.text = "";
    });
    var recordingDataController = StreamController<Food>();
    recordingDataSubscription = recordingDataController.stream.listen((buffer) {
      if (buffer is FoodData) {
        waveform.addAll(uint8LtoInt16List(buffer.data!));
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

    // setState(() {
    //   isRecognizing = true;
    //   statusController.text = "正在识别...";
    // });
    // final intData = uint8LList2Int16List(data);

    // await inference(waveform);

    // setState(() {
    //   isRecognizing = false;
    //   statusController.text = "识别完成";
    // });
  }

  // 进行推理，包括 VAD 和语音识别
  inference(List<int> intData) async {
    try {
      List<List<int>>? segments;
      if (useVAD && vaDetector!.isInitialed) {
        setState(() {
          statusController.text = "正在获取VAD结果";
        });
        segments = await vaDetector?.predictASync(intData);
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
          result = await speechRecognizer?.predictWithVADAsync(intData, segments);
        } else {
          result = await speechRecognizer?.predictAsync(intData);
        }

        if (result != null) {
          if (usePunc && erniePunctuation!.isInitialed) {
            recognizeResult = await erniePunctuation?.predictAsync(result["char"] as List<String>);
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
    }
    stop();
  }

  int2time(int count) {
    String time = "";
    int hours = (count ~/ 60);
    int minutes = count % 60;

    time = "${hours.toString().padLeft(2, '0')}:${minutes.toString().padLeft(2, '0')}";
    return time;
  }

  setTimer() {
    _timer = Timer.periodic(const Duration(milliseconds: 1000), (t) {
      setState(() {
        _count++;
      });
    });
    startIdx = 0;
    int offset = 200;
    vadTimer = Timer.periodic(const Duration(milliseconds: 1000), (t) async {
      if (!isRecording && (waveform.length - startIdx < 800)) {
        if (vadTimer!.isActive) vadTimer?.cancel();
        logger.i("stop vad timer");
      }
      // logger.i("before step: $step,  wavlen ${waveform.length}");
      if (waveform.length > 640000) {
        step = 480000;
      } else {
        step = min(waveform.length - startIdx, 16000);
      }
      // logger.i("after step: $step,  wavlen ${waveform.length}");

      if (step < 800) return;
      List<int> seg = waveform.sublist(startIdx, startIdx + step);
      waveform = waveform.sublist(startIdx + step);
      List<List<int>>? segments = await vadSegment(seg);

      // print(segments);
      if (segments == null) {
        // waveform = waveform.sublist(startIdx + step);
        // startIdx += step;
        // startIdx += step;
      } else {
        if (segments.isEmpty) {
          // startIdx += step;
          // waveform = waveform.sublist(startIdx + step);
          // startIdx += step;
        } else {
          // int lastStartIdx = 0;
          // for (var segment in segments){
          //   if (segment[1] * 16 < startIdx + step){
          //     voice.add(waveform.sublist(startIdx + segment[0] * 16, startIdx + segment[1]*16));
          //     print("${startIdx + segment[0] * 16}, ${startIdx + segment[1] * 16}");
          //     lastStartIdx = segment[1] * 16;
          //   }else{
          //     lastStartIdx = segment[0] * 16;
          //   }
          // }
          // // waveform = waveform.sublist(lastStartIdx);
          // startIdx += lastStartIdx;
          for (var segment in segments) {
            int segb = segment[0] * 16;
            int sege = segment[1] * 16;
            print(
                "分段模型：startIdx 为 $startIdx, 识别波形长度为 ${seg.length}, 原始分段为 [${segment[0]}, ${segment[1]}] 识别段为 [$segb, $sege]");
            if (segb > startIdx + offset && sege < startIdx + step - offset) {
              voice.add({0: seg.sublist(segb, sege)});
              print(0);
            } else if (segb > startIdx + offset && sege >= startIdx + step - offset) {
              voice.add({1: seg.sublist(segb)});
              print(1);
            } else if (segb <= startIdx + offset && sege < startIdx + step - offset) {
              voice.add({2: seg.sublist(0, sege)});
              print(2);
            } else {
              voice.add({3: seg});
              print(3);
            }
          }
        }
      }
      // if(waveform.length > 64000){
      //   voice.add(waveform.sublist(0, 64000));
      //   waveform = waveform.sublist(64000, );
      //   logger.i("输入过长，直接截断");
      // }

      // lastWaveformLength = waveform.length;
      // logger.i(waveform.length);
    });

    srTimer = Timer.periodic(const Duration(milliseconds: 800), (t) async {
      if (!vadTimer!.isActive && voice.isEmpty) {
        if (srTimer!.isActive) srTimer?.cancel();
        logger.i("srTimer is stop");
      }
      if (voice.isEmpty) {
        return;
      }
      if (lastVoiceLength == voice.length) {
        counter++;
      }
      List<int>? cacheVoice;
      if (counter >= 3) {
        counter = 0;
        cacheVoice = concatVoice(voice);
        voice.clear();
        if (cacheVoice == null) {
          lastVoiceLength = 0;
          return;
        }
        Map? result = await speechRecognize(cacheVoice);
        lastStepResult += result?["char"].join(" ") + "，";
        resultController.text = lastStepResult;
      } else {
        cacheVoice = concatVoice(voice);
        if (cacheVoice == null) {
          voice.clear();
          lastVoiceLength = 0;
          return;
        }
        Map? result = await speechRecognize(cacheVoice);
        resultController.text = lastStepResult + result?["char"].join(" ");
      }

      lastVoiceLength = voice.length;
      // if (vadTimer!.isActive) {
      //   cacheVoice = voiceMachine(voice);
      // } else {
      //   cacheVoice = concatVoice(voice);
      // }
      // print(voice.length);
      // if (cacheVoice == null) return;
      // Map? result = await speechRecognize(cacheVoice);
      // resultController.text += result?["char"].join(" ") + "，";

      // for(var v in voice){
      //   if (v.keys.first == 0){
      //     Map? result = await speechRecognize();
      //   }
      // }
      //
      // resultController.text += result?["char"].join(" ") + "，";
    });
  }

  List<int>? concatVoice(List<Map<int, List<int>>> voice) {
    List<int> cacheVoice = List<int>.empty(growable: true);
    for (var v in voice) {
      cacheVoice.addAll(v.values.first);
    }
    // voice.clear();
    if (cacheVoice.length < 800) {
      return null;
    } else {
      return cacheVoice;
    }
  }

  List<int>? voiceMachine(List<Map<int, List<int>>> voice) {
    List<int> cacheVoice = List<int>.empty(growable: true);
    int lastLabel = -1;
    bool findEnd = false;
    List<int> waitToRemove = List.empty(growable: true);
    for (var i = 0; i < voice.length; i++) {
      Map<int, List<int>> v = voice[i];
      int label = v.keys.first;

      if (lastLabel == -1) {
        waitToRemove.add(i);
        cacheVoice.addAll(v[label]!);
        lastLabel = label;
        continue;
      }
      if (label == 0 || label == 1) {
        findEnd = true;
        break;
      } else if (label == 2) {
        if (lastLabel == 0 || lastLabel == 2) {
          findEnd = true;
          break;
        } else if (lastLabel == 1 || lastLabel == 3) {
          cacheVoice.addAll(v[label]!);
          waitToRemove.add(i);
        }
      } else {
        if (lastLabel == 0 || lastLabel == 2) {
          findEnd = true;
          break;
        } else {
          cacheVoice.addAll(v[label]!);
          waitToRemove.add(i);
        }
      }
      lastLabel = label;
    }
    if (findEnd) {
      for (var i in waitToRemove) {
        voice.removeAt(i);
      }
      if (cacheVoice.length < 1200) {
        return null;
      }
      return cacheVoice;
    } else {
      print("未发现整段语音");
      return null;
    }
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
              padding: const EdgeInsets.only(left: 10, top: 0, right: 10, bottom: 0),
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
                  TextSpan(text: " 录音时间：${int2time(_count)}", style: const TextStyle(fontSize: 18)),
                ],
              ),
            ),
            const SizedBox(
              height: 40,
            ),
            GestureDetector(
              onTap: isSRModelInitialed
                  ? () {
                      if (isRecording) {
                        setState(() {
                          isRecording = false;
                          statusController.text = "";
                          textShow = "开始录音";
                        });
                        hideVoiceView();
                      } else {
                        _count = 0;
                        setState(() {
                          isRecording = true;
                          statusController.text = "正在录音...";
                          textShow = "停止录音";
                        });
                        start();
                        setTimer();
                      }
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
                        paraformerOnline.extractor.reset();
                        paraformerOnline.reset();
                        Map cache = {};
                        List<int>? fCache;
                        try {
                          WavLoader wavLoader = WavLoader();
                          final rawData = await rootBundle.load("assets/audio/asr_example.wav");
                          List<int> intData = List.empty(growable: true);
                          List<int> wavInfo = await wavLoader.loadByteData(rawData);
                          for (var i = wavInfo[0]; i < wavInfo[0] + wavInfo[1]; i += 2) {
                            intData.add(rawData.getInt16(i, Endian.little).toInt());
                          }
                          int chunkStep = paraformerOnline.step;
                          int spLen = intData.length;
                          int spOff = 0;
                          List<int> flags;
                          String finalResult = "";
                          for (spOff = 0; spOff < intData.length; spOff += chunkStep) {
                            if (spOff + chunkStep >= intData.length - 1) {
                              chunkStep = spLen - spOff;
                              flags = [1];
                            } else {
                              flags = [0];
                            }
                            Set result = await paraformerOnline
                                .predictAsync({"waveform": intData.sublist(spOff, spOff + chunkStep), "flags": flags, "cache": cache, "f_cache":fCache});
                            if(result.first == ""){
                              finalResult += "，";
                            }else{
                              finalResult += result.first;
                            }
                            cache = result.elementAt(1);
                            fCache = result.last;
                            print(result.first);
                          }
                          print(finalResult);
                          // await inference(intData);
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
                padding: const EdgeInsets.only(left: 10, top: 0, right: 10, bottom: 0),
                child: ScrollableTextField(
                  controller: resultController,
                  hintText: '识别结果',
                )
                // TextFormField(
                //   // expands: true,
                //   controller: resultController,
                //   decoration: const InputDecoration(
                //     hintText: '识别结果',
                //     border: OutlineInputBorder(),
                //   ),
                //   readOnly: true,
                //   minLines: null,
                //   maxLines: null,
                // ),
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
                      padding: EdgeInsets.only(left: 10, top: 10, right: 10, bottom: 0),
                      child: Text(
                        "是否使用VAD",
                        style: TextStyle(fontSize: 16),
                      ),
                    ),
                    Padding(
                        padding: const EdgeInsets.only(left: 0, top: 10, right: 0, bottom: 0),
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
                                vaDetector?.initModel("assets/models/fsmn_vad.onnx");
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
                      padding: EdgeInsets.only(left: 10, top: 10, right: 10, bottom: 0),
                      child: Text(
                        "使用标点模型",
                        style: TextStyle(fontSize: 16),
                      ),
                    ),
                    Padding(
                        padding: const EdgeInsets.only(left: 0, top: 10, right: 0, bottom: 0),
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

import 'dart:collection';
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
import 'package:srapp_online/utils/similarity_text_index.dart';
import 'package:srapp_online/utils/sound_utils.dart';
import 'package:srapp_online/utils/speech_recognizer.dart';
import 'package:srapp_online/utils/type_converter.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:logger/logger.dart';

// 录音时，每次获取数据后，判断长度是否满足要求，如何足够长就传给 paraformer online 进行推理
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
      debugShowCheckedModeBanner: false,
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

  List<int> waveform = List<int>.empty(growable: true);

  String? audioPath;
  StreamSubscription? recordingDataSubscription;
  bool mRecorderIsInitialed = false;

  List<String> tempAudioPaths = List.empty(growable: true);

  SpeechRecognizer? speechRecognizer;
  FsmnVaDetector? vaDetector;
  ErniePunctuation? erniePunctuation;

  Map cache = {};
  List<int>? fCache;

  static const sampleRate = 16000;

  bool isSRModelInitialed = false;
  bool isRecording = false;
  bool isRecognizing = false; // 是否正在识别

  String? recognizeResult;
  bool usePunc = false;

  Timer? vadTimer;
  Timer? srTimer;

  String keyword = "开始识别"; // 关键词识别
  String cacheText = ""; // 用于储存 keyword 之前的识别结果
  bool lastStepIsWord = false; // 上一步识别的结果是否为文字，是为 ture，如果不是为空，则为false
  // int keywordIdx = 0;        // keyword 的末尾的位置
  int puncIdx = 0; // 标点模型处理文本的起始位置
  String puncedResult = ""; // 标点好的结果

  ParaformerOnline paraformerOnline = ParaformerOnline();

  var logger = Logger(
    filter: null, // Use the default LogFilter (-> only log in debug mode)
    printer: PrettyPrinter(), // Use the PrettyPrinter to format and print log
    output: null, // Use the default LogOutput (-> send everything to console)
  );

  resetRecognition() {
    waveform.clear();
    voice.clear();
    cache = {};
    fCache = null;
    cacheText = "";
    lastStepIsWord = false;
    puncIdx = 0;
    puncedResult = "";
  }

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
    if (isRecognizing) {
      showToastWrapper("正在识别，请稍后");
      return;
    }
    assert(mRecorderIsInitialed);
    // data.clear();
    resetRecognition();
    setState(() {
      resultController.text = "";
    });
    var recordingDataController = StreamController<Food>();
    recordingDataSubscription = recordingDataController.stream.listen((buffer) async {
      if (buffer is FoodData) {
        waveform.addAll(uint8LtoInt16List(buffer.data!));
        if (waveform.length > 9600) {
          onlineInference(waveform.sublist(0, 9600), 0);
          waveform.removeRange(0, 9600);
        }
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
    try {
      if (waveform.length > 16 * 60) {
        onlineInference(waveform, 1);
      } else {
        if (resultController.text != "" || resultController.text != "，") {
          if (resultController.text.endsWith("，")) {
            resultController.text = "${resultController.text.substring(0, resultController.text.length - 1)}。";
          }
        }
        setState(() {
          isRecognizing = false;
          statusController.text = "识别完成";
        });
      }
    } catch (e) {
      logger.e(e.toString());
    }
    if (_timer!.isActive) {
      _timer?.cancel();
    }
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
  }

  // 使用 Paraformer online 进行在线推理
  Future<void> onlineInference(List<int> intData, int isFinal) async {
    Set result = await paraformerOnline.predictAsync({
      "waveform": intData,
      "flags": [isFinal],
      "cache": cache,
      "f_cache": fCache
    });

    if (isRecognizing) {
      if (result.first == "") {
        if (lastStepIsWord) {
          resultController.text += "，";
          // 需要注意此处需要清空 cache 和 fCache，否则可能会出现
          cache = {};
          fCache = null;
        }
        lastStepIsWord = false;
        // if(usePunc && resultController.text.length - puncIdx > 10){
        //   puncIdx = resultController.text.length;
        //   String? tempPuncedResult = await erniePunctuation?.predictAsync(string2List(resultController.text, puncIdx));
        //   if (tempPuncedResult != null){
        //     puncedResult += tempPuncedResult;
        //     resultController.text = puncedResult;
        //   }
        // }
      } else {
        lastStepIsWord = true;
        resultController.text += result.first;
        cache = result.elementAt(1);
        fCache = result.last;
      }
    } else {
      cacheText += result.first;
      cache = result.elementAt(1);
      fCache = result.last;
    }

    if (!isRecognizing) {
      int idx = fuzzySearch(cacheText);
      // logger.i("cacheText: $cacheText, idx: $idx");
      if (idx != -1) {
        cacheText = "";
        // 这里需要将 cacheText 清空，否则在结束录音时，有时由于数据量太小，于是直接设置 isRecognizing 为 false
        // 但是此时可能还有线程在运行语音识别，这样便会再次执行这段函数
        // 如果不清空 cacheText 便会再次设置 isRecognizing = true，导致逻辑错误
        cache = {}; // 此处清空之前关键词唤醒的缓存，可以避免后面在语音识别时重新识别到之间的关键词
        fCache = null;
        isRecognizing = true;
        if (idx <= cacheText.length) {
          resultController.text = cacheText.substring(idx);
        } else {
          resultController.text = "";
        }
        setState(() {
          statusController.text = "开始识别";
        });
      }
    }
    if (isFinal == 1) {
      if (resultController.text != "" || resultController.text != "，") {
        if (resultController.text.endsWith("，")) {
          resultController.text = "${resultController.text.substring(0, resultController.text.length - 1)}。";
        }
      }

      setState(() {
        isRecognizing = false;
        statusController.text = "识别完成";
      });
    }
  }

  int2time(int count) {
    String time = "";
    int hours = (count ~/ 60);
    int minutes = count % 60;

    time = "${hours.toString().padLeft(2, '0')}:${minutes.toString().padLeft(2, '0')}";
    return time;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('语音识别', style: TextStyle(color: Colors.white, fontWeight: FontWeight.w600),),
        backgroundColor: Colors.blueAccent,
      ),
      body: Center(
        child: Column(
          // mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            Padding(
              padding: const EdgeInsets.only(left: 10, top: 10, right: 10, bottom: 0),
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
              height: 20,
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
                        stop();
                      } else {
                        _count = 0;
                        setState(() {
                          isRecording = true;
                          statusController.text = "正在录音...";
                          textShow = "停止录音";
                        });
                        start();
                        _timer = Timer.periodic(const Duration(milliseconds: 1000), (t) {
                          setState(() {
                            _count++;
                          });
                        });
                        // setTimer();
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
                            Set result = await paraformerOnline.predictAsync({
                              "waveform": intData.sublist(spOff, spOff + chunkStep),
                              "flags": flags,
                              "cache": cache,
                              "f_cache": fCache
                            });
                            if (result.first == "") {
                              finalResult += "，";
                            } else {
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
                    : () {
                        showToastWrapper("正在初始化模型，请稍等");
                        return;
                      },
                child: const Text("打开文件")),
            const SizedBox(
              height: 20,
            ),
            Padding(
                padding: const EdgeInsets.only(left: 10, top: 0, right: 10, bottom: 0),
                child: ScrollableTextField(
                  controller: resultController,
                  hintText: '识别结果',
                )),
          ],
        ),
      ),
    );
  }
}

import 'dart:async';
import 'package:audio_session/audio_session.dart';
import 'package:flutter/material.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:keyword_spotting/pages/scrollable_text_field.dart';
import 'package:keyword_spotting/pages/show_toasts.dart';
import 'package:keyword_spotting/utils/fsmnvad_dector.dart';
import 'package:keyword_spotting/utils/ort_env_utils.dart';
import 'package:keyword_spotting/utils/paraformer_online.dart';
import 'package:keyword_spotting/utils/sentence_analysis.dart';
import 'package:keyword_spotting/utils/similarity_text_index.dart';
import 'package:keyword_spotting/utils/sound_utils.dart';
import 'package:keyword_spotting/utils/speech_recognizer.dart';
import 'package:keyword_spotting/utils/type_converter.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:logger/logger.dart';

// 录音时，每次获取数据后，判断长度是否满足要求，如何足够长就传给 vad+biparaformer 进行识别
// 在 main_2 的基础上加上Stream，从而实现顺序执行

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

  String textShow = "开始录音";

  ///默认隐藏状态
  Timer? _timer;
  int _count = 0;

  int startIdx = 0;
  int offset = 400;
  List<List<int>> voice = List<List<int>>.empty(growable: true);

  FlutterSoundRecorder? mRecorder = FlutterSoundRecorder();
  FlutterSoundPlayer? mPlayer = FlutterSoundPlayer();

  List<int> waveform = List<int>.empty(growable: true);

  String? audioPath;
  StreamSubscription? recordingDataSubscription;
  StreamController<Future<void> Function()>? taskController;

  bool mRecorderIsInitialed = false;

  List<String> tempAudioPaths = List.empty(growable: true);

  SpeechRecognizer? speechRecognizer;
  FsmnVaDetector? vaDetector;

  static const sampleRate = 16000;

  final step = 16000;

  bool isSRModelInitialed = false;
  bool isRecording = false;
  bool isRecognizing = false; // 是否正在识别

  String? recognizeResult;
  bool useVAD = false;
  bool usePunc = false;

  Timer? vadTimer;
  Timer? srTimer;

  String keyword = "开始识别"; // 关键词识别
  String cacheText = ""; // 用于储存 keyword 之前的识别结果
  bool thisStepIsWord = false; // 当前步识别的结果是否为文字，是为 ture，如果不是为空，则为false
  String thisStepResult = "";   // 当前步已经识别得到的结果，resultController.text = thisStepResult

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
    speechRecognizer?.release();
    paraformerOnline.release();
    releaseOrtEnv();
    super.dispose();
  }

  void playRemindSound() async {
    await mPlayer!.startPlayer(fromDataBuffer: remindSound, codec: Codec.pcm16WAV);
  }

  resetRecognition(){
    waveform.clear();
    voice.clear();
    cacheText = "";
    thisStepIsWord = false;
    thisStepResult = "";
  }

  ///开始语音录制的方法
  void start() async {
    if(isRecognizing) {
      showToastWrapper("正在识别，请稍等");
      return;
    }
    assert(mRecorderIsInitialed);
    resetRecognition();
    setState(() {
      resultController.text = "";
    });
    var recordingDataController = StreamController<Food>();
    taskController = StreamController();
    executeTasks(taskController!.stream);
    recordingDataSubscription = recordingDataController.stream.listen((buffer) async {
      if (buffer is FoodData) {
        waveform.addAll(uint8LtoInt16List(buffer.data!));
        // print(waveform.length);
        if (waveform.length > step) {    // 如果将其设置为 9600 会出问题，不知道为什么
          List<int> subWaveform = waveform.sublist(0, step);     // 需要先保存截取的数据
          taskController?.add(() async{
            streamingInference(subWaveform, 0);
          });
          waveform.removeRange(0, step);
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
    await mRecorder!.stopRecorder();
    if (_timer!.isActive) {
      _timer?.cancel();
    }
    try {
      if (waveform.length > 3200) {
        int numPadding = 16000 - waveform.length;
        if(numPadding > 0) waveform.addAll(List<int>.generate(numPadding, (m) => 0));

        taskController?.add(()async{
          streamingInference(waveform, 1);
        });
      }else{
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
      print(e.toString());
    }
    await taskController?.close();
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

  Future<void> executeTasks(Stream<Future<void> Function()> taskStream) async {
    await for (var task in taskStream){
      await task();
    }
    setState(() {
      isRecognizing = false;
      statusController.text = "识别完成";
    });
    logger.i("all reorganization tasks have been done");
  }

  streamingInference(List<int> seg, int isFinal) async {
    List<List<int>>? segments;
    segments = await vaDetector?.predictASync(seg);
    if (segments == null || segments.isEmpty) {
      thisStepIsWord = false;
    }else{
      for (var segment in segments) {
        int segB = segment[0] * 16;
        int segE = segment[1] * 16;
        logger.i(
            "分段模型：startIdx 为 $startIdx, 识别波形长度为 ${seg.length}, 原始分段为 [${segment[0]}, ${segment[1]}] 识别段为 [$segB, $segE]");
        if (segB > startIdx + offset && segE < startIdx + step - offset) {
          voice.add(seg.sublist(segB, segE));
        } else if (segB > startIdx + offset && segE >= startIdx + step - offset) {
          voice.add(seg.sublist(segB));
        } else if (segB <= startIdx + offset && segE < startIdx + step - offset) {
          voice.add(seg.sublist(0, segE));
        } else {
          voice.add(seg);
        }
      }
      thisStepIsWord = true;
    }
    if (voice.isEmpty) {
      if(isFinal == 1){
        setState(() {
          isRecognizing = false;
          statusController.text = "识别完成";
        });
      }
      return;
    }

    List<int>? cacheVoice;
    Map? result;
    String cacheTempText = "";
    if (isRecognizing) {
      if (!thisStepIsWord) {   // 如果当前步无语音，且存在之前的语音段，则证明已经暂停了说话
        cacheVoice = concatVoice(voice);
        voice.clear();
        if (cacheVoice == null) {
          return;
        }
        result = await speechRecognize(cacheVoice);
        thisStepResult += "${simpleSentenceProcess(result?["char"])}，";
        resultController.text = thisStepResult;
      } else {           // 如果当前步存在语音，且存在之前的语音段，则证明还在说话，只显示当前结果
        cacheVoice = concatVoice(voice);
        if (cacheVoice == null) {
          voice.clear();
          return;
        }
        result = await speechRecognize(cacheVoice);
        resultController.text = thisStepResult + result?["char"].join("");
      }
    } else {
      if (!thisStepIsWord) {
        cacheVoice = concatVoice(voice);
        voice.clear();
        if (cacheVoice == null) {
          return;
        }
        result = await speechRecognize(cacheVoice);
        cacheText += result?["char"].join();
      } else {
        cacheVoice = concatVoice(voice);
        if (cacheVoice == null) {
          voice.clear();
          return;
        }
        result = await speechRecognize(cacheVoice);
        cacheTempText = cacheText + result?["char"].join();
      }
    }
    logger.i("cacheText: $cacheText, cacheTempText: $cacheTempText");

    if (!isRecognizing) {
      int idx1 = fuzzySearch(cacheText);
      int idx2 = fuzzySearch(cacheTempText);
      if (idx1 != -1) {
        cacheText = "";  // cacheText 清空，否则在结束录音时，有时由于数据量太小，于是直接设置 isRecognizing 为 false
        // 但是此时可能还有线程在运行语音识别，这样便会再次执行这段函数
        // 如果不清空 cacheText 便会再次设置 isRecognizing = true，导致逻辑错误
        cacheTempText = "";
        voice.clear();

        isRecognizing = true;
        if (idx1 <= cacheText.length) {
          resultController.text = cacheText.substring(idx1);
        } else {
          resultController.text = "";
        }
        setState(() {
          statusController.text = "开始识别";
        });
      } else if(idx2 != -1){
        cacheText = "";
        cacheTempText = "";
        isRecognizing = true;
        voice.clear();
        if (idx2 <= cacheText.length) {
          resultController.text = cacheTempText.substring(idx2);
        } else {
          resultController.text = "";
        }
        setState(() {
          statusController.text = "开始识别";
        });
      }
    }

    if (isFinal == 1) {
      voice.clear();
      setState(() {
        isRecognizing = false;
        statusController.text = "识别完成";
      });
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

  int2time(int count) {
    String time = "";
    int hours = (count ~/ 60);
    int minutes = count % 60;

    time = "${hours.toString().padLeft(2, '0')}:${minutes.toString().padLeft(2, '0')}";
    return time;
  }

  List<int>? concatVoice(List<List<int>> voice) {
    List<int> cacheVoice = List<int>.empty(growable: true);
    for (var v in voice) {
      cacheVoice.addAll(v);
    }
    // voice.clear();
    if (cacheVoice.length < 800) {
      return null;
    } else {
      return cacheVoice;
    }
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
              height: 30,
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
                    // statusController.text = "";
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
            const SizedBox(
              height: 20,
            ),
            Padding(
                padding: const EdgeInsets.only(left: 10, top: 0, right: 10, bottom: 0),
                child: ScrollableTextField(
                  controller: resultController,
                  hintText: '识别结果',
                )),
            const SizedBox(
              height: 20,
            ),

          ],
        ),
      ),
    );
  }
}

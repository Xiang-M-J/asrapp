
import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';

class SpeechEmotionRecognizer{
  OrtSessionOptions? _sessionOptions;
  OrtSession? _session;

  static const int _batch = 1;
  reset() {}

  release() {
    _sessionOptions?.release();
    _sessionOptions = null;
    _session?.release();
    _session = null;
  }

  initModel() async {
    _sessionOptions = OrtSessionOptions()
      ..setInterOpNumThreads(1)
      ..setIntraOpNumThreads(1)
      ..setSessionGraphOptimizationLevel(GraphOptimizationLevel.ortEnableAll);
    const assetFileName = 'assets/models/.quant.onnx';
    final rawAssetFile = await rootBundle.load(assetFileName);
    final bytes = rawAssetFile.buffer.asUint8List();
    _session = OrtSession.fromBuffer(bytes, _sessionOptions!);
  }

  int predict(){

  }

}
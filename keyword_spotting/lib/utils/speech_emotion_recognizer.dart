import 'dart:math';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:keyword_spotting/utils/type_converter.dart';
import 'package:onnxruntime/onnxruntime.dart';

import 'feature_utils.dart';

class SpeechEmotionRecognizer {
  OrtSessionOptions? _sessionOptions;
  OrtSession? _session;

  static const labels = ['angry', 'disgusted', 'fear', 'happy', 'neutral', 'other', 'sad', 'surprised', '<unk>'];
  static const labelsMy = ["angry", "fear", "happy", "neutral", "sad", "surprise"];
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
    const assetFileName = 'assets/models/emotion2vec.quant.onnx';
    final rawAssetFile = await rootBundle.load(assetFileName);
    final bytes = rawAssetFile.buffer.asUint8List();
    _session = OrtSession.fromBuffer(bytes, _sessionOptions!);
  }

  initMyModel() async{
    _sessionOptions = OrtSessionOptions()
      ..setInterOpNumThreads(1)
      ..setIntraOpNumThreads(1)
      ..setSessionGraphOptimizationLevel(GraphOptimizationLevel.ortEnableAll);
    const assetFileName = 'assets/models/mtcn.quant.onnx';
    final rawAssetFile = await rootBundle.load(assetFileName);
    final bytes = rawAssetFile.buffer.asUint8List();
    _session = OrtSession.fromBuffer(bytes, _sessionOptions!);
  }

  int argmax(List<double> logits) {
    int idx = 0;
    double max = -1e8;
    for (var i = 0; i < logits.length; i++) {
      if (logits[i] > max) {
        max = logits[i];
        idx = i;
      }
    }
    return idx;
  }

  Future<String?> predictAsync(List<int> data) {
    return compute(predict, data);
  }

  mean(List<double> doubleData) {
    double m = 0;
    for (var d in doubleData) {
      m += d;
    }
    return m / doubleData.length;
  }

  variance(List<double> doubleData, double m) {
    double v = 0;
    for (var d in doubleData) {
      v += (m - d) * (m - d);
    }
    return v / doubleData.length;
  }

  layerNorm(List<double> doubleData) {
    double m = mean(doubleData);
    double v = variance(doubleData, m);
    for (var i = 0; i < doubleData.length ;i++){
      doubleData[i] = (doubleData[i] - m) / sqrt(v + 1e-5);
    }
  }

  String? predict(List<int> data) {
    List<double> doubleData = data.map((m) => m / (1 << 15)).toList();
    layerNorm(doubleData);
    final inputOrt =
        OrtValueTensor.createTensorWithDataList(Float32List.fromList(doubleData), [_batch, doubleData.length]);
    final runOptions = OrtRunOptions();
    final inputs = {'speech': inputOrt};
    final List<OrtValue?>? outputs;
    outputs = _session?.run(runOptions, inputs);
    if (outputs == null) {
      return null;
    }
    inputOrt.release();

    runOptions.release();

    /// Output probability & update h,c recursively
    final logits = (outputs[0]?.value as List<List<double>>)[0];

    for (var element in outputs) {
      element?.release();
    }
    print(logits);
    return labels[argmax(logits)];
  }

  Future<String?> predictMyAsync(List<int> data){
    return compute(predictMy, data);
  }

  String? predictMy(List<int> data){
    final feature = extractFbankOnline(data);
    final inputOrt = OrtValueTensor.createTensorWithDataList(
        doubleList2FloatList(feature), [1, feature.length, 400]);
    final runOptions = OrtRunOptions();
    final inputs = {"feats": inputOrt};
    final List<OrtValue?>? outputs;

    outputs = _session?.run(runOptions, inputs);
    inputOrt.release();
    if(outputs == null) return null;

    runOptions.release();

    final output = (outputs[0]?.value as List<List<double>>)[0];

    for (var element in outputs) {
      element?.release();
    }
    print(output);
    return labelsMy[argmax(output)];
    print(output);
  }
}

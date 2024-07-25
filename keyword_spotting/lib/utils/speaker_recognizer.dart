import 'dart:math';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:keyword_spotting/utils/type_converter.dart';
import 'package:onnxruntime/onnxruntime.dart';

import 'feature_utils.dart';

class SpeechEmotionRecognizer {
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
    String assetFileName = 'assets/models/campplus.quant.onnx';
    final rawAssetFile = await rootBundle.load(assetFileName);
    final bytes = rawAssetFile.buffer.asUint8List();
    _session = OrtSession.fromBuffer(bytes, _sessionOptions!);
  }

  Future<List<List<int>>?> predictAsync(List<List<int>> data) async {
    if (data.length == 1) return null;
    List<List<double>> embeddings = [];
    for (var i = 0; i < data.length; i++) {
      List<double>? e = await compute(predict, data[i]);
      if (e == null) return null;
      embeddings.add(e);
    }
    List<List<int>> m = [];
    List<int> waitForGroup = List.empty(growable: true);
    for(var i = 0; i<embeddings.length; i++){
      waitForGroup.add(i);
    }
    for (var i = 0; i < embeddings.length; i++) {
      waitForGroup.remove(i);
      m.add([i]);
      if(waitForGroup.isEmpty){
        break;
      }
      for (var j in waitForGroup){
        double sim = await calCosineSimilarity(embeddings[i], embeddings[j]);
        if(sim > 0.8){
          m.last.add(j);
          waitForGroup.remove(j);
        }
      }
    }
    return m;
  }
  
  Future<double> calCosineSimilarity(List<double> e1, List<double> e2) async{
    return compute(cosineSimilarity, [e1, e2]);
  }
  

  double cosineSimilarity(List<List<double>> e)  {
    double c = 0;
    for (var i = 0; i < e[0].length; i++) {
      c += e[0][i] * e[1][i];
    }
    return c / (norm(e[0]) * norm(e[1]) + 1e-5);
  }

  norm(List<double> e) {
    double s = e.reduce((a, b) => a + b);
    return sqrt(s);
  }

  List<double>? predict(List<int> data) {
    List<List<double>> feats = extractFbankOnly(data);
    List<Float32List> floatList = doubleList2FloatList(feats);
    final inputOrt = OrtValueTensor.createTensorWithDataList(floatList, [_batch, feats.length, feats[0].length]);
    final runOptions = OrtRunOptions();
    final inputs = {'feats': inputOrt};
    final List<OrtValue?>? outputs;
    outputs = _session?.run(runOptions, inputs);
    if (outputs == null) {
      return null;
    }
    inputOrt.release();

    runOptions.release();

    /// Output probability & update h,c recursively
    final embedding = (outputs[0]?.value as List<List<double>>)[0];

    for (var element in outputs) {
      element?.release();
    }
    return embedding;
  }
}

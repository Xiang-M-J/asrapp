import 'dart:math';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:keyword_spotting/utils/type_converter.dart';
import 'package:onnxruntime/onnxruntime.dart';

import 'feature_utils.dart';


class SpeakerRecognizer {
  OrtSessionOptions? _sessionOptions;
  OrtSession? _session;

  List<List<double>> cacheEmbeddings = [];

  static const int _batch = 1;
  reset() {}

  release() {
    cacheEmbeddings.clear();
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

  Future<int?> predictAsyncWithCache(List<int> data) async {

    List<double>? e = await compute(predict, data);
    if (e == null) return null;
    if(cacheEmbeddings.isEmpty){
      cacheEmbeddings.add(e);
      return cacheEmbeddings.length - 1;
    }
    for(var i = 0; i<cacheEmbeddings.length; i++){
      double sim = cosineSimilarity(e, cacheEmbeddings[i]);
      print(sim);
      if(sim > 0.5) {
        cacheEmbeddings[i] = e;
        return i;
      }
    }
    cacheEmbeddings.add(e);
    return cacheEmbeddings.length - 1;
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
    while(waitForGroup.isNotEmpty){
      int idx = waitForGroup.first;
      waitForGroup.removeWhere((v) => v == idx);
      m.add([idx]);
      if (waitForGroup.isEmpty){
        break;
      }
      List<int> waitForDeleted = [];
      for (var j in waitForGroup){
        double sim = cosineSimilarity(embeddings[idx], embeddings[j]);
        print(sim);
        if(sim > 0.75){
          m.last.add(j);
          waitForDeleted.add(j);
        }
      }
      for(var j in waitForDeleted){
        waitForGroup.remove(j);
      }
    }
    return m;
  }


  double cosineSimilarity(List<double> e1, List<double> e2)  {
    double c = 0;
    for (var i = 0; i < e1.length; i++) {
      c += e1[i] * e2[i];
    }
    return c / (norm(e1) * norm(e2) + 1e-5);
  }

  norm(List<double> e) {
    double s = e.map((a) => a * a).reduce((a, b) => a  + b);
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

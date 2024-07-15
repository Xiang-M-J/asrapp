import 'dart:math';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:srapp_online/utils/tokenizer.dart';
import 'package:srapp_online/utils/type_converter.dart';
import 'package:onnxruntime/onnxruntime.dart';

import 'feature_utils.dart';

class SpeechRecognizer {
  OrtSessionOptions? _sessionOptions;
  OrtSession? _session;

  static const int _batch = 1;

  Tokenizer tokenizer = Tokenizer();
  SpeechRecognizer();

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
    const assetFileName = 'assets/models/BiCifParaformer.quant.onnx';
    final rawAssetFile = await rootBundle.load(assetFileName);
    final bytes = rawAssetFile.buffer.asUint8List();
    _session = OrtSession.fromBuffer(bytes, _sessionOptions!);
    await tokenizer.init();
  }

  initModelAsync() async {
    const assetFileName = 'assets/models/BiCifParaformer.quant.onnx';
    final rawAssetFile = await rootBundle.load(assetFileName);
    await tokenizer.init();
    return compute(_initModel, rawAssetFile);
  }

  bool _initModel(ByteData rawAssetFile) {
    _sessionOptions = OrtSessionOptions()
      ..setInterOpNumThreads(1)
      ..setIntraOpNumThreads(1)
      ..setSessionGraphOptimizationLevel(GraphOptimizationLevel.ortEnableAll);

    final bytes = rawAssetFile.buffer.asUint8List();
    _session = OrtSession.fromBuffer(bytes, _sessionOptions!);
    return true;
  }

  List<String> greedyDecode(List<List<double>> logits) {
    final predictedIds = logits.map((e) => maxIndex(e)).toList();

    final decodedOutput = tokenizer.id2Token(predictedIds);

    // decoded_output = [token for token in decoded_output if token not in ('<s>', '<pad>', '</s>', '<unk>')]
    return decodedOutput;
  }

  int maxIndex(List<double> values) {
    int index = 0;
    double preValue = -10000;
    for (var i = 0; i < values.length; i++) {
      if (values[i] > preValue) {
        index = i;
        preValue = values[i];
      }
    }
    return index;
  }

  padSequence(List<List<int>> intDataAll, int maxSegmentLen) {
    if (intDataAll.length == 1) {
    } else {
      for (var i = 0; i < intDataAll.length; i++) {
        int numPadding = maxSegmentLen - intDataAll[i].length;
        if (numPadding == 0) continue;
        for (var j = 0; j < numPadding; j++) {
          intDataAll[i].add(0);
        }
      }
    }
  }

  padSequenceFloat(List<List<Float32List>> dataAll, int maxSegmentLen) {
    if (dataAll.length == 1) {
    } else {
      for (var i = 0; i < dataAll.length; i++) {
        int numPadding = maxSegmentLen - dataAll[i].length;
        if (numPadding == 0) continue;
        for (var j = 0; j < numPadding; j++) {
          dataAll[i].add(Float32List(560));
        }
      }
    }
  }

  Future<Map<String, List>?> predictWithVADAsync(List<int> data, List<List<int>> segments) {
    Map input = {"data": data, "segments": segments};
    return compute(predictWithVAD, input);
  }

  Map<String, List>? predictWithVAD(Map input) {
    List<List<int>> intDataAll = [];
    int maxSegmentLen = 0;
    // Map sorted_segments = {};
    // for (var i = 0; i < input["segments"].length; i++) {
    //   sorted_segments[i] = input["segments"][i];
    // }
    // sorted_segments = SplayTreeMap.from(
    //     sorted_segments,
    //     (key1, key2) => (sorted_segments[key1][1] - sorted_segments[key1][0])!
    //         .compareTo(sorted_segments[key2][1] - sorted_segments[key2][0]));
    for (var i = 0; i < input["segments"].length; i++) {
      int begin = max(0, input["segments"][i][0] * 16);
      int end = min(input["data"].length, input["segments"][i][1] * 16);
      if (end - begin < 80) {
        continue;
      }
      maxSegmentLen = max(maxSegmentLen, end - begin);

      intDataAll.add(input["data"].sublist(begin, end));
    }
    int batch = intDataAll.length;
    List<int> feats_len = [];
    int max_feats_len = 0;
    // padSequence(intDataAll, maxSegmentLen);
    List<List<Float32List>> floatData = [];
    for (var i = 0; i < batch; i++) {
      floatData.add(doubleList2FloatList(extractFbank(intDataAll[i])));
      feats_len.add(floatData.last.length);
      max_feats_len = max(max_feats_len, feats_len.last);
    }
    padSequenceFloat(floatData, max_feats_len);
    int axis1 = floatData[0].length;
    int axis2 = floatData[0][0].length;
    final inputOrt = OrtValueTensor.createTensorWithDataList(
        floatData, [batch, axis1, axis2]);

    final lengthOrt = OrtValueTensor.createTensorWithDataList(
        Int32List.fromList(feats_len), [batch]);

    final runOptions = OrtRunOptions();
    final inputs = {'speech': inputOrt, "speech_lengths": lengthOrt};
    final List<OrtValue?>? outputs;
    // if (concurrent) {
    //   outputs = await _session?.runAsync(runOptions, inputs);
    // } else {
    //   outputs = _session?.run(runOptions, inputs);
    // }
    outputs = _session?.run(runOptions, inputs);

    if (outputs == null) {
      return null;
    }
    inputOrt.release();

    runOptions.release();

    /// Output probability & update h,c recursively
    List<List<List<double>>> logits =
        (outputs[0]?.value as List<List<List<double>>>);
    // final toke_num = (outputs[1]?.value as List<double>)[0];
    List<List<double>> usAlphas = (outputs[2]?.value as List<List<double>>);
    List<List<double>> usCifPeak = (outputs[3]?.value as List<List<double>>);

    Map<String, List> result = {
      "char": [],
      "timestamp": [],
      "segments": List<int>.empty(growable: true)
    };
    for (var i = 0; i < batch; i++) {
      List<String> charList = greedyDecode(logits[i]);

      List<List<int>> timestamp =
          getTimeStamp(usAlphas[i], usCifPeak[i], charList);
      int idx = charList.indexOf("</s>");
      if (idx != -1) {
        charList = charList.sublist(0, idx);
        timestamp = timestamp.sublist(0, idx);
      }

      result["char"]?.addAll(charList);
      result["timestamp"]?.addAll(timestamp);
      result["segments"]?.add(charList.length);
    }

    for (var element in outputs) {
      element?.release();
    }

    return result;
  }

  Future<Map<String, List<dynamic>>?> predictAsync(List<int> data) {
    return compute(predict, data);
  }

  Map<String, List<dynamic>>? predict(List<int> intData) {
    List<List<double>> fbank = extractFbank(intData);
    int axis1 = fbank.length;
    int axis2 = fbank[0].length;
    List<Float32List> floatData = doubleList2FloatList(fbank);
    final inputOrt = OrtValueTensor.createTensorWithDataList(
        floatData, [_batch, axis1, axis2]);

    final lengthOrt = OrtValueTensor.createTensorWithDataList(
        Int32List.fromList([axis1]), [_batch]);

    final runOptions = OrtRunOptions();
    final inputs = {'speech': inputOrt, "speech_lengths": lengthOrt};
    final List<OrtValue?>? outputs;
    // if (concurrent) {
    //   outputs = await _session?.runAsync(runOptions, inputs);
    // } else {
    //   outputs = _session?.run(runOptions, inputs);
    // }
    outputs = _session?.run(runOptions, inputs);

    if (outputs == null) {
      return null;
    }
    inputOrt.release();

    runOptions.release();

    /// Output probability & update h,c recursively
    final logits = (outputs[0]?.value as List<List<List<double>>>)[0];
    // final toke_num = (outputs[1]?.value as List<double>)[0];
    final usAlphas = (outputs[2]?.value as List<List<double>>)[0];
    final usCifPeak = (outputs[3]?.value as List<List<double>>)[0];

    List<String> charList = greedyDecode(logits);

    List<List<int>> timestamp = getTimeStamp(usAlphas, usCifPeak, charList);

    for (var element in outputs) {
      element?.release();
    }

    // final result = greedy_decode(output);
    // print(result);
    Map<String, List<dynamic>> result = {};
    if (charList.last == "</s>") {
      charList.removeLast();
    }
    result["char"] = charList;
    result["timestamp"] = timestamp;
    // for(var i = 0; i<charList.length; i++){
    //   result[charList[i]] = timestamp[i];
    // }
    return result;
  }

  puncByVAD(List<List<int>> segments, Map result) {
    List<int> endMs = segments.map((e) => e[1]).toList();
    String decodeResult = "";
    int idx = 0;
    if (result.keys.contains("segments")) {
      List<int> seg = result["segments"];
      int b = 0;
      for (var s in seg) {
        decodeResult += result["char"].sublist(b, b + s).join(" ");
        decodeResult += "，";
        b += s;
      }
      if (decodeResult.endsWith("，")) {
        decodeResult = decodeResult.substring(0, decodeResult.length - 1);
      }
      decodeResult += "。";
      return decodeResult;
    }
    if (result["char"].length != result["timestamp"].length) {
      decodeResult = result["char"].join(" ") + ".";
    } else {
      for (var i = 0; i < result["char"].length; i++) {
        String char = result["char"][i];
        List<int> timestamp = result["timestamp"][i];
        if (timestamp[1] < endMs[idx]) {
          decodeResult += char;
        } else if (timestamp[0] > endMs[idx]) {
          decodeResult += "，$char";
          idx++;
          if (idx == endMs.length) {
            break;
          }
        } else {
          decodeResult += char;
        }
      }
    }
    decodeResult += "。";
    return decodeResult;
  }

  List<List<int>> getTimeStamp(
    List<double> usAlphas,
    List<double> usPeaks,
    List<String> charList, [
    double vadOffset = 0.0,
    double forceTimeShift = -1.5,
    bool silInStr = true,
    int upsampleRate = 3,
  ]) {
    int startEndThreshold = 5;
    int maxTokenDuration = 12;
    double timeRate = 10.0 * 6 / 1000 / upsampleRate;
    if (charList.last == "</s>") {
      charList = charList.sublist(0, charList.length - 1);
    }

    List<double> firePlace = List.empty(growable: true);
    for (var i = 0; i < usPeaks.length; i++) {
      if (usPeaks[i] >= 1.0 - 1e-4) {
        firePlace.add(i + forceTimeShift);
      }
    }
    double usAlphasSum = usAlphas.reduce((v, e) => (v + e));

    if (firePlace.length != charList.length + 1) {
      usAlphas = usAlphas
          .map((e) => (e / (usAlphasSum / (charList.length + 1))))
          .toList();
      usPeaks = cif_wo_hidden(usAlphas, 1.0 - 1e-4);
      firePlace.clear();
      for (var i = 0; i < usPeaks.length; i++) {
        if (usPeaks[i] >= 1.0 - 1e-4) {
          firePlace.add(i + forceTimeShift);
        }
      }
    }
    int numFrames = usPeaks.length;

    List<List<double>> timestampList = [];
    List<String> newCharList = [];
    if (firePlace[0] > startEndThreshold) {
      timestampList.add([0.0, firePlace[0] * timeRate]);
      newCharList.add("<sil>");
    }
    for (var i = 0; i < firePlace.length - 1; i++) {
      newCharList.add(charList[i]);
      if (maxTokenDuration < 0 ||
          firePlace[i + 1] - firePlace[i] <= maxTokenDuration) {
        timestampList
            .add([firePlace[i] * timeRate, firePlace[i + 1] * timeRate]);
      } else {
        double split = firePlace[i] + maxTokenDuration;
        timestampList.add([firePlace[i] * timeRate, split * timeRate]);
        timestampList.add([split * timeRate, firePlace[i + 1] * timeRate]);
        newCharList.add("<sil>");
      }
    }
    if (numFrames - firePlace.last > startEndThreshold) {
      var _end = (numFrames + firePlace.last) * 0.5;
      // _end = fire_place[-1]
      timestampList.last[1] = _end * timeRate;
      timestampList.add([_end * timeRate, numFrames * timeRate]);
      newCharList.add("<sil>");
    } else {
      timestampList.last[1] = numFrames * timeRate;
    }
    if (vadOffset != 0) {
      for (var i = 0; i < timestampList.length;) {
        timestampList[i][0] = timestampList[i][0] + vadOffset / 1000.0;
        timestampList[i][1] = timestampList[i][1] + vadOffset / 1000.0;
      }
    }
    List<List<int>> res = [];
    for (var i = 0; i < newCharList.length; i++) {
      String char = newCharList[i];
      if (char != "<sil>") {
        List<double> timestamp = timestampList[i];
        res.add([(timestamp[0] * 1000).toInt(), (timestamp[1] * 1000).toInt()]);
      }
    }
    return res;
  }

  cif_wo_hidden(List<double> alphas, double threshold) {
    int lenTime = alphas.length;
    double integrate = 0;

    // intermediate vars along time
    List<double> listFires = List.empty(growable: true);
    for (var t = 0; t < lenTime; t++) {
      var alpha = alphas[t];
      integrate += alpha;
      listFires.add(integrate);
      var firePlace = integrate >= threshold;
      if (firePlace) {
        integrate -= 1;
      }
    }
    return listFires;
  }
}

import 'dart:math';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:srapp_online/utils/sentence_analysis.dart';
import 'package:srapp_online/utils/tokenizer.dart';
import 'package:srapp_online/utils/type_converter.dart';

import 'feature_utils.dart';

sinusoidalPositionEncoderOnline(List<List<double>> x, {int startIdx = 0}) {
  int timeSteps = x.length;
  int inputDim = x[0].length;
  List<int> positions = List<int>.generate(timeSteps + startIdx, (m) => m + 1);
  double logTimescaleIncrement = log(10000) / (inputDim / 2 - 1);
  int len = (inputDim / 2).ceil();
  List<double> invTimescales = List.generate(len, (m) => exp(-logTimescaleIncrement * m));
  for (var i = startIdx; i < startIdx + timeSteps; i++) {
    for (var j = 0; j < len; j++) {
      x[i - startIdx][j] += sin(positions[i] * invTimescales[j]);
    }
    for (var j = len; j < 2 * len; j++) {
      x[i - startIdx][j] += cos(positions[i] * invTimescales[j - len]);
    }
  }
  return x;
}

class ParaformerOnline {
  List<int> chunkSize = [5, 10, 5];
  int intraOpNumThreads = 4;
  int encoderOutputSize = 512;
  int fsmnLayer = 16;
  int fsmnLorder = 10;
  int fsmnDims = 512;
  int featsDims = 80 * 7;
  double cifThreshold = 1.0;
  double tailThreshold = 0.45;
  int step = 9600;
  // Map cache = {};
  OrtSessionOptions? _sessionOptions;
  OrtSession? encoderSession;
  OrtSession? decoderSession;
  Tokenizer tokenizer = Tokenizer();
  final assetEncoderFileName = 'assets/models/ParaformerEncoder.quant.onnx';
  final assetDecoderFileName = 'assets/models/ParaformerDecoder.quant.onnx';
  final extractor = WavFrontendWithCache();

  ParaformerOnline() {
    step = chunkSize[1] * 960;
  }

  release() {
    _sessionOptions?.release();
    _sessionOptions = null;
    encoderSession?.release();
    encoderSession = null;
    decoderSession?.release();
    decoderSession = null;
    // cache.clear();
  }

  void reset() {
    // cache.clear();
  }

  initModel() async {
    _sessionOptions = OrtSessionOptions()
      ..setInterOpNumThreads(1)
      ..setIntraOpNumThreads(intraOpNumThreads)
      ..setSessionGraphOptimizationLevel(GraphOptimizationLevel.ortEnableAll);

    final rawAssetEncoderFile = await rootBundle.load(assetEncoderFileName);
    final bytes1 = rawAssetEncoderFile.buffer.asUint8List();
    encoderSession = OrtSession.fromBuffer(bytes1, _sessionOptions!);

    final rawAssetDecoderFile = await rootBundle.load(assetDecoderFileName);
    final bytes2 = rawAssetDecoderFile.buffer.asUint8List();
    decoderSession = OrtSession.fromBuffer(bytes2, _sessionOptions!);

    await tokenizer.init();
  }

  prepareCache(Map cache) {
    cache["start_idx"] = 0;
    cache["cif_hidden"] = List<double>.filled(encoderOutputSize, 0.0); // b, 1, encoderOutputSize
    cache["cif_alphas"] = List<double>.filled(1, 0.0); // b, 1
    cache["chunk_size"] = chunkSize;
    cache["last_chunk"] = false;
    cache["feats"] = List<List<double>>.generate(
        chunkSize[0] + chunkSize[2], (m) => List<double>.filled(featsDims, 0.0)); // b, c[0]+c[1], featsDims
    cache["decoder_fsmn"] = List<List<List<double>>>.empty(growable: true); // fsmnLayer, b, fsmnDims, fsmnLorder

    for (var i = 0; i < fsmnLayer; i++) {
      cache["decoder_fsmn"].add(List<List<double>>.generate(fsmnDims, (m) => List<double>.filled(fsmnLorder, 0.0)));
    }
    return cache;
  }

  addOverlapChunk(List<List<double>> feats, Map cache) {
    if (cache.isEmpty) {
      return feats;
    }
    List<List<double>> overlapFeats = [...cache["feats"]];
    overlapFeats.addAll(feats);
    if (cache["is_final"]) {
      cache["feats"] = overlapFeats.sublist(overlapFeats.length - chunkSize[0]);
      if (!cache["last_chunk"]) {
        int paddingLength = chunkSize[0] + chunkSize[1] + chunkSize[2] - overlapFeats.length;
        overlapFeats.addAll(List<List<double>>.filled(paddingLength, List<double>.filled(featsDims, 0.0)));
      }
    } else {
      cache["feats"] = overlapFeats.sublist(overlapFeats.length - chunkSize[0] - chunkSize[2]);
    }
    return overlapFeats;
  }

  predictAsync(Map<String, dynamic> inputs) async {
    return compute(predict, inputs);
  }

  predict(Map<String, dynamic> inputs) {
    List<int>? waveform = inputs["waveform"];
    List<int>? flags = inputs["flags"];
    Map cache = inputs["cache"];
    List<int>? fCache = inputs["f_cache"];
    if (waveform == null || flags == null) return null;
    bool isFinal = flags[0] == 1;

    if (waveform.length < 16 * 60 && isFinal && cache.isNotEmpty) {
      cache["last_chunk"] = true;
      List<List<double>> feats = cache["feats"];
      String asrResult = infer(feats, cache);
      return {asrResult, cache, extractor.cache};
    }
    extractor.cache = fCache;
    List<List<double>> feats = extractor.extractOnlineFeature(waveform);

    if (cache.isEmpty) {
      cache = prepareCache(cache);
    }
    cache["is_final"] = isFinal;

    feats = sinusoidalPositionEncoderOnline(feats, startIdx: cache["start_idx"]);
    cache["start_idx"] += feats.length;
    if (isFinal) {
      if (feats.length + chunkSize[2] <= chunkSize[1]) {
        cache["last_chunk"] = true;
        feats = addOverlapChunk(feats, cache);
      } else {
        List<List<double>> featsChunk1;
        if (chunkSize[1] >= feats.length) {
          featsChunk1 = addOverlapChunk(feats, cache);
        } else {
          featsChunk1 = addOverlapChunk(feats.sublist(0, (chunkSize[1])), cache);
        }

        String asrResChunk1 = infer(featsChunk1, cache);

        cache["last_chunk"] = true;
        List<List<double>> featsChunk2 = addOverlapChunk(feats.sublist(chunkSize[1] - chunkSize[2]), cache);
        String asrResChunk2 = infer(featsChunk2, cache);

        String asrResChunk = asrResChunk1 + asrResChunk2;
        return {asrResChunk, cache, extractor.cache};
      }
    } else {
      feats = addOverlapChunk(feats, cache);
    }
    String asrRes = infer(feats, cache);
    return {asrRes, cache, extractor.cache};
  }

  vectorAddMul(List<num> a, List<num> b, num m) {
    for (var i = 0; i < a.length; i++) {
      a[i] += m * b[i];
    }
    return a;
  }

  cifSearch(List<List<double>> hidden, List<double> alphas, Map cache) {
    int timeLen = hidden.length;
    int hiddenSize = hidden[0].length;
    List<int> tokenLength = [];
    List<List<double>> listFrame = List.empty(growable: true);
    List<double> listFire = List.empty(growable: true);
    List<double> cacheAlphas = [];
    List<double> cacheHiddens = [];

    if (cache.isNotEmpty && cache.containsKey("cif_alphas") && cache.containsKey("cif_hidden")) {
      hidden.insert(0, cache["cif_hidden"]);
      alphas.insertAll(0, cache["cif_alphas"]);
    }
    if (cache.isNotEmpty && cache.containsKey("last_chunk") && cache["last_chunk"]) {
      List<double> tailHidden = List<double>.filled(hiddenSize, 0.0);
      // List<double> tailAlphas = List<double>.filled(1, tailThreshold);
      hidden.add(tailHidden);
      alphas.add(tailThreshold);
    }
    timeLen = alphas.length;
    double integrate = 0.0;
    List<double> frames = List<double>.filled(hiddenSize, 0.0);

    for (var t = 0; t < timeLen; t++) {
      double alpha = alphas[t];
      if (alpha + integrate < cifThreshold) {
        integrate += alpha;
        listFire.add(integrate);
        frames = vectorAddMul(frames, hidden[t], alpha);
      } else {
        frames = vectorAddMul(frames, hidden[t], (cifThreshold - integrate));
        listFrame.add(frames);
        integrate += alpha;
        listFire.add(integrate);
        integrate -= cifThreshold;
        frames = hidden[t].map((m) => m * integrate).toList();
      }
    }

    cacheAlphas.add(integrate);
    if (integrate > 0.0) {
      cacheHiddens = frames.map((m) => m / integrate).toList();
    } else {
      cacheHiddens = frames;
    }
    tokenLength.add(listFrame.length);

    cache["cif_alphas"] = cacheAlphas;
    cache["cif_hidden"] = cacheHiddens;
    return {listFrame, tokenLength};
  }

  String infer(List<List<double>> feats, Map cache) {
    int axis1 = feats.length;
    int axis2 = feats[0].length;
    List<int> length = [axis1];
    List<Float32List> floatData = doubleList2FloatList(feats);
    final inputOrt = OrtValueTensor.createTensorWithDataList(floatData, [1, axis1, axis2]);

    final lengthOrt = OrtValueTensor.createTensorWithDataList(Int32List.fromList(length) as List, [1]);

    final runOptions = OrtRunOptions();
    final inputs = {'speech': inputOrt, "speech_lengths": lengthOrt};
    final List<OrtValue?>? encoderOutputs;

    encoderOutputs = encoderSession?.run(runOptions, inputs);

    if (encoderOutputs == null) {
      return "";
    }
    for (var k in inputs.keys) {
      inputs[k]?.release();
    }
    // inputOrt.release();

    runOptions.release();

    final enc = (encoderOutputs[0]?.value as List<List<List<double>>>)[0];
    int encLens = (encoderOutputs[1]?.value as List<dynamic>)[0];
    final cifAlphas = (encoderOutputs[2]?.value as List<List<double>>)[0];

    for (var element in encoderOutputs) {
      element?.release();
    }

    final searchResult = cifSearch([...enc], [...cifAlphas], cache);
    List<List<double>> acousticEmbeds = searchResult.first;
    List<int> acousticEmbedsLen = searchResult.last;
    if (acousticEmbeds.isNotEmpty) {
      int axis1 = enc.length;
      int axis2 = enc[0].length;
      floatData = doubleList2FloatList(enc);
      final inputOrt = OrtValueTensor.createTensorWithDataList(floatData, [1, axis1, axis2]);

      final lengthOrt = OrtValueTensor.createTensorWithDataList(Int32List.fromList([encLens]), [1]);
      final embedsFloat = doubleList2FloatList(acousticEmbeds);
      final embedsOrt =
          OrtValueTensor.createTensorWithDataList(embedsFloat, [1, acousticEmbeds.length, acousticEmbeds[0].length]);
      final embedsLengthOrt = OrtValueTensor.createTensorWithDataList(Int32List.fromList(acousticEmbedsLen), [1]);

      final runOptions = OrtRunOptions();
      final inputs = {
        'enc': inputOrt,
        "enc_len": lengthOrt,
        "acoustic_embeds": embedsOrt,
        "acoustic_embeds_len": embedsLengthOrt
      };
      axis1 = cache["decoder_fsmn"][0].length;
      axis2 = cache["decoder_fsmn"][0][0].length;
      for (var i = 0; i < cache["decoder_fsmn"].length; i++) {
        List<Float32List> fd = doubleList2FloatList(cache["decoder_fsmn"][i]);
        final cacheOrt = OrtValueTensor.createTensorWithDataList(fd, [1, axis1, axis2]);
        inputs["in_cache_$i"] = cacheOrt;
      }
      final decoderOutput = decoderSession?.run(runOptions, inputs);
      if (decoderOutput == null) {
        return "";
      }
      for (var v in inputs.values) {
        v.release();
      }
      runOptions.release();

      final logits = (decoderOutput[0]?.value as List<List<List<double>>>)[0];
      // final sampleIds = (decoderOutput[1]?.value as List<List<int>>)[0];

      for (var i = 2; i < decoderOutput.length; i++) {
        cache["decoder_fsmn"][i - 2] = (decoderOutput[i]?.value as List<List<List<double>>>)[0];
        int l1 = cache["decoder_fsmn"][i - 2].length;
        int l2 = cache["decoder_fsmn"][i - 2][0].length;
        for (var j = 0; j < l1; j++) {
          cache["decoder_fsmn"][i - 2][j].removeRange(0, l2 - fsmnLorder);
        }
      }
      // final asrRes = [];
      List<String> preds = greedyDecode(logits, acousticEmbedsLen[0]);
      String asrRes = sentencePostprocess(preds);
      return asrRes;
    }
    return "";
  }

  List<String> greedyDecode(List<List<double>> logits, int embedsLen) {
    final predictedIds = logits.map((e) => maxIndex(e)).toList();

    final decodedOutput = tokenizer.id2Token(predictedIds);
    // decodedOutput.remove("<s>");
    // decodedOutput.removeWhere((m) => ["<s>", "</s>", "<blank>"].contains(m));
    return decodedOutput.sublist(0, embedsLen);
  }

  int maxIndex(List<double> values) {
    int index = 0;
    double preValue = -100000;
    for (var i = 0; i < values.length; i++) {
      if (values[i] > preValue) {
        index = i;
        preValue = values[i];
      }
    }
    return index;
  }
}

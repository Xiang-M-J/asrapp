import 'dart:math';

import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:srapp_online/utils/tokenizer.dart';

import 'feature_utils.dart';


class SinusoidalPositionEncoderOnline {

  encode(List<int> positions, int depth){
    double logTimescaleIncrement = log(10000) / (depth / 2 - 1);
    List<double> invTimescales = List.generate((depth/2).ceil(), (m) => -logTimescaleIncrement * exp(m));
    // TODO 需要完善操作
    List<List<double>> encoding = List<List<double>>.filled((depth/2).ceil(), List<double>.filled(2 * (depth/2).ceil(), 0.0));
    
    // size: depth / 2  depth
  }

  forward(List<List<double>> x, [int startIdx = 0]) {
    int timeSteps = x.length;
    int inputDim = x[0].length;
    List<int> positions = List<int>.generate(timeSteps + startIdx, (m) => m+1);

    // Tensor positions = np.arange(1, timesteps + 1 + startIdx)[None, :];
    // Tensor positionEncoding = encode(positions, inputDim, dtype: x.dtype);

    // return x + positionEncoding.slice(start: [0, startIdx, 0], end: [batchSize, startIdx + timesteps, inputDim]);
  }
}




class ParaformerOnline{
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
  Map cache = {};
  OrtSessionOptions? _sessionOptions;
  OrtSession? encoderSession;
  OrtSession? decoderSession;
  Tokenizer tokenizer = Tokenizer();
  final assetEncoderFileName = 'assets/models/ParaformerEncoder.quant.onnx';
  final assetDecoderFileName = 'assets/models/ParaformerDecoder.quant.onnx';

  ParaformerOnline(){
    step = chunkSize[1] * 960;
  }

  release() {
    _sessionOptions?.release();
    _sessionOptions = null;
    encoderSession?.release();
    encoderSession = null;
    decoderSession?.release();
    decoderSession = null;
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

  prepareCache(Map cache){
    cache["start_idx"] = 0;
    cache["cif_hidden"] = List<List<double>>.generate(1, (m) => List<double>.filled(encoderOutputSize, 0.0));  // b, 1, encoderOutputSize
    cache["cif_alphas"] = List<double>.filled(1, 0.0);      // b, 1
    cache["chunk_size"] = chunkSize;
    cache["last_chunk"] = false;
    cache["feats"] = List<List<double>>.generate(chunkSize[0]+chunkSize[1],
            (m) => List<double>.filled(featsDims, 0.0));  // b, c[0]+c[1], featsDims
    cache["decoder_fsmn"] = List<List<List<double>>>.empty(growable: true);   // fsmnLayer, b, fsmnDims, fsmnLorder

    for (var i = 0; i< fsmnLayer; i++){
      cache["decoder_fsmn"].add(List<List<double>>.generate(fsmnDims, (m) => List<double>.filled(fsmnLorder, 0.0)));
    }
    return cache;
  }

  addOverlapChunk(List<List<double>> feats, Map cache){
    if (cache.isEmpty){
      return feats;
    }
    List<List<double>> overlapFeats = cache["feats"].addAll(feats);
    if( cache["is_final"]){
      cache["feats"] = overlapFeats.sublist(overlapFeats.length - chunkSize[0]);
      if(! cache["last_chunk"]){
        int paddingLength = chunkSize[0] + chunkSize[1] + chunkSize[2] - overlapFeats.length;
        overlapFeats.addAll(List<List<double>>.filled(paddingLength, List<double>.filled(featsDims, 0.0)));
      }
    }
    else{
      cache["feats"] = overlapFeats.sublist(overlapFeats.length - chunkSize[0] - chunkSize[2]);
    }
    return overlapFeats;
  }

  predictAsync(){

  }

  predict(Map<String, List<int>> inputs){
    List<int>? waveform = inputs["waveform"];
    List<int>? flags = inputs["flags"];
    if(waveform == null || flags == null) return null;
    bool isFinal = flags[0] == 1;

    if(waveform.length < 16 * 60 && isFinal && cache.isNotEmpty){
      cache["last_chunk"] = true;
      List<List<double>> feats = cache["feats"];
      int featsLen = feats.length;
      String asrResult = infer(feats, featsLen, cache);
      return asrResult;
    }
    List<List<double>> feats = extractFbank(waveform, encoderOutputSize: encoderOutputSize, rescale : true);
    int featsLen = feats.length;

    if(featsLen != 0){
      cache["isFinal"] = isFinal;

    }

    if(cache.isEmpty){
      cache = prepareCache(cache);
    }


  }

  String infer(List<List<double>> feats, int featsLen, Map cache){
    return "";
  }



}
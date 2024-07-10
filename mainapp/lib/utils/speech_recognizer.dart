import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:mainapp/utils/tokenizer.dart';
import 'package:onnxruntime/onnxruntime.dart';

class SpeechRecognizer {

  OrtSessionOptions? _sessionOptions;
  OrtSession? _session;


  static const int _batch = 1;

  Tokenizer tokenizer = Tokenizer();
  SpeechRecognizer() {
  }

  reset() {
  }

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
    const assetFileName = 'assets/models/BiCifParaformer.onnx';
    final rawAssetFile = await rootBundle.load(assetFileName);
    final bytes = rawAssetFile.buffer.asUint8List();
    _session = OrtSession.fromBuffer(bytes, _sessionOptions!);
    await tokenizer.init();
  }

  String greedyDecode(List<List<double>> logits) {
    final predictedIds = logits.map((e) => maxIndex(e)).toList();

    final decodedOutput = tokenizer.id2Token(predictedIds);
    
    // decoded_output = [token for token in decoded_output if token not in ('<s>', '<pad>', '</s>', '<unk>')]
    return decodedOutput.join(" ");
  }

  int maxIndex(List<double> values){
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

  // String greedy_decode(List<double> logits){
  //   predicted_ids = torch.argmax(logits, dim=-1);

  //   tokens = [labels[id] for id in predicted_ids];

  //   final tokens = List.generate(, (index) => null);
  //   final decoded_output = List<String>.empty(growable: true);
  //   String prev_token = "";

  //   for (var token in tokens) {
  //     if (token != prev_token) {
  //       decoded_output.add(token);
  //       prev_token = token;
  //     }
  //   }

  //   decoded_output = [token for token in decoded_output if token not in ('<s>', '<pad>', '</s>', '<unk>')]
  //   return decoded_output.toString();
  // }

  doubleList2FloatList(List<List<double>> data){
    List<Float32List> out = List.empty(growable: true);
    for (var i = 0; i < data.length; i++) {
      var flist = Float32List.fromList(data[i]);
      out.add(flist);
    }
    
    return out;
  }

  Future<String?> predictWrapper(List<List<double>> data){
    return compute(predict, data);
  }

  String? predict(List<List<double>> data) {
    List<Float32List> data_f = doubleList2FloatList(data);
    final inputOrt = OrtValueTensor.createTensorWithDataList(
        data_f, [_batch, data.length, data[0].length]);
    
    int len = data.length;
    final lengthOrt = OrtValueTensor.createTensorWithDataList(Int64List.fromList([len]), [_batch]);

    final runOptions = OrtRunOptions();
    final inputs = {'speech': inputOrt, "speech_lengths":lengthOrt};
    final List<OrtValue?>? outputs;
    // if (concurrent) {
    //   outputs = await _session?.runAsync(runOptions, inputs);
    // } else {
    //   outputs = _session?.run(runOptions, inputs);
    // }
    outputs = _session?.run(runOptions, inputs);

    if(outputs == null){
      return null;
    }
    inputOrt.release();

    runOptions.release();

    /// Output probability & update h,c recursively
    final output = (outputs?[0]?.value as List<List<List<double>>>)[0];

    final decodedResult = greedyDecode(output);
    print(decodedResult);

    outputs?.forEach((element) {
      element?.release();
    });
    
    // final result = greedy_decode(output);
    // print(result);
    return decodedResult;
  }
}

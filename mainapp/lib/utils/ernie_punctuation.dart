import 'dart:math';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:mainapp/utils/type_converter.dart';
import 'package:onnxruntime/onnxruntime.dart';


class ErniePunctuation {
  OrtSessionOptions? _sessionOptions;
  OrtSession? _session;

  static const int _batch = 1;
  List<String> puncList = ["", "，", "。", "？", "！", "、"];
  List<String>? vocab;
  bool isInitialed = false;
  ErniePunctuation();

  initVocab() async {
    String tokenString = await rootBundle.loadString("assets/vocab.txt");
    vocab = tokenString.split("\n");
  }

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
    const assetFileName = 'assets/models/ernie_punc.quant.onnx';
    final rawAssetFile = await rootBundle.load(assetFileName);
    final bytes = rawAssetFile.buffer.asUint8List();
    _session = OrtSession.fromBuffer(bytes, _sessionOptions!);
    isInitialed = true;
  }


  String decodeList(List<int> encList, List predPunc ) {

    String puncedText = "";
    for(var i=1; i<encList.length-1 ;i++){
      puncedText += vocab![encList[i]];
      puncedText += puncList[predPunc[i]];
    }

    return puncedText;
  }

  Future<String?> predictAsync(List<String> data) {
    return compute(predict, data);
  }

  encodeText(List<String> text){
    List<int> intList = [1];
    for(var v in text){
      int? idx = vocab?.indexOf(v);
      if(idx != null){
        intList.add(idx);
      }else{
        intList.add(vocab!.length-1);
      }
    }
    intList.add(2);
    return intList;
  }

  String? predict(List<String> text) {
    List<int> intList = encodeText(text);
    int encLen = text.length+2;
    final inputOrt = OrtValueTensor.createTensorWithDataList(
        Int64List.fromList(intList), [_batch, encLen]);

    final tokenOrt = OrtValueTensor.createTensorWithDataList(
        Int64List(encLen), [_batch,encLen]);

    final runOptions = OrtRunOptions();
    final inputs = {'input_ids': inputOrt, "token_type_ids": tokenOrt};
    final List<OrtValue?>? outputs;

    outputs = _session?.run(runOptions, inputs);

    if (outputs == null) {
      return null;
    }
    inputOrt.release();

    runOptions.release();

    /// Output probability & update h,c recursively
    final logits = (outputs[0]?.value as List);


    String puncedText = decodeList(intList, logits);


    return puncedText ;
  }
}

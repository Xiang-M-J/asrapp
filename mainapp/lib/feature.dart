import 'dart:ffi';

import 'dart:io' show Platform;
import 'package:ffi/ffi.dart';



final DynamicLibrary fbankLib = Platform.isAndroid
    ? DynamicLibrary.open("libFbank.so")
    : DynamicLibrary.process();

typedef FbankFunc = Void Function(Pointer<Float>, Pointer<Pointer<Float>>, Int32);

final void Function(Pointer<Float>, Pointer<Pointer<Float>>, int) wavFrontend = fbankLib.lookup<NativeFunction<FbankFunc>>("WavFrontend").asFunction();

Pointer<Float> Uint8list2FloatPointer(List<int> list){
  int s = list.length * 4;
  Pointer<Float> fp = calloc<Float>(s);
  for (var i = 0; i < list.length; i++) {
    fp[i] = list[i].toDouble();
  }
  return fp;
}

List<List<double>> FloatMat2FloatList(Pointer<Pointer<Float>> m, int axis1, int axis2){
  List<List<double>> a = List.empty(growable: true);
  for (var i = 0; i < axis1; i++) {
    List<double> b = List.empty(growable: true);
    for (var j = 0; j < axis2; j++) {
      
      b.add(m[i][j]) ;
    }
    a.add(b);
  }
  return a;
}

List<List<double>> extractFbank(List<int> waveform){
  Pointer<Float> waveformPointer = Uint8list2FloatPointer(waveform);
  int sampleNum = waveform.length;
  int wlen = 25 * 16;
  int inc = 10 * 16;
  int m = 1 + (sampleNum - wlen) ~/ inc;
  int axis1 = (m/6.0).ceil();
  int axis2 = 7 * 80;
  Pointer<Pointer<Float>> output = calloc<Pointer<Float>>(axis1 * 4);
  for (var i = 0; i < axis1; i++) {
    output[i] = calloc<Float>(axis2 * 4);
    for (var j = 0; j < axis2; j++) {
      output[i][j] = 0.0;
    }
  }
  wavFrontend(waveformPointer, output, sampleNum);

  List<List<double>> fbank = FloatMat2FloatList(output, axis1, axis2);
  calloc.free(waveformPointer);
  for (var i = 0; i < axis1; i++) {
    calloc.free(output[i]);
  }
  calloc.free(output);
  return fbank;
}

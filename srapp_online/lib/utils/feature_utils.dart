import 'dart:ffi';

import 'dart:io' show Platform;
import 'dart:math';
import 'package:ffi/ffi.dart';

// https://blog.csdn.net/eieihihi/article/details/119219348

final class Cache extends Struct {
  external Pointer<Float> reserve_waveforms;

  external Pointer<Float> input_cache;
  external Pointer<Float> lfr_splice_cache;

  external Pointer<Float> waveforms;
  external Pointer<Float> fbanks;
  external Pointer<Float> fbanks_len;
}

// 语音唤醒、语音识别改为开关、实时识别

final DynamicLibrary fbankLib = Platform.isAndroid
    ? DynamicLibrary.open("libFbank.so")
    : DynamicLibrary.process();

typedef FbankFunc = Void Function(
    Pointer<Float>, Pointer<Pointer<Float>>, Int32);

final void Function(Pointer<Float>, Pointer<Pointer<Float>>, int) wavFrontend =
    fbankLib.lookup<NativeFunction<FbankFunc>>("WavFrontend").asFunction();
final void Function(Pointer<Float>, Pointer<Pointer<Float>>, int)
    wavFrontendOnline = fbankLib
        .lookup<NativeFunction<FbankFunc>>("WavFrontendOnline")
        .asFunction();

class WavFrontendWithCache {
  List<int>? cache;
  int encoderOutputSize = 512;

  WavFrontendWithCache();

  reset(){
    cache = null;
  }

  extractOnlineFeature(List<int> inputs) {
    if (cache != null) {
      inputs.addAll(cache!);
    }
    Pointer<Float> waveformPointer = uint8list2FloatPointer(inputs);
    int sampleNum = inputs.length;
    int wlen = 25 * 16;
    int inc = 10 * 16;
    int m = 1 + (sampleNum - wlen) ~/ inc;

    cache = inputs.sublist(m * inc);
    int axis1 = (m / 6.0).ceil();
    int axis2 = 7 * 80;
    Pointer<Pointer<Float>> output = calloc<Pointer<Float>>(axis1 * 4);
    for (var i = 0; i < axis1; i++) {
      output[i] = calloc<Float>(axis2 * 4);
      for (var j = 0; j < axis2; j++) {
        output[i][j] = 0.0;
      }
    }
    wavFrontend(waveformPointer, output, sampleNum);
    List<List<double>> fbank = floatMat2FloatListWithRescale(output, axis1, axis2, sqrt(encoderOutputSize));
    calloc.free(waveformPointer);
    for (var i = 0; i < axis1; i++) {
      calloc.free(output[i]);
    }
    calloc.free(output);
    return fbank;
  }
}

Pointer<Float> uint8list2FloatPointer(List<int> list) {
  int s = list.length * 4;
  Pointer<Float> fp = calloc<Float>(s);
  for (var i = 0; i < list.length; i++) {
    fp[i] = list[i].toDouble();
  }
  return fp;
}

List<List<double>> floatMat2FloatList(
    Pointer<Pointer<Float>> m, int axis1, int axis2) {
  List<List<double>> a = List.empty(growable: true);
  for (var i = 0; i < axis1; i++) {
    List<double> b = List.empty(growable: true);
    for (var j = 0; j < axis2; j++) {
      b.add(m[i][j]);
    }
    a.add(b);
  }
  return a;
}

List<List<double>> floatMat2FloatListWithRescale(
    Pointer<Pointer<Float>> m, int axis1, int axis2, double size) {
  List<List<double>> a = List.empty(growable: true);
  for (var i = 0; i < axis1; i++) {
    List<double> b = List.empty(growable: true);
    for (var j = 0; j < axis2; j++) {
      b.add(m[i][j] * size);
    }
    a.add(b);
  }
  return a;
}

List<List<double>> extractFbank(List<int> waveform,
{int encoderOutputSize = 512, bool rescale = false}) {
  Pointer<Float> waveformPointer = uint8list2FloatPointer(waveform);
  int sampleNum = waveform.length;
  int wlen = 25 * 16;
  int inc = 10 * 16;
  int m = 1 + (sampleNum - wlen) ~/ inc;
  int axis1 = (m / 6.0).ceil();
  int axis2 = 7 * 80;
  Pointer<Pointer<Float>> output = calloc<Pointer<Float>>(axis1 * 4);
  for (var i = 0; i < axis1; i++) {
    output[i] = calloc<Float>(axis2 * 4);
    for (var j = 0; j < axis2; j++) {
      output[i][j] = 0.0;
    }
  }
  wavFrontend(waveformPointer, output, sampleNum);
  List<List<double>> fbank;
  // if (rescale) {
  //   fbank = floatMat2FloatListWithRescale(
  //       output, axis1, axis2, sqrt(encoderOutputSize));
  // } else {
    fbank = floatMat2FloatList(output, axis1, axis2);
  // }
  calloc.free(waveformPointer);
  for (var i = 0; i < axis1; i++) {
    calloc.free(output[i]);
  }
  calloc.free(output);
  return fbank;
}

List<List<double>> extractFbankOnline(List<int> waveform) {
  Pointer<Float> waveformPointer = uint8list2FloatPointer(waveform);
  int sampleNum = waveform.length;
  int wlen = 25 * 16;
  int inc = 10 * 16;
  int lfr_m = 5;
  int lfr_n = 1;
  int m = 1 + (sampleNum - wlen) ~/ inc;
  int axis1 = ((m) ~/ (lfr_n));
  int axis2 = lfr_m * 80;
  Pointer<Pointer<Float>> output = calloc<Pointer<Float>>(axis1 * 4);
  for (var i = 0; i < axis1; i++) {
    output[i] = calloc<Float>(axis2 * 4);
    for (var j = 0; j < axis2; j++) {
      output[i][j] = 0.0;
    }
  }
  wavFrontendOnline(waveformPointer, output, sampleNum);

  List<List<double>> fbank = floatMat2FloatList(output, axis1, axis2);
  calloc.free(waveformPointer);
  for (var i = 0; i < axis1; i++) {
    calloc.free(output[i]);
  }
  calloc.free(output);
  return fbank;
}

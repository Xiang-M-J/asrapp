import 'dart:ffi';

import 'dart:io' show Platform;
import 'package:ffi/ffi.dart';


final DynamicLibrary sndfile = Platform.isAndroid
    ? DynamicLibrary.open("libsndfile.so")
    : DynamicLibrary.process();

typedef readFileFunc = Pointer<Short> Function(Pointer<Utf8>);
typedef getFrameSizeFunc = Long Function(Pointer<Utf8>);
typedef readFileShortFunc = Int Function(Pointer<Utf8>, Pointer<Short>, Long);

final Pointer<Short> Function(Pointer<Utf8>) readFile = sndfile.lookup<NativeFunction<readFileFunc>>("read_file").asFunction();
final int Function(Pointer<Utf8>) getFrameSize = sndfile.lookup<NativeFunction<getFrameSizeFunc>>("get_frame_sizes").asFunction();
final int Function(Pointer<Utf8>, Pointer<Short>, int) readFileShort = sndfile.lookup<NativeFunction<readFileShortFunc>>("sf_read_short").asFunction();

List<int> shortPoint2IntList(Pointer<Short> sp, int size){
  List<int> sl = List.empty(growable: true);
  for (var i = 0; i < size ; i++){
    sl.add(sp[i].toSigned(16));
  }
  return sl;
}

List<int> readF(String path){
  Pointer<Utf8> ppath = path.toNativeUtf8();
  int size = getFrameSize(ppath);

  // Pointer<Short> pShort = calloc.allocate(2 * size);
  // for(var i = 0; i < size; i++){
  //   pShort[i]= -1;
  // }
  // int flag = readFileShort(ppath, pShort, size);
  // print(flag);
  Pointer<Short> pShort = readFile(ppath);

  List<int> waveform = shortPoint2IntList(pShort, size);

  calloc.free(ppath);

  calloc.free(pShort);

  // Pointer<Short> output = readFile(ppath);
  // List<int> sl = shortPoint2IntList(output);
  // calloc.free(ppath);
  //
  // calloc.free(output);
  return waveform;
}


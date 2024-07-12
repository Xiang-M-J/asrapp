import 'dart:core';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';

class WavLoader{
  int sampleRate = 16000;
  int bitRate = 32000;
  int channels = 1;
  Future<List<int>> loadUint8List(ByteData fileData) async {
    // if (fileBytes.sublist(0, 4).toString() != "RIFF"){
    //   throw Exception("error format");
    // }
    // ByteData fileData = ByteData.sublistView(fileBytes);

    channels = fileData.getInt16(22, Endian.little).toInt();
    sampleRate = fileData.getInt32(24, Endian.little).toInt();
    bitRate = fileData.getInt32(28, Endian.little).toInt();
    var i = 34;
    for(i = 34; i < 1000; i++){
      if(fileData.getUint16(i, Endian.big)== 0x6461 && fileData.getUint16(i+2, Endian.big)== 0x7461){
        break;
      }
    }
    final dataSize = fileData.getUint32(i+4, Endian.little).toInt();
    return [i + 8, dataSize];
  }

  Future<ByteData> load(String path) async {
    File file = File(path);
    Uint8List fileBytes = await file.readAsBytes();
    if (fileBytes.sublist(0, 4).toString() != "RIFF"){
      throw Exception("error format");
    }
    ByteData fileData = ByteData.sublistView(fileBytes);
    channels = fileData.getInt16(22).toInt();
    sampleRate = fileData.getInt32(24).toInt();
    bitRate = fileData.getInt32(28).toInt();
    var i = 34;
    for(i = 34; i < 1000; i++){
      if(fileData.getUint16(i, Endian.big)== 0x6461 && fileData.getUint16(i+2, Endian.big)== 0x7461){
        break;
      }
    }
    final dataSize = fileData.getUint32(i+4).toInt();
    ByteData wavData = ByteData.sublistView(fileBytes.sublist(i+8, i + 8 + dataSize));
    return wavData;
  }
}

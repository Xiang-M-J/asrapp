import 'dart:core';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:keyword_spotting/utils/sound_utils.dart';

class WavLoader {
  int sampleRate = 16000;
  int bitRate = 32000;
  int channels = 1;
  Future<List<int>> loadByteData(ByteData fileData) async {
    channels = fileData.getInt16(22, Endian.little).toInt();
    sampleRate = fileData.getInt32(24, Endian.little).toInt();
    bitRate = fileData.getInt32(28, Endian.little).toInt();
    var i = 34;
    for (i = 34; i < 1000; i++) {
      if (fileData.getUint16(i, Endian.big) == 0x6461 &&
          fileData.getUint16(i + 2, Endian.big) == 0x7461) {
        break;
      }
    }
    final dataSize = fileData.getUint32(i + 4, Endian.little).toInt();
    return [i + 8, dataSize];
  }

  Future<ByteData> load(String path) async {
    File file = File(path);
    Uint8List fileBytes = await file.readAsBytes();
    if (fileBytes.sublist(0, 4).toString() != "[82, 73, 70, 70]") {
      throw Exception("error format");
    }
    ByteData fileData = ByteData.sublistView(fileBytes);
    channels = fileData.getInt16(22, Endian.little).toInt();
    sampleRate = fileData.getInt32(24, Endian.little).toInt();
    bitRate = fileData.getInt32(28, Endian.little).toInt();
    var i = 34;
    for (i = 34; i < 1000; i++) {
      if (fileData.getUint16(i, Endian.big) == 0x6461 &&
          fileData.getUint16(i + 2, Endian.big) == 0x7461) {
        break;
      }
    }
    final dataSize = fileData.getUint32(i + 4, Endian.little).toInt();
    ByteData wavData =
        ByteData.sublistView(fileBytes.sublist(i + 8, i + 8 + dataSize));
    return wavData;
  }
}

// Future<String?> audioTransformUtils(String originPath) async {
//   String newPath = await getTemporaryAudioPath("wav");
//   try{
//     await FFmpegKit.execute('-i $originPath -ar 16000 -ab 32k -ac 1 $newPath')
//         .then((session) async {
//       final returnCode = await session.getReturnCode();
//       if (ReturnCode.isSuccess(returnCode)) {
//         return newPath;
//       } else if (ReturnCode.isCancel(returnCode)) {
//         return null;
//       } else {
//         return null;
//       }
//     });
//     return newPath;
//   }catch(e){
//     print(e.toString());
//     return null;
//   }
// }

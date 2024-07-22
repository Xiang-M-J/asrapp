import 'dart:typed_data';

List<int> uint8LList2Int16List(List<Uint8List> rawData) {
  List<int> intArray = List.empty(growable: true);

  for (var i = 0; i < rawData.length; i++) {
    ByteData byteData = ByteData.sublistView(rawData[i]);
    for (var i = 0; i < byteData.lengthInBytes; i += 2) {
      intArray.add(byteData.getInt16(i, Endian.little).toInt());
    }
  }
  return intArray;
}

List<int> uint8LtoInt16List(Uint8List rawData) {
  List<int> intArray = List.empty(growable: true);
  ByteData byteData = ByteData.sublistView(rawData);
  for (var i = 0; i < byteData.lengthInBytes; i += 2) {
    intArray.add(byteData.getInt16(i, Endian.little).toInt());
  }
  return intArray;
}

doubleList2FloatList(List<List<double>> data) {
  List<Float32List> out = List.empty(growable: true);
  for (var i = 0; i < data.length; i++) {
    var floatList = Float32List.fromList(data[i]);
    out.add(floatList);
  }
  return out;
}

List<double> intList2doubleList(List<int> intData){
  List<double> doubleData = intData.map((e) => e / 32768).toList();
  return doubleData;
}

List<String> string2List(String result, int startIdx){
  List<String> stringList = result.substring(startIdx).split("");
  stringList.remove("，");
  stringList.remove("。");
  return stringList;
}

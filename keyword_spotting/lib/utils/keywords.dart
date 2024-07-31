import 'package:keyword_spotting/utils/pinyin_utils.dart';

List<String> keywords = ["求求你", "求你了", "别打了", "别打我", "救命啊", "不敢了", "我要打你", "不要打了"];

// Map<String, String> kwmWithTone = {
//   "求求你": "qiu2qiu2ni3",
//   "求你了": "qiu2ni3le5",
//   "别打了": "bie2da3le5",
//   "救命啊": "jiu4ming4a1",
//   "我不敢了": "wo3bu4gan3le5",
//   "不敢了": "bu4gan3le5",
//   "我要打你": "wo3yao4da3ni3",
// };

Map<String, String> pinyin2Keywords = {};

getKeywordsPinyin({bool withTone = true}){
  List<String> pinyins = [];
  for(var v in keywords){
    pinyins.add(getPinyin(v, withTone: withTone));
  }
  return pinyins;
}


// 初始化 pinyin2Keywords
initMap({bool withTone = true}){
  for(var v in keywords){
    pinyin2Keywords[getPinyin(v, withTone: withTone)] = v;
  }
}

void main(){
  // print("const Map<String, String> pinyin2Keywords = {");
  // for(var k in kwmWithTone.keys){
  //   print("\"${kwmWithTone[k]}\": \"$k\", ");
  // }
  // print("};");
}
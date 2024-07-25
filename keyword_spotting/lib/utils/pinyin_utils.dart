import 'lpinyin/pinyin_format.dart';
import 'lpinyin/pinyin_helper.dart';

String getPinyin(String originText, {bool withTone=true}){
  RegExp removePattern = RegExp(r'[a-zA-Z0-9 @.,!?，。]');
  String chinese = originText.replaceAll(removePattern, "");
  String pinyin = "";
  if (withTone){
    pinyin = PinyinHelper.getPinyinE(chinese, separator: "", format: PinyinFormat.WITH_TONE_NUMBER);
  }else{
    pinyin = "${PinyinHelper.getPinyinE(chinese, separator: "|", format: PinyinFormat.WITHOUT_TONE)}|";
  }
  return pinyin;
}

String removeDigital(String originText){
  RegExp removePattern = RegExp(r'[0-9]');
  return originText.replaceAll(removePattern, "");
}

void main() {
  RegExp removePattern = RegExp(r'[a-zA-Z0-9 @.,!?，。]');

  String str = "下雨天";
  str = str.replaceAll(removePattern, "");
  print(getPinyin(str));
}

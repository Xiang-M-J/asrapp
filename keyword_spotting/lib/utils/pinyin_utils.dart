import 'lpinyin/pinyin_format.dart';
import 'lpinyin/pinyin_helper.dart';

String getPinyin(String originText){
  RegExp removePattern = RegExp(r'[a-zA-Z0-9 @.,!?，。]');
  String chinese = originText.replaceAll(removePattern, "");
  String pinyin = PinyinHelper.getPinyinE(chinese, separator: "", format: PinyinFormat.WITH_TONE_NUMBER);
  return pinyin;
}

void main() {
  RegExp removePattern = RegExp(r'[a-zA-Z0-9 @.,!?，。]');

  String str = "下雨天";
  str = str.replaceAll(removePattern, "");
  print(getPinyin(str));
}

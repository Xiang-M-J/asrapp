Map<String, String> keywords = {
  "求求你": "qiu2qiu2ni3",
  "别打了": "bie2da3le5",
  "救命啊": "jiu4ming4a1",
  "我不敢了": "wo3bu4gan3le5",
  "不敢了": "bu4gan3le5",
  "我要打你": "wo3yao4da3ni3",
};

Map<String, String> pinyin2Keywords = {
  "qiu2qiu2ni3": "求求你",
  "bie2da3le5": "别打了",
  "jiu4ming4a1": "救命啊",
  "wo3bu4gan3le5": "我不敢了",
  "bu4gan3le5": "不敢了",
  "wo3yao4da3ni3": "我要打你",
};

getKeywordsPinyin(){
  List<String> pinyins = [];
  for(var v in keywords.values){
    pinyins.add(v);
  }
  return pinyins;
}



void main(){
  print("Map<String, String> pinyin2Keywords = {");
  for(var k in keywords.keys){
    print("\"${keywords[k]}\": \"$k\", ");
  }
  print("};");
}
int fuzzySearch(String cacheText){
  String reversedText = cacheText.split("").reversed.join();

  List<String> keywords = ["别识始开", "频视始开", "识始开"];
  for(var keyowrd in keywords){
    int idx = reversedText.indexOf(keyowrd);
    if(idx != -1) return (reversedText.length - idx);
  }
  return -1;
}

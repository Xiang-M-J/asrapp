import 'dart:core';

bool isChinese(String ch) {
  if ("\u4e00".codeUnitAt(0) <= ch.codeUnitAt(0) &&
          ch.codeUnitAt(0) <= "\u9fff".codeUnitAt(0) ||
      "\u0030".codeUnitAt(0) <= ch.codeUnitAt(0) &&
          ch.codeUnitAt(0) <= "\u0039".codeUnitAt(0)) {
    return true;
  }
  return false;
}

bool isAllChinese(List<String> word) {
  List<String> wordLists = [];
  for (var i in word) {
    String cur = i.toString().replaceAll(" ", "");
    cur = cur.replaceAll("</s>", "");
    cur = cur.replaceAll("<s>", "");
    wordLists.add(cur);
  }

  if (wordLists.isEmpty) {
    return false;
  }

  for (String ch in wordLists) {
    if (!isChinese(ch)) {
      return false;
    }
  }
  return true;
}

bool isStringAllChinese(String word) {
  String nWord = word.replaceAll("</s>", "");
  nWord = word.replaceAll(" ", "");
  nWord = nWord.replaceAll("<s>", "");

  return isChinese(nWord);
}

bool isStringAllAlpha(String word) {
  String nWord = word.replaceAll("</s>", "");
  nWord = word.replaceAll(" ", "");
  nWord = nWord.replaceAll("<s>", "");

  return isAlpha(nWord);
}

bool isAllAlpha(List<String> word) {
  List<String> wordLists = [];
  for (var i in word) {
    String cur = i.toString().replaceAll(" ", "");
    cur = cur.replaceAll("</s>", "");
    cur = cur.replaceAll("<s>", "");
    wordLists.add(cur);
  }

  if (wordLists.isEmpty) {
    return false;
  }

  for (String ch in wordLists) {
    if (!ch.contains(RegExp(r'^[a-zA-Z]+$')) && ch != "'") {
      return false;
    } else if (ch.contains(RegExp(r'^[a-zA-Z]+$')) && isChinese(ch)) {
      return false;
    }
  }
  return true;
}

bool isAlpha(String ch) {
  return ch.contains(RegExp("^[a-zA-Z]+"));
}

List<String> abbrDispose(List<String> words, [List<List<int>>? timeStamp]) {
  int wordsSize = words.length;
  List<String> wordLists = [];
  List<int> abbrBegin = [];
  List<int> abbrEnd = [];
  int lastNum = -1;
  List<List<int>> tsLists = [];
  List<int> tsNums = [];
  int tsIndex = 0;
  int begin = 0;
  int end = 0;

  for (int num = 0; num < wordsSize; num++) {
    if (num <= lastNum) {
      continue;
    }

    if (words[num].length == 1 && isAlpha(words[num])) {
      if (num + 1 < wordsSize &&
          words[num + 1] == " " &&
          num + 2 < wordsSize &&
          words[num + 2].length == 1 &&
          isAlpha(words[num + 2])) {
        // found the begin of abbr
        abbrBegin.add(num);
        num += 2;
        abbrEnd.add(num);
        // to find the end of abbr
        while (true) {
          num += 1;
          if (num < wordsSize && words[num] == " ") {
            num += 1;
            if (num < wordsSize &&
                words[num].length == 1 &&
                isAlpha(words[num])) {
              abbrEnd.removeLast();
              abbrEnd.add(num);
              lastNum = num;
            } else {
              break;
            }
          } else {
            break;
          }
        }
      }
    }
  }

  for (int num = 0; num < wordsSize; num++) {
    if (words[num] == " ") {
      tsNums.add(tsIndex);
    } else {
      tsNums.add(tsIndex);
      tsIndex += 1;
    }
  }
  lastNum = -1;
  for (int num = 0; num < wordsSize; num++) {
    if (num <= lastNum) {
      continue;
    }

    if (abbrBegin.contains(num)) {
      if (timeStamp != null) {
        begin = timeStamp[tsNums[num]][0];
      }
      wordLists.add(words[num].toUpperCase());
      num += 1;
      while (num < wordsSize) {
        if (abbrEnd.contains(num)) {
          wordLists.add(words[num].toUpperCase());
          lastNum = num;
          break;
        } else {
          if (isAlpha(words[num])) {
            wordLists.add(words[num].toUpperCase());
          }
        }
        num += 1;
      }
      if (timeStamp != null) {
        end = timeStamp[tsNums[num]][1];
        tsLists.add([begin, end]);
      }
    } else {
      wordLists.add(words[num]);
      if (timeStamp != null && words[num] != " ") {
         begin = timeStamp[tsNums[num]][0];
         end = timeStamp[tsNums[num]][1];
        tsLists.add([begin, end]);
        begin = end;
      }
    }
  }

  if (timeStamp != null) {
    return wordLists;
  } else {
    return wordLists;
  }
}

String sentencePostprocess(List<String> words, [List<List<int>>? timeStamp]) {
  List<String> middleLists = [];
  List<String> wordLists = [];
  String wordItem = "";
  List<List<int>> tsLists = [];
  int begin = 0;
  int end = 0;
  // wash words lists
  for (var i in words) {
    String word = "";
    word = i;

    if (["<s>", "</s>", "<unk>"].contains(word)) {
      continue;
    } else {
      middleLists.add(word);
    }
  }

  // all chinese characters
  if (isAllChinese(middleLists)) {
    for (var i = 0; i < middleLists.length; i++) {
      wordLists.add(middleLists[i].replaceAll(" ", ""));
    }
    if (timeStamp != null) {
      tsLists = timeStamp;
    }
  }
  // all alpha characters
  else if (isAllAlpha(middleLists)) {
    bool tsFlag = true;
    for (var i = 0; i < middleLists.length; i++) {
      if (tsFlag && timeStamp != null) {
        begin = timeStamp[i][0];
        end = timeStamp[i][1];
      }
      String word = "";
      if (middleLists[i].contains("@@")) {
        word = middleLists[i].replaceAll("@@", "");
        wordItem += word;
        if (timeStamp != null) {
          tsFlag = false;
          end = timeStamp[i][1];
        }
      } else {
        wordItem += middleLists[i];
        wordLists.add(wordItem);
        wordLists.add(" ");
        wordItem = "";
        if (timeStamp != null) {
          tsFlag = true;
          end = timeStamp[i][1];
          tsLists.add([begin, end]);
          begin = end;
        }
      }
    }
  }
  // mix characters
  else {
    bool alphaBlank = false;
    bool tsFlag = true;
     begin = -1;
     end = -1;
    for (var i = 0; i < middleLists.length; i++) {
      if (tsFlag && timeStamp != null) {
        begin = timeStamp[i][0];
        end = timeStamp[i][1];
      }
      String word = "";
      if (isStringAllChinese(middleLists[i])) {
        if (alphaBlank) {
          wordLists.removeLast();
        }
        wordLists.add(middleLists[i]);
        alphaBlank = false;
        if (timeStamp != null) {
          tsFlag = true;
          tsLists.add([begin, end]);
          begin = end;
        }
      } else if (middleLists[i].contains("@@")) {
        word = middleLists[i].replaceAll("@@", "");
        wordItem += word;
        alphaBlank = false;
        if (timeStamp != null) {
          tsFlag = false;
          end = timeStamp[i][1];
        }
      } else if (isStringAllAlpha(middleLists[i])) {
        wordItem += middleLists[i];
        wordLists.add(wordItem);
        wordLists.add(" ");
        wordItem = "";
        alphaBlank = true;
        if (timeStamp != null) {
          tsFlag = true;
          end = timeStamp[i][1];
          tsLists.add([begin, end]);
          begin = end;
        }
      } else {
        throw Exception("invalid character: ${middleLists[i]}");
      }
    }
  }

  if (timeStamp != null) {
    wordLists = abbrDispose(wordLists, tsLists);
    return wordLists.join("");
  } else {
    wordLists = abbrDispose(wordLists);

    return wordLists.join("");
  }
}


// void main(){
//   print(sentencePostprocess(["你", "好", "hello", "world"]));
// }

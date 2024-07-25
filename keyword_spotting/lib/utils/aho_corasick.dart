Map<String, int> decodeDict = {
  'a': 0,
  'b': 1,
  'c': 2,
  'd': 3,
  'e': 4,
  'f': 5,
  'g': 6,
  'h': 7,
  'i': 8,
  'j': 9,
  'k': 10,
  'l': 11,
  'm': 12,
  'n': 13,
  'o': 14,
  'p': 15,
  'q': 16,
  'r': 17,
  's': 18,
  't': 19,
  'u': 20,
  'v': 21,
  'w': 22,
  'x': 23,
  'y': 24,
  'z': 25,
  '1': 26,
  '2': 27,
  '3': 28,
  '4': 29,
  '5': 30,
};
Map<int, String> encodeDict = {
  0: 'a',
  1: 'b',
  2: 'c',
  3: 'd',
  4: 'e',
  5: 'f',
  6: 'g',
  7: 'h',
  8: 'i',
  9: 'j',
  10: 'k',
  11: 'l',
  12: 'm',
  13: 'n',
  14: 'o',
  15: 'p',
  16: 'q',
  17: 'r',
  18: 's',
  19: 't',
  20: 'u',
  21: 'v',
  22: 'w',
  23: 'x',
  24: 'y',
  25: 'z',
  26: '1',
  27: '2',
  28: '3',
  29: '4',
  30: '5',
};

class AhoCorasick {
  List<String> keywords;
  late int maxStates;
  int maxCharacters = 26 + 5; // 26 为 26个字母，5 为 5 个声调（包括轻声）
  late List<int> out;
  late List<int> fail;
  late List<List<int>> goto;
  late int statesCount;
  RegExp pattern = RegExp(r"\d");

  bool withTone = true;
  AhoCorasick(this.keywords, {bool withTone = true}) {
    withTone = withTone;
    if (!withTone) {
      maxCharacters = 26 + 1; // 1 为分割号
      decodeDict["|"] = 26;
      pattern = RegExp(r"\|");
    }
    maxStates = keywords.map((m) => m.length).reduce((a, b) => a + b);
    out = List.filled(maxStates + 1, 0);
    fail = List.filled(maxStates + 1, -1);
    goto = List.generate(maxStates + 1, (m) => List.filled(maxCharacters, -1));
    statesCount = buildMatchingMachine();
  }

  buildMatchingMachine() {
    int k = keywords.length;
    int states = 1;
    for (var i = 0; i < k; i++) {
      String keyword = keywords[i];
      int currentState = 0;
      for (var j = 0; j < keyword.length; j++) {
        String character = keyword[j];
        int ch = decodeDict[character]!;
        if (goto[currentState][ch] == -1) {
          goto[currentState][ch] = states;
          states += 1;
        }
        currentState = goto[currentState][ch];
      }
      out[currentState] |= (1 << i);
    }
    for (var ch = 0; ch < maxCharacters; ch++) {
      if (goto[0][ch] == -1) {
        goto[0][ch] = 0;
      }
    }
    List<int> queue = List.empty(growable: true);
    for (var ch = 0; ch < maxCharacters; ch++) {
      if (goto[0][ch] != 0) {
        fail[goto[0][ch]] = 0;
        queue.add(goto[0][ch]);
      }
    }

    while (queue.isNotEmpty) {
      int state = queue.removeLast();
      for (var ch = 0; ch < maxCharacters; ch++) {
        if (goto[state][ch] != -1) {
          int failure = fail[state];
          while (goto[failure][ch] == -1) {
            failure = fail[failure];
          }
          failure = goto[failure][ch];
          fail[goto[state][ch]] = failure;

          out[goto[state][ch]] |= out[failure];
          queue.add(goto[state][ch]);
        }
      }
    }
    return states;
  }

  findNextStates(int currentState, String nextInput) {
    int answer = currentState;
    int ch = decodeDict[nextInput]!;
    while (goto[answer][ch] == -1) {
      answer = fail[answer];
    }
    return goto[answer][ch];
  }

  Map<String, List<int>> searchWords(String text) {
    int currentState = 0;
    Map<String, List<int>> result = {};
    for (var i = 0; i < text.length; i++) {
      currentState = findNextStates(currentState, text[i]);
      if (out[currentState] == 0) {
        continue;
      }
      for (var j = 0; j < keywords.length; j++) {
        if (out[currentState] & (1 << j) > 0) {
          String word = keywords[j];
          if (result.containsKey(word)) {
            result[word]?.add(pattern.allMatches(text.substring(0, i - word.length + 1)).length);

            // result[word]?.add(i - word.length + 1);
          } else {
            result[word] = [pattern.allMatches(text.substring(0, i - word.length + 1)).length];

          }
        }
      }
    }
    return result;
  }
}

class AhoCorasickSearcher {
  List<String> words;
  AhoCorasick? ahoCorasick;
  AhoCorasickSearcher(this.words, {bool withTone=true}) {
    ahoCorasick = AhoCorasick(words, withTone: withTone);
  }

  reset(List<String> words) {
    ahoCorasick = null;
    ahoCorasick = AhoCorasick(words);
  }

  Map<String, List<int>>? search(String pinyin) {
    Map<String, List<int>>? result = ahoCorasick?.searchWords(pinyin);
    return result;
  }
}

void main() {
  // List<String> words = ["he", "she", "hers", "his"];
  // String texts = "ahishersheshehis";
  //
  // AhoCorasick ahoCorasick = AhoCorasick(words);
  // Map result = ahoCorasick.searchWords(texts);
  // for (var k in result.keys) {
  //   for (var i in result[k]) {
  //     print("Word: $k, appears from ${i.toString()} to ${i + k.length - 1}");
  //   }
  // }
  var pattern = RegExp(r"\|");
  var text = "wo|wo|se|de|";
  print(pattern.allMatches(text).length);
}

import 'dart:convert';

import 'package:flutter/services.dart';

class Tokenizer {
  late List<dynamic> vocab;

  Future<void> init() async {
    vocab = await loadJsonFromAssets();
  }

  Future<List<dynamic>> loadJsonFromAssets() async {
    String jsonString = await rootBundle.loadString("assets/tokens.json");
    return jsonDecode(jsonString);
  }

  List<String> id2Token(List<int> id){
    List<String> token = List.empty(growable: true);
    for (var e in id) {
      token.add(vocab[e]);
    }
    return token;
  }

}

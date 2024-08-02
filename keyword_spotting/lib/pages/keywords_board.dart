import 'package:flutter/material.dart';


class KeywordsBoard extends StatefulWidget {
  final List<String> keywords;
  final List<String> emotion;
  final List<int> speaker;
  final ScrollController controller;
  const KeywordsBoard({super.key, required this.keywords, required this.emotion, required this.speaker, required this.controller});

  @override
  KeywordsBoardState createState() => KeywordsBoardState();
}

class KeywordsBoardState extends State<KeywordsBoard> {
  @override
  void initState() {
    super.initState();
  }

  getEmotion(String id){
    if (id == "neutral"){    // 中性
      return "😐中性";
    }else if(id == "happy"){   // 喜悦
      return "😊喜悦";
    }else if(id == "sad"){   // 伤心
      return "😢悲伤";
    }
    else if(id == "fear"){   // 恐惧
      return "😣恐惧";
    }
    else if(id == "angry"){
      return "😡生气";
    }
    else{
      return "😶未知";
    }
  }

  List<Widget> getData(){
    List<Widget> list = List<Widget>.empty(growable: true);
    list.add(const ListTile(
      dense: true,
      leading: Text("   情绪   ", style: TextStyle(fontSize: 16),),
      title: Text("  关键词", style: TextStyle(fontSize: 16),),
      // trailing: Text("说话人", style: TextStyle(fontSize: 16),),
    ));
    for(var i = 0; i<widget.keywords.length; i++){
      list.add(ListTile(
        dense: true,
        leading: Text("  ${getEmotion(widget.emotion[i])}  ", style: const TextStyle(fontSize: 14)),
        title: Text("  ${widget.keywords[i]}", style: const TextStyle(fontSize: 14)),
        trailing: Text(widget.speaker[i].toString(), style: const TextStyle(fontSize: 14)),
      ));
    }
    return list;
  }

  @override
  Widget build(BuildContext context) {
      final height = MediaQuery.of(context).size.height;
      return Container(
        decoration: BoxDecoration(
          border: Border.all(

          )
        ),
      height: 0.3 * height,
      child:ListView(
        // shrinkWrap: true,
        controller: widget.controller,
        children: getData(),
      ) ,
    )
    ;
  }
}

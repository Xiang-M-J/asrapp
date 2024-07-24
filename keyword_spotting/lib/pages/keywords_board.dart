import 'package:flutter/material.dart';


class KeywordsBoard extends StatefulWidget {
  List<String> keywords;
  List<String> emotion;
  List<int> times;
  KeywordsBoard({super.key, required this.keywords, required this.emotion, required this.times});

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
      return "😐";
      return const Icon(Icons.add);
    }else if(id == "happy"){   // 喜悦
      return "😊";
      return const Icon(Icons.abc);
    }else if(id == "sad"){   // 伤心
      return "😢";
    }
    else if(id == "fear"){   // 恐惧
      return "😣";
    }
    else if(id == "angry"){
      return "😡";
    }
    else{
      return "😶";
      return const Icon(Icons.update);
    }
  }

  List<Widget> getData(){
    final size = 16;
    List<Widget> list = List<Widget>.empty(growable: true);
    list.add(const ListTile(
      dense: true,
      leading: Text("情绪", style: TextStyle(fontSize: 16),),
      title: Text("关键词", style: TextStyle(fontSize: 16),),
      trailing: Text("次数", style: TextStyle(fontSize: 16),),
    ));
    for(var i = 0; i<widget.keywords.length; i++){
      list.add(ListTile(
        dense: true,
        leading: Text(getEmotion(widget.emotion[i]), style: const TextStyle(fontSize: 16)),
        title: Text(widget.keywords[i], style: const TextStyle(fontSize: 12)),
        trailing: Text(widget.times[i].toString(), style: const TextStyle(fontSize: 16)),
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
        children: getData(),
      ) ,
    )
    ;
  }
}

import 'package:flutter/material.dart';


class ScrollableTextField extends StatefulWidget {
  final TextEditingController controller;
  final String hintText;

  const ScrollableTextField({super.key, required this.controller, required this.hintText});

  @override
  ScrollableTextFieldState createState() => ScrollableTextFieldState();
}

class ScrollableTextFieldState extends State<ScrollableTextField> {
  @override
  void initState() {
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    return
      Container(
        padding: const EdgeInsets.only(left: 5, top: 0, bottom: 0, right: 5),
        height: 180,
        decoration: BoxDecoration(
          border: Border.all(
            color: Colors.blue, // 边框颜色
            width: 2, // 边框宽度
          ),
        ),
        child:  SingleChildScrollView(
          child: TextField(
            controller: widget.controller,
            maxLines: null, // 允许多行输入
            decoration: InputDecoration(
              border: InputBorder.none,
              hintText: widget.hintText,
              hintStyle: const TextStyle(
                color: Colors.grey
              )
            ),
            readOnly: true,
          ),
        ),
      );

      SizedBox(
      height: 180, // 设置固定高度

      child:  SingleChildScrollView(
        child: TextField(
          controller: widget.controller,
          maxLines: null, // 允许多行输入
          decoration: InputDecoration(
            border: const OutlineInputBorder(),
            hintText: widget.hintText,
          ),
          readOnly: true,
        ),
      ),
    );
  }
}

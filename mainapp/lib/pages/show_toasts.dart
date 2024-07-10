import 'package:flutter/material.dart';
import 'package:fluttertoast/fluttertoast.dart';

showRecordingToast(){
 return Fluttertoast.showToast(
      msg: "正在识别，请稍等",
      toastLength: Toast.LENGTH_SHORT,
      gravity: ToastGravity.CENTER,
      timeInSecForIosWeb: 1,
      backgroundColor: Colors.red,
      textColor: Colors.white,
      fontSize: 16.0
  );
}

import 'package:flutter/material.dart';
import 'package:fluttertoast/fluttertoast.dart';

showToastWrapper(String toastText){
 return Fluttertoast.showToast(
      msg: toastText,
      toastLength: Toast.LENGTH_SHORT,
      gravity: ToastGravity.TOP,
      timeInSecForIosWeb: 1,
      backgroundColor: Colors.red,
      textColor: Colors.white,
      fontSize: 16.0
  );
}

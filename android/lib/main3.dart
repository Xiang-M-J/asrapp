import 'dart:async';

import 'package:flutter/material.dart';
import 'package:record/record.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const MyHomePage(title: 'Flutter Demo Home Page'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  bool _isRecording = false;

  final record = AudioRecorder();
  final streamController = StreamController();


  void start() async {
    _isRecording = true;
    setState(() {

    });
    
    if (await record.hasPermission()){
      final stream = await record.startStream(const RecordConfig(encoder: AudioEncoder.pcm16bits));
      await streamController.addStream(stream);
    }
    
  }
  void stop() async {
    _isRecording = false;
    setState(() {

    });
    
    await record.stop();
    await streamController.close();
    List<dynamic> info = await streamController.stream.toList();
    print(info);
  }

  @override
  Widget build(BuildContext context) {
   
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: Text(widget.title),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            TextButton.icon(
              onPressed: _isRecording ? stop : start,
              icon:
                  _isRecording ? const Icon(Icons.stop) : const Icon(Icons.mic),
              label: _isRecording ? const Text("停止录音") : const Text("开始录音"),
            )
          ],
        ),
      ),
    );
  }

  @override
  void initState() {
    super.initState();
    // record.hasPermission();

  }
  
  @override
  void dispose(){
    record.dispose();
    super.dispose();
  }
}

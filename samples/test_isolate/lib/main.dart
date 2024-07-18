import 'dart:async';
import 'dart:isolate';
import 'package:flutter/material.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      home: IsolateExample(),
    );
  }
}

class IsolateExample extends StatefulWidget {
  const IsolateExample({super.key});

  @override
  IsolateExampleState createState() => IsolateExampleState();
}

class IsolateExampleState extends State<IsolateExample> {
  Isolate? _isolate;
  ReceivePort? _receivePort;
  SendPort? _sendPort;
  bool _isRunning = false;
  int _parameter = 0;
  List<int> wave = List.empty(growable: true);

  @override
  void initState() {
    super.initState();
  }

  @override
  void dispose() {
    _stopIsolate();
    super.dispose();
  }

  void _startIsolate() async {
    _receivePort = ReceivePort();
    _isolate = await Isolate.spawn(_isolateEntry, _receivePort!.sendPort);

    Timer.periodic(const Duration(seconds: 1), (t){
      _sendParameterToIsolate(wave);
    });
    _receivePort!.listen((message) {
      if (message is SendPort) {
        _sendPort = message;
      } else {
        setState(() {
          print("Received from isolate: $message");
          print(wave.length);
        });
      }
    });
    setState(() {
      _isRunning = true;
    });
  }

  void _stopIsolate() {
    if (_isolate != null) {
      _isolate!.kill(priority: Isolate.immediate);
      _isolate = null;
      _receivePort!.close();
    }
    setState(() {
      _isRunning = false;
    });
  }

  void _sendParameterToIsolate(List<int> param) {
    _sendPort?.send(param);
  }

  static void _isolateEntry(SendPort mainSendPort) {
    ReceivePort isolateReceivePort = ReceivePort();
    mainSendPort.send(isolateReceivePort.sendPort);

    isolateReceivePort.listen((param) {
        List<int> result = _heavyComputation(param);
        mainSendPort.send(result);
    });
  }

  static List<int> _heavyComputation(List<int> param) {
    param.add(100);
    return param;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Isolate Example'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            _isRunning
                ? ElevatedButton(
              onPressed: _stopIsolate,
              child: const Text('Stop Isolate'),
            )
                : ElevatedButton(
              onPressed: _startIsolate,
              child: const Text('Start Isolate'),
            ),
            ElevatedButton(
              onPressed: () {
                _parameter += 1000000;
                // _sendParameterToIsolate(_parameter);
              },
              child: const Text('Send Parameter to Isolate'),
            ),
          ],
        ),
      ),
    );
  }
}

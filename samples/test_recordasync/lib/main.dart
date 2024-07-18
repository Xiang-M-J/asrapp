import 'dart:async';
import 'dart:isolate';

import 'package:flutter/material.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: Text('Flutter Isolate Example'),
        ),
        body: Center(
          child: IsolateDemo(),
        ),
      ),
    );
  }
}

class IsolateDemo extends StatefulWidget {
  @override
  _IsolateDemoState createState() => _IsolateDemoState();
}

class _IsolateDemoState extends State<IsolateDemo> {
  String _output = "Press the button to start computation";

  Future<void> _startIsolate() async {

    final ReceivePort receivePort = ReceivePort();

    await Isolate.spawn(_isolateTask, receivePort.sendPort);

    final SendPort sendPort = await receivePort.first;

    final response = ReceivePort();
    sendPort.send(['Compute Fibonacci', 45, response.sendPort]);

    final result = await response.first;
    setState(() {
      _output = 'Fibonacci result: $result';
    });
  }

  static void _isolateTask(SendPort initialReplyTo) async {
    final port = ReceivePort();
    initialReplyTo.send(port.sendPort);

    await for (final message in port) {
      final data = message[0] as String;
      final number = message[1] as int;
      final sendPort = message[2] as SendPort;

      if (data == 'Compute Fibonacci') {
        final result = _fibonacci(number);
        sendPort.send(result);
      }
    }
  }

  static int _fibonacci(int n) {
    if (n <= 1) {
      return n;
    }
    return _fibonacci(n - 1) + _fibonacci(n - 2);
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: <Widget>[
        Text(_output),
        const SizedBox(height: 20),
        ElevatedButton(
          onPressed: _startIsolate,
          child: const Text('Start Computation'),
        ),
      ],
    );
  }
}

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
      home: IsolateExample(),
    );
  }
}

class IsolateExample extends StatefulWidget {
  @override
  _IsolateExampleState createState() => _IsolateExampleState();
}

class _IsolateExampleState extends State<IsolateExample> {
  Isolate? _isolate;
  ReceivePort? _receivePort;
  SendPort? _sendPort;
  bool _isRunning = false;
  int _parameter = 0;
  List<int> _tasks = [];
  Timer? _timer;

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
    _receivePort!.listen((message) {
      if (message is SendPort) {
        _sendPort = message;
      } else {
        setState(() {
          print("Received from isolate: $message");
        });
      }
    });

    _timer = Timer.periodic(Duration(seconds: 5), (timer) {
      if (_tasks.isNotEmpty) {
        _sendPort?.send(_tasks.removeAt(0));
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
    _timer?.cancel();
    setState(() {
      _isRunning = false;
    });
  }

  void _sendParameterToIsolate(int param) {
    _tasks.add(param); // 将任务添加到队列
  }

  static void _isolateEntry(SendPort mainSendPort) {
    ReceivePort isolateReceivePort = ReceivePort();
    mainSendPort.send(isolateReceivePort.sendPort);

    isolateReceivePort.listen((param) async {
      if (param is int) {
        await _executeTasksSequentially(param, mainSendPort);
      }
    });
  }

  static Future<void> _executeTasksSequentially(int param, SendPort mainSendPort) async {
    for (int i = 0; i < param; i++) {
      Completer<void> completer = Completer<void>();
      int result = await _heavyComputation(i);
      mainSendPort.send(result);
      completer.complete();
      await completer.future;
    }
  }

  static Future<int> _heavyComputation(int param) async {
    // 模拟耗时工作
    await Future.delayed(Duration(seconds: 3)); // 假设计算需要 3 秒
    return param * DateTime.now().millisecondsSinceEpoch;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Isolate Example'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            _isRunning
                ? ElevatedButton(
              onPressed: _stopIsolate,
              child: Text('Stop Isolate'),
            )
                : ElevatedButton(
              onPressed: _startIsolate,
              child: Text('Start Isolate'),
            ),
            ElevatedButton(
              onPressed: () {
                _parameter += 1;
                _sendParameterToIsolate(_parameter);
              },
              child: Text('Send Parameter to Isolate'),
            ),
          ],
        ),
      ),
    );
  }
}

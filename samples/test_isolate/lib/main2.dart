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
  ReceivePort? _receive1Port;
  ReceivePort? _receive2Port;
  SendPort? _send1Port;
  SendPort? _send2Port;
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
    _receive1Port = ReceivePort();
    _receive2Port = ReceivePort();
    _isolate = await Isolate.spawn(_isolateEntry, {"a": _receive1Port!.sendPort, "b": _receive2Port!.sendPort});
    _receive1Port!.listen((message) {
      if (message is SendPort) {
        _send1Port = message;
      } else {
        setState(() {
          print("Received from isolate: $message");
        });
      }
    });

    _receive2Port!.listen((message){
      if (message is SendPort) {
        _send2Port = message;
      }else{
        setState(() {
          print("Received from isolate: $message");
        });
      }
    });

    _timer = Timer.periodic(const Duration(seconds: 5), (timer) {
      if (_tasks.isNotEmpty) {
        _send1Port?.send(_tasks.removeAt(0));
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
      _receive1Port!.close();
    }
    _timer?.cancel();
    setState(() {
      _isRunning = false;
    });
  }

  void _sendParameterToIsolate(int param) {
    _tasks.add(param); // 将任务添加到队列
  }

  static void _isolateEntry( Map sendPorts) {
    SendPort mainSendPort = sendPorts["a"];
    SendPort anotherPort = sendPorts["b"];
    ReceivePort isolateReceivePort1 = ReceivePort();
    ReceivePort isolateReceivePort2 = ReceivePort();
    mainSendPort.send(isolateReceivePort1.sendPort);
    anotherPort.send(isolateReceivePort2.sendPort);

    isolateReceivePort1.listen((param) async {
      if (param is int) {
        await _executeTasksSequentially(param, mainSendPort);
      }
    });

    isolateReceivePort2.listen((param) async{
      print("from port2: $param");
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
    await Future.delayed(const Duration(seconds: 3)); // 假设计算需要 3 秒
    return param * DateTime.now().millisecondsSinceEpoch;
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
                _parameter += 1;
                _sendParameterToIsolate(_parameter);
              },
              child: const Text('Send Parameter to Isolate'),
            ),
          ],
        ),
      ),
    );
  }
}

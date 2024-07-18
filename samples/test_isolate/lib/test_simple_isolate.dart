// import 'dart:isolate';

// // 主线程中的 SendPort 和 ReceivePort
// SendPort? isolateASendPort;
// SendPort? isolateBSendPort;
// ReceivePort mainReceivePort = ReceivePort();

// void main() async {
//   // 创建两个 Isolate
//   isolateASendPort = await createIsolate(isolateA, );
//   isolateBSendPort = await createIsolate(isolateB);

//   // 主线程中的消息处理
//   mainReceivePort.listen((message) {
//     print('Main received: $message');
//   });

//   // 启动和停止 Isolate 的示例
//   startIsolateA(isolateASendPort!);
//   await Future.delayed(const Duration(seconds: 5));
//   stopIsolateA(isolateASendPort!);
// }

// Future<SendPort> createIsolate(Function isolateFunction) async {
//   ReceivePort receivePort = ReceivePort();
//   await Isolate.spawn(isolateFunction, receivePort.sendPort);
//   return await receivePort.first as SendPort;
// }

// void startIsolateA(SendPort isolateASendPort) {
//   isolateASendPort.send('start');
// }

// void stopIsolateA(SendPort isolateASendPort) {
//   isolateASendPort.send('stop');
// }

// void isolateA(SendPort sendPort) {
//   ReceivePort receivePort = ReceivePort();
//   sendPort.send(receivePort.sendPort);

//   List<int> data = [];
//   int dataLengthThreshold = 10;

//   receivePort.listen((message) {
//     if (message == 'start') {
//       // 生成数据
//       for (int i = 0; i < 20; i++) {
//         data.add(i);
//         if (data.length >= dataLengthThreshold) {
//           // 将数据传递给 isolateB
//           sendPort.send(data);
//           data.clear();
//         }
//       }
//     } else if (message == 'stop') {
//       // 将剩余的数据传递给 isolateB
//       if (data.isNotEmpty) {
//         sendPort.send(data);
//         data.clear();
//       }
//       Isolate.exit();
//     }
//   });
// }

// void isolateB(SendPort sendPort) {
//   ReceivePort receivePort = ReceivePort();
//   sendPort.send(receivePort.sendPort);

//   receivePort.listen((data) {
//     // 处理数据
//     print('IsolateB received: $data');
//     // 自动停止
//     if (data.isEmpty) {
//       Isolate.exit();
//     }
//   });
// }

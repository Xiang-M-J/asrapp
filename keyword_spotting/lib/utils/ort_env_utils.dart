import 'package:onnxruntime/onnxruntime.dart';

void initOrtEnv(){
  OrtEnv.instance.init();
  OrtEnv.instance.availableProviders().forEach((element) {
    print('onnx provider=$element');
  });
}

void releaseOrtEnv(){
  OrtEnv.instance.release();
}
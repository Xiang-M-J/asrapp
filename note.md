[Mobile image recognition on Android | onnxruntime](https://onnxruntime.ai/docs/tutorials/mobile/deploy-android.html)



**flutter runtime**

[onnxruntime | Flutter package (pub.dev)](https://pub.dev/packages/onnxruntime)



**runtime**

[Java | onnxruntime](https://onnxruntime.ai/docs/get-started/with-java.html)

[Mobile | onnxruntime](https://onnxruntime.ai/docs/get-started/with-mobile.html)





使用 flutter onnxruntime 时，如果报错缺少 libonnxruntime.so 文件，需要从 [Maven Repository: com.microsoft.onnxruntime » onnxruntime-android » 1.18.0 (mvnrepository.com)](https://mvnrepository.com/artifact/com.microsoft.onnxruntime/onnxruntime-android/latest) 下载对应的文件，这里需要注意下载文件的入口在表格的 Files 一栏中，下载 aar 文件，下载好后，将 aar 后缀改为 zip 后缀，解压后可以在 jni 文件夹找到 so 文件，将其复制到对应的文件夹中 build/app/intermediates/merged_native_libs 和 build/app/intermediates/stripped_native_libs 这两个文件夹。


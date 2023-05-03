/*
 * Copyright 2023 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:imageclassification/classifier.dart';
import 'package:integration_test/integration_test.dart';
//import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';
import 'package:tflite_flutter/tflite_flutter.dart'; // as tfl;

import 'classifier_test_helper.dart';

const sampleFileName = 'assets/lion.jpg';
const labelFileName = 'assets/labels.txt';

const model_float = 'mobilenet_v1_1.0_224.tflite';
const model_quant = 'mobilenet_v1_1.0_224_quant.tflite';

//flutter driver --driver=test_driver/integration_test.dart --target=integration_test/image_classification_test.dart
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('inference', () {
    late img.Image testImage;

    setUp(() async {
      ByteData imageFile = await rootBundle.load(sampleFileName);
      testImage = img.decodeImage(imageFile.buffer.asUint8List())!;
    });

    group('float', () {
      late Classifier classifier;

      setUpAll(() {
        classifier = ClassifierFloatTest();
      });

      test('run', () {
        Category prediction = classifier.predict(testImage);
        expect(prediction.label, "lion");
      });

      tearDownAll(() {
        classifier.close();
      });
    });
    group('quant', () {
      late ClassiferTest classifier;

      setUpAll(() {
        classifier = ClassifierQuantTest();
      });

      test('run', () {
        Category prediction = classifier.predict(testImage);
        expect(prediction.label, "lion");
      });

      tearDownAll(() {
        classifier.close();
      });
    });
  });
}

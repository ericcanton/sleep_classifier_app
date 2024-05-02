import 'package:flutter/material.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      title: 'Sleep Wake Classifier',
      home: SleepClassifier(),
    );
  }
}

class SleepClassifier extends StatefulWidget {
  const SleepClassifier({super.key});

  @override
  State<SleepClassifier> createState() => _SleepClassifierState();
}

class _SleepClassifierState extends State<SleepClassifier> {
  late Interpreter _interpreter;
  List<dynamic> _output = []; // Result placeholders

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  @override
  void dispose() {
    _interpreter.close(); // Clean up when the widget is disposed
    super.dispose();
  }

  // Load the TF Lite model
  _loadModel() async {
    _interpreter =
        await Interpreter.fromAsset('assets/sleep_wake_classifier.tflite');
    print('Model loaded successfully');
  }

  // Placeholder for making predictions (will need to update with your model input/output)
  _makePrediction(input) async {
    // Prepare input data (adjust according to your model's requirements)

    // Run inference
    _interpreter.run(input, _output);

    setState(() {
      // Update UI with results
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        appBar: AppBar(
          title: const Text('Sleep Wake Classifier'),
        ),
        body: Center(
            // Design your UI, including buttons to trigger predictions
            // and display the results from the _outputs variable.
            ));
  }
}

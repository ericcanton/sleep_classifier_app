import 'dart:io';

import 'package:flutter/material.dart';
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
import 'package:file_picker/file_picker.dart';

const String modelFile = 'assets/best-mo-walch-no-4018081.tflite';
const int inputWidth = 15360;
const int inputHeight = 32;

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
  late tfl.Interpreter _interpreter;
  List<dynamic> _output = []; // Result placeholders
  List<List<double>> _csvData = [];

  // ... (Your existing functions)

  // Function to pick a CSV file
  Future<void> _pickCSV() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['csv', 'out', 'txt'],
    );

    if (result != null) {
      String filePath = result.files.single.path!;
      _loadCSV(filePath);
    }
  }

  // Function to load CSV data
  void _loadCSV(String filePath) async {
    final csvFile = await File(filePath).readAsString();

    List<List<double>> csvData = csvFile.split('\n').map((String row) {
      try {
        return row.split(',').map(double.parse).toList();
      } catch (e) {
        return <double>[];
      }
    }).toList();

    setState(() {
      _csvData = csvData;
    });
  }

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
    _interpreter = await tfl.Interpreter.fromAsset(modelFile);
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
            child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton(
              onPressed: _pickCSV,
              child: const Text('Select CSV File'),
            ),
            // ... (Display or use the _csvData)
          ],
        )));
  }
}

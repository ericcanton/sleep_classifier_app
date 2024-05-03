import 'dart:io';

import 'package:flutter/material.dart';
import 'package:syncfusion_flutter_charts/charts.dart';
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
import 'package:file_picker/file_picker.dart';

const String modelFile = 'assets/best-mo-walch-no-4018081.tflite';
const int inputWidth = 15360;
const int inputHeight = 32;
const int nEpochs = 1024;
const int nClasses = 4;

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
  List<List<double>> _csvData = [];
  final List<List<List<double>>> _output = [
    List.generate(nEpochs, (index) => List.generate(nClasses, (index) => 0.0))
  ]; // Result placeholders
  // final List<List<List<List<double>>>>  _modelInput = List.generate(
  final List<List<List<List<double>>>> _modelInput = [
    List.generate(inputWidth,
        (index) => List.generate(inputHeight, (index) => [0.0, 0.0]))
  ];

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

    _prepareInputData();

    setState(() {
      _csvData = csvData;
    });
  }

  /// Prepare input data
  /// Read CSV of shape (n, 32) and return a List of shape (n, 32, 2)
  /// where n is the number of rows in the CSV
  ///
  /// We "reflect" the 2D data to 3D by
  /// repeating the rows, reversed
  _prepareInputData() {
    // copy the data into modelInput
    for (int i = 0; i < inputWidth; i++) {
      for (int j = 0; j < inputHeight; j++) {
        if (i >= _csvData.length || j >= _csvData[i].length) {
          continue;
        }
        // only one layer, always [0]
        _modelInput[0][i][j][0] = _csvData[i][j];
        // "reflect" the array across the y-axis, as in: _csvData[x][y]
        _modelInput[0][i][j][1] = _csvData[i][_csvData[i].length - j - 1];
      }
    }
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

  _predictFromCSVData() async {
    await _makePrediction(_modelInput);
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
            ElevatedButton(
              onPressed: _predictFromCSVData,
              child: const Text('Make prediction'),
            ),
            _csvDataHeatMap(),
            _hypnogram()
            // ... (Display or use the _csvData)
          ],
        )));
  }

  Widget _csvDataHeatMap() {
    return _csvData.isNotEmpty
        // ? Expanded(child: HeatmapWidget(csvData: _csvData))
        ? Text('CSV data loaded with ${_csvData.length} rows')
        : const Text('No data to display');
  }

  Widget _hypnogram() {
    print("hypnogram");
    return _output.isNotEmpty
        ? Expanded(child: HypnogramWidget(stageProbabilities: _output[0]))
        : const Text('No hypnogram to display');
  }
}

class HeatMapPainter extends CustomPainter {
  final List<List<double>> _csvData;

  HeatMapPainter(this._csvData);

  /// Paint the heatmap
  /// The heatmap is a grid of rectangles, where each rectangle represents a value
  /// The color of the rectangle is determined by the value
  /// The color scale can be adjusted as needed
  @override
  void paint(Canvas canvas, Size size) {
    // Define the color scale
    final absMax = _csvData
        .expand((row) => row)
        .reduce((a, b) => a.abs() > b.abs() ? a : b)
        .abs();

    final absMin = _csvData
        .expand((row) => row)
        .reduce((a, b) => a.abs() < b.abs() ? a : b)
        .abs();
    final colorScale = 255 / (absMax - absMin);

    // Define the size of each rectangle
    final rectWidth = size.width / _csvData[0].length;
    final rectHeight = size.height / _csvData.length;

    // Paint the heatmap
    for (int i = 0; i < _csvData.length; i++) {
      for (int j = 0; j < _csvData[i].length; j++) {
        final value = _csvData[i][j] - absMin;
        final color = Color.fromARGB(
          255,
          (value * colorScale).toInt(),
          (value * colorScale).toInt(),
          (value * colorScale).toInt(),
        );

        final rect =
            Rect.fromLTWH(j * rectWidth, i * rectHeight, rectWidth, rectHeight);
        final paint = Paint()..color = color;
        canvas.drawRect(rect, paint);
      }
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}

int argmax<X extends Comparable>(List<X> list) {
  return list.indexWhere((element) =>
      element == list.reduce((a, b) => a.compareTo(b) >= 0 ? a : b));
}

class HypnogramWidget extends StatelessWidget {
  final List<double> psgData = []; // Your CSV data

  HypnogramWidget({Key? key, required List<List<double>> stageProbabilities})
      : super(key: key) {
    // take argmax to convert probabilities to maximum likelihood class
    psgData.addAll(stageProbabilities.map((e) => argmax(e).toDouble()));
  }

  @override
  Widget build(BuildContext context) {
    return Container(
        height: 30, // specify the height
        width: 300, // specify the width
        child: SfCartesianChart(
          primaryXAxis: NumericAxis(
              isVisible: false, minimum: 0, maximum: nEpochs.toDouble()),
          primaryYAxis: NumericAxis(
            isVisible: false,
            minimum: -1.5,
            maximum: 4.5,
          ),
          series: <CartesianSeries>[
            LineSeries(
              dataSource: _psgData
                  .asMap()
                  .entries
                  .map((e) => ChartData(e.key, e.value))
                  .toList(),
              xValueMapper: (data, _) => data.x,
              yValueMapper: (data, _) => data.y,
            ),
          ],
        ));
  }
}

class ChartData {
  final int x;
  final double y;

  ChartData(this.x, this.y);
}

class HeatmapWidget extends StatefulWidget {
  final List<List<double>> _csvData = []; // Your CSV data

  HeatmapWidget({super.key, required List<List<double>> csvData}) : super() {
    _csvData.addAll(csvData);
  }

  @override
  State<StatefulWidget> createState() => _HeatmapWidgetState();
}

class _HeatmapWidgetState extends State<HeatmapWidget> {
  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: HeatMapPainter(widget._csvData),
    );
  }
}

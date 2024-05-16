import 'dart:collection';

class TensorLike<X> extends ListBase {
  final List<TensorLike<X>> _subTensors = [];
  final List<X> _list = [];

  bool get _is1D => _subTensors.isEmpty;

  TensorLike.fromList(List<X> list) {
    _list.addAll(list);
  }

  TensorLike.fromList2D(List<List<X>> list) {
    for (final sublist in list) {
      _subTensors.add(TensorLike.fromList(sublist));
    }
  }

  @override
  int get length => _list.length;

  List<int> get shape =>
      _subTensors.isEmpty ? [length] : [length, ..._subTensors.first.shape];

  @override
  set length(int newLength) {
    _list.length = newLength;
  }

  @override
  operator [](int index) => _list[index];

  @override
  operator []=(int index, value) {
    // if (value is List) {
    //   _list[index] = TensorLike.fromList(value);
    // } else {
    _list[index] = value;
    // }
  }

  TensorLike<X> slice(int start, int end) {
    return TensorLike.fromList(_list.sublist(start, end));
  }

  TensorLike<X> reshape(List<int> newShape) {
    if (newShape.reduce((a, b) => a * b) != length) {
      throw ArgumentError('New shape must have the same number of elements');
    }
    if (newShape.length == 1) {
      return TensorLike.fromList(_list);
    }
    final reshapedTensor = TensorLike<X>.fromList2D([]);
    final subLength = newShape.first;
    for (var i = 0; i < length; i += subLength) {
      reshapedTensor.add(slice(i, i + subLength));
    }
    return reshapedTensor;
  }

  TensorLike<X> flatten() {
    if (_is1D) {
      return this;
    }
    final flattenedTensor = TensorLike<X>.fromList([]);
    for (final subTensor in _subTensors) {
      flattenedTensor.addAll(subTensor.flatten());
    }
    return flattenedTensor;
  }
}

const medianPython = """"
    if isinstance(x, pl.DataFrame):
        x = x.to_numpy()

    window = (
        (fs * window_size + 1) if (fs * window_size) % 2 == 0 else (fs * window_size)
    )

    reduce_dims = False
    if len(x.shape) == 1:
        x = np.expand_dims(x, axis=-1)
        reduce_dims = True
    x_norm = np.zeros((x.shape))

    for idx in range(x.shape[-1]):

        x_med = np.ones((x.shape[0])) * np.median(x[:, idx])

        x_pd = pd.Series(x[:, idx])
        med_ = x_pd.rolling(window).median()
        x_med[int(window / 2) : -int(window / 2)] = med_[window - 1 :]
        x_med[: int(window / 2)] = med_[window - 1]
        x_med[-int(window / 2) :] = med_[-1:]

        x_med[np.isnan(x_med)] = 0  # remove nan

        x_norm[:, idx] = x[:, idx] - x_med

    if reduce_dims:
        x_norm = x_norm[:, 0]
    return x_norm
""";
List<double> median(List<double> list) {
  list.sort();
  if (list.length % 2 == 1) {
    return [list[list.length ~/ 2]];
  } else {
    return [
      list[list.length ~/ 2 - 1],
      list[list.length ~/ 2],
    ];
  }
}

List<double> rollingMedian(List<double> list, int fs, int windowSize) {
  List<double> xMed = List<double>.filled(list.length, 0);
  List<double> xNorm = List<double>.filled(list.length, 0);
  List<double> xPd = List<double>.filled(list.length, 0);
  List<double> med = List<double>.filled(list.length, 0);
  int window =
      (fs * windowSize + 1) % 2 == 0 ? fs * windowSize : fs * windowSize;
  for (int idx = 0; idx < list.length; idx++) {
    xMed[idx] = median(list)[0];
    xPd[idx] = list[idx];
    for (int i = 0; i < list.length; i++) {
      med[i] = xPd.sublist(i, i + window).reduce((a, b) => a + b) / window;
    }
    xMed
        .sublist(window ~/ 2, list.length - window ~/ 2)
        .setAll(window - 1, med.sublist(window - 1));
    xMed.sublist(0, window ~/ 2).setAll(window - 1, [med[window - 1]]);
    xMed
        .sublist(list.length - window ~/ 2)
        .setAll(window - 1, [med[window - 1]]);
    for (int i = 0; i < list.length; i++) {
      if (xMed[i].isNaN) {
        xMed[i] = 0;
      }
    }
    xNorm[idx] = list[idx] - xMed[idx];
  }
  return xNorm;
}

const pythonIQR = """
def iqr_normalization_adaptive(x, fs, median_window, iqr_window):
    def normalize(x, fs, median_window, iqr_window, iqr_upper=0.75, iqr_lower=0.25):

        # add noise
        x_ = x + np.random.normal(loc=0, scale=sys.float_info.epsilon, size=(x.shape))

        # fix window parameters to odd number
        med_window = (
            (fs * median_window + 1)
            if (fs * median_window) % 2 == 0
            else (fs * median_window)
        )
        iqr_window = (
            (fs * iqr_window + 1) if (fs * iqr_window) % 2 == 0 else (fs * iqr_window)
        )

        # preallocation
        x_med = np.ones((x.shape)) * np.median(x_)
        x_iqr_up = np.ones((x.shape)) * np.quantile(x_, iqr_upper)
        x_iqr_lo = np.ones((x.shape)) * np.quantile(x_, iqr_lower)

        # find rolling median
        x_pd = pd.Series(x_)
        med_ = x_pd.rolling(med_window).median()
        x_med[int(med_window / 2) : -int(med_window / 2)] = med_[med_window - 1 :]
        x_med[np.isnan(x_med)] = 0  # remove nan

        # find rolling quantiles
        x_iqr_upper = x_pd.rolling(iqr_window).quantile(iqr_upper)
        x_iqr_lower = x_pd.rolling(iqr_window).quantile(iqr_lower)

        # border padding
        x_iqr_up[int(iqr_window / 2) : -int(iqr_window / 2)] = x_iqr_upper[
            iqr_window - 1 :
        ]
        x_iqr_lo[int(iqr_window / 2) : -int(iqr_window / 2)] = x_iqr_lower[
            iqr_window - 1 :
        ]

        # remove nan
        x_iqr_up[np.isnan(x_iqr_up)] = 0
        x_iqr_lo[np.isnan(x_iqr_lo)] = 0

        # return normalize
        return (x_ - x_iqr_lo) / (x_iqr_up - x_iqr_lo + sys.float_info.epsilon) * 2 - 1

    x_norm = np.zeros((x.shape))
    if len(x.shape) == 1:
        x_norm[:] = normalize(x, fs, median_window, iqr_window)
    else:
        for n in range(x.shape[1]):
            x_norm[:, n] = normalize(x[:, n], fs, median_window, iqr_window)
    return x_norm
    """;

List<double> normalize(
    List<double> x, int fs, int medianWindow, int iqrWindow) {
  List<double> xMed = List<double>.filled(x.length, 0);
  List<double> xIqrUp = List<double>.filled(x.length, 0);
  List<double> xIqrLo = List<double>.filled(x.length, 0);
  List<double> xPd = List<double>.filled(x.length, 0);
  List<double> med = List<double>.filled(x.length, 0);
  List<double> xIqrUpper = List<double>.filled(x.length, 0);
  List<double> xIqrLower = List<double>.filled(x.length, 0);
  int medWindow =
      (fs * medianWindow + 1) % 2 == 0 ? fs * medianWindow : fs * medianWindow;
  iqrWindow = (fs * iqrWindow + 1) % 2 == 0 ? fs * iqrWindow : fs * iqrWindow;
  for (int idx = 0; idx < x.length; idx++) {
    xMed[idx] = median(x)[0];
    xPd[idx] = x[idx];
    for (int i = 0; i < x.length; i++) {
      med[i] =
          xPd.sublist(i, i + medWindow).reduce((a, b) => a + b) / medWindow;
    }
    xMed
        .sublist(medWindow ~/ 2, x.length - medWindow ~/ 2)
        .setAll(medWindow - 1, med.sublist(medWindow - 1));
    xMed.sublist(0, medWindow ~/ 2).setAll(medWindow - 1, [med[medWindow - 1]]);
    xMed
        .sublist(x.length - medWindow ~/ 2)
        .setAll(medWindow - 1, [med[medWindow - 1]]);
    for (int i = 0; i < x.length; i++) {
      if (xMed[i].isNaN) {
        xMed[i] = 0;
      }
    }
    xIqrUp[idx] =
        xPd.sublist(idx, idx + iqrWindow).reduce((a, b) => a + b) / iqrWindow;
    xIqrLo[idx] =
        xPd.sublist(idx, idx + iqrWindow).reduce((a, b) => a + b) / iqrWindow;
    xIqrUpper[idx] =
        xPd.sublist(idx, idx + iqrWindow).reduce((a, b) => a + b) / iqrWindow;
    xIqrLower[idx] =
        xPd.sublist(idx, idx + iqrWindow).reduce((a, b) => a + b) / iqrWindow;
    xIqrUp
        .sublist(iqrWindow ~/ 2, x.length - iqrWindow ~/ 2)
        .setAll(iqrWindow - 1, xIqrUpper.sublist(iqrWindow - 1));
    xIqrLo
        .sublist(iqrWindow ~/ 2, x.length - iqrWindow ~/ 2)
        .setAll(iqrWindow - 1, xIqrLower.sublist(iqrWindow - 1));
    for (int i = 0; i < x.length; i++) {
      if (xIqrUp[i].isNaN) {
        xIqrUp[i] = 0;
      }
      if (xIqrLo[i].isNaN) {
        xIqrLo[i] = 0;
      }
    }
    xMed[idx] = (xPd[idx] - xIqrLo[idx]) / (xIqrUp[idx] - xIqrLo[idx]);
  }

  return xMed;
}

// List<double> iqrNormalizationAdaptive(
//     List<double> x, int fs, int medianWindow, int iqrWindow) {
//   List<List<double>> xNorm =
//       List.generate(x.length, (_) => List.filled(x[0].length, 0));
//   if (x[0].length == 1) {
//     xNorm = normalize(x, fs, medianWindow, iqrWindow);
//   } else {
//     for (int n = 0; n < x[0].length; n++) {
//       for (int i = 0; i < x.length; i++) {
//         xNorm[i][n] = normalize([x[i][n]], fs, medianWindow, iqrWindow)[0];
//       }
//     }
//   }
//   return xNorm;
// }

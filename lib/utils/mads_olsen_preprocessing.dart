import './nd_list.dart';

/// Provides a Dart implementation of the following function from mads Olsen.
/// def median(x, fs, window_size):
///    if isinstance(x, pl.DataFrame):
///        x = x.to_numpy()
///
///    window = (
///        (fs * window_size + 1) if (fs * window_size) % 2 == 0 else (fs * window_size)
///    )
///
///    reduce_dims = False
///    if len(x.shape) == 1:
///        x = np.expand_dims(x, axis=-1)
///        reduce_dims = True
///    x_norm = np.zeros((x.shape))
///
///    for idx in range(x.shape[-1]):
///
///        x_med = np.ones((x.shape[0])) * np.median(x[:, idx])
///
///        x_pd = pd.Series(x[:, idx])
///        med_ = x_pd.rolling(window).median()
///        x_med[int(window / 2) : -int(window / 2)] = med_[window - 1 :]
///        x_med[: int(window / 2)] = med_[window - 1]
///        x_med[-int(window / 2) :] = med_[-1:]
///
///        x_med[np.isnan(x_med)] = 0  # remove nan
///
///        x_norm[:, idx] = x[:, idx] - x_med
///
///    if reduce_dims:
///        x_norm = x_norm[:, 0]
///    return x_norm

// NDList<double> sliding_median(
//     NDList<double> triaxialAccel, int fs, int windowSize) {
//   // directly taken from MO:
//   // (fs * windowSize + 1) % 2 == 0 ? fs * windowSize + 1 : fs * windowSize;
//   final window = fs * windowSize + 1 - ((fs * windowSize + 1) % 2);
//   final reduceDims = triaxialAccel.shape.length == 1;
//   if (reduceDims) {
//     triaxialAccel = NDList.from<double>([triaxialAccel]);
//   }
//   final xNorm = NumNDList.zerosLike<double>(triaxialAccel);
  // for (var idx = 0; idx < triaxialAccel.shape.last; idx++) {
  //   final xMed = NDList<double>.filled(
  //       [triaxialAccel.shape.first], triaxialAccel[idx].median());
  //   final xPd = triaxialAccel[idx].toSeries();
  //   final med = xPd.rolling(window).median();
  //   xMed['${window ~/ 2}:${-window ~/ 2}'] = med['window - 1 :'];
  //   xMed.sublist(0, window ~/ 2).fill(med[window - 1]);
  //   xMed.sublist(-window ~/ 2).fill(med.last);
  //   xMed.where((element) => element.isNaN).forEach((element) => element = 0);
  //   xNorm[idx] = triaxialAccel[idx] - xMed;
  // }
//   return reduceDims ? xNorm[0] : xNorm;
// }

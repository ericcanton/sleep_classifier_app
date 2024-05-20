int getLinearIndex(List<int> shape, List<int> index) {
  if (shape.length != index.length) {
    throw ArgumentError('Shape and index must have the same length');
  }

  int linearIndex = 0;
  int size = 1;

  for (int i = shape.length - 1; i >= 0; i--) {
    linearIndex += index[i] * size;
    size *= shape[i];
  }

  return linearIndex;
}

/// Wrapper on multi-dimensional lists to provide easier indexing and slicing.
/// This class is inspired by NumPy's ndarray.
///
/// Example:
/// ```dart
/// final data = [
///  [1.0, 2.0, 3.0],
/// [4.0, 5.0, 6.0],
/// [7.0, 8.0, 9.0]
/// ];
/// final ndList = NDList.from<double>(data);
///
/// final sliced = ndList[['1:3', '0:2']];
/// ```
/// That is, you can use Python-like slice syntax to access elements.
/// In the end, `sliced` would represent `[[4.0, 5.0], [7.0, 8.0]]`.
class NDList<X> {
  final List<X> _list = [];
  final List<int> _shape = [];

  @override
  String toString() {
    // pretty print the array
    if (_shape.isEmpty) {
      return '[]';
    }
    if (_shape.length == 1) {
      return _list.toString();
    }
    if (_shape.length == 2) {
      return _list
          .map((e) => e.toString())
          .join('\n')
          .replaceAll('[', '')
          .replaceAll(']', '');
    }
    // now we have a (3+)D array, so we need to print each 2D slice
    final slices = List.generate(_shape[0], (i) => this[i]);
    return slices.map((e) => e.toString()).join('\n\n');
  }

  NDList.empty() {
    _shape.add(0);
  }

  NDList.filled(List<int> shape, X fill) {
    _list.addAll(List<X>.filled(NDList._product(shape), fill));
    _shape.addAll(shape);
  }

  static NDList<X> stacked<X>(List<NDList<X>> ndLists, {int axis = 0}) {
    if (axis != 0) {
      throw UnimplementedError('Only axis 0 is supported at the moment');
    }
    if (ndLists.isEmpty) {
      return NDList.empty();
    }

    if (ndLists
        .skip(1)
        .any((element) => !ndLists.first[0]._shapeMatches(element[0]))) {
      throw ArgumentError(
          'All NDLists passed must have matching shapes except for $axis');
    }

    final shape = [ndLists.length, ...ndLists.first._shape];

    return NDList._(ndLists.expand((element) => element._list).toList(), shape);
  }

  static NDList<E> from<E>(List multiList) {
    if (multiList.isEmpty) {
      return NDList.empty();
    }
    var shape = [multiList.length];
    var list = multiList;
    while (list[0] is List) {
      final raggedElementIndex =
          list.indexWhere((element) => element.length != list[0].length);
      if (raggedElementIndex != -1) {
        throw ArgumentError(
            'Ragged array detected! First ragged index: $raggedElementIndex, which has ${list[raggedElementIndex].length} elements, but the 0th element has ${list[0].length}');
      }
      // shape.insert(0, list[0].length);
      shape.add(list[0].length);
      list = list.expand((element) => element).toList();
    }

    if (_product(shape) != list.length) {
      throw ArgumentError('Ragged array detected!.');
    }

    final typedList = list.whereType<E>().toList();

    if (_product(shape) == typedList.length) {
      return NDList._(typedList, shape);
    } else {
      throw ArgumentError('Invalid list');
    }
  }

  NDList._(List<X> list, List<int> shape, {X? fill}) {
    if (_product(shape) != list.length && fill == null) {
      throw ArgumentError('Shape does not match the length of the list');
    }
    _list.addAll(list);
    if (fill != null && _product(shape) > list.length) {
      _list.addAll(List<X>.filled(_product(shape) - list.length, fill));
    }
    _shape.addAll(shape);
  }

  static int _product(List<int> list) {
    return list.fold(1, (a, b) => a * b);
  }

  /// By default, if nd ~ [1, 2] and we call nd[0], it actually returns a wrapped list, [1].
  /// To get 1 itself, call .item (and check this is not null)
  X? get item => _list.length == 1 ? _list[0] : null;

  int get count => _list.length;

  List<int> get shape => _shape;

  /// This method checks if the shapes are equal element-wise.
  ///
  /// Checking `shape == other.shape` does not work because lists are not equal based on element-wise comparison.
  bool _shapeMatches(NDList other) {
    if (shape.length != other.shape.length) {
      return false;
    }
    for (var i = 0; i < shape.length; i++) {
      if (shape[i] != other.shape[i]) {
        return false;
      }
    }
    return true;
  }

  NDList<Y> map<Y>(Y Function(X) f) {
    return NDList._(_list.map(f).toList(), _shape);
  }

  (int, int)? _parseSlice(String slice) {
    try {
      if (slice.isEmpty) {
        return (0, 0);
      }
      // ':' => parts == ['', ''] => start = 0, end = _shape[0]
      // '1:' => parts == ['1', ''] => start = 1, end = _shape[0]
      // ':2' => parts == ['', '2'] => start = 0, end = 2
      final parts = slice.split(':');
      final start = parts[0].isEmpty ? 0 : int.parse(parts[0]);
      final end = parts[1].isEmpty ? _shape[0] : int.parse(parts[1]);
      return (start, end);
    } catch (e) {
      return null;
    }
  }

  NDList<X> operator [](index) {
    if (_list.isEmpty) {
      throw ArgumentError('Empty NDList, cannot index.');
    }
    if (index is List) {
      return _listIndex(index);
    } else if (index is int) {
      return _intIndex(index);
    } else if (index is String) {
      // wrap with [] to make it a list, see note in _listIndex doctsring
      return _listIndex([index]);
    } else {
      throw ArgumentError('Invalid index');
    }
  }

  /// This method is used to index the NDList with a list of valid indices, i.e. ints and formatted slice strings.
  NDList<X> _listIndex(List index) {
    if (index.length == 1 && index[0] is int) {
      return this._intIndex(index[0]);
    } else if (index.length == 1 && index[0] is String) {
      final parsed = _parseSlice(index[0]);
      if (parsed == null) {
        throw ArgumentError('Invalid slice');
      }
      return this.slice(parsed.$1, parsed.$2, axis: 0);
    }
    var sliced = this;
    for (var i = 0; i < index.length; i++) {
      if (index[i] is String) {
        List<String> csSlices = index[i]
            .split(',')
            .map((s) => s.trim())
            .whereType<String>()
            .toList();
        if (csSlices.length > 1) {
          return sliced._listIndex([...csSlices, ...index.sublist(i + 1)]);
        }
        try {
          // is it just an int in string format?
          // .parse throws if cannot be parsed as an int
          return this._intIndex(int.parse(index[i]));
        } catch (e) {
          // just move on, it's not an int
        }

        final parsed = _parseSlice(index[i]);
        if (parsed == null) {
          throw ArgumentError('Invalid slice');
        }
        sliced = sliced.slice(parsed.$1, parsed.$2, axis: i);
      } else if (index[i] is int) {
        sliced = sliced[index[i]];
      } else {
        throw ArgumentError(
            'Invalid index, "${index[i]}" in position $i is not an int or a string.');
      }
    }
    return sliced;
  }

  NDList<X> _intIndex(int index) {
    // return the appropriate axis-0 slice
    if (_shape.length == 1) {
      return NDList._([_list[index]], [1]);
    }
    final returnShape = _shape.sublist(1);
    final subLength = _product(returnShape);
    if (index < 0) {
      // -1 => _shape[0] - 1 (aka last element)
      // -2 => second last element, etc.
      index += _shape[0];
    }
    final theSlice = NDList._(
        _list.sublist(index * subLength, (index + 1) * subLength), returnShape);
    return theSlice;
  }

  operator []=(index, X value) {
    throw UnimplementedError();
  }

  NDList<X> slice(int start, int end, {int axis = 0}) {
    if (start < 0) {
      start += _shape[axis];
    }
    if (end < 0) {
      end += _shape[axis];
    }
    if (end < start) {
      return this.slice(end, start, axis: axis);
    }
    if (end == start) {
      return NDList._([], [0]);
    }
    if (_shape.length == 1) {
      return NDList._(_list.sublist(start, end), [end - start]);
    }
    if (end > _shape[axis]) {
      throw ArgumentError(
          'End index $end is greater than the length (${_shape[axis]}) of the axis $axis');
    }

    if (axis > _shape.length - 1) {
      throw ArgumentError(
          'Invalid axis $axis for ${_shape.length}D list with shape $_shape');
    }

    if (axis == 0) {
      // we know (_shape.length >= 2) since checked == 1 above
      return NDList._([
        ..._list.sublist(start * _product(_shape.sublist(1)),
            end * _product(_shape.sublist(1)))
      ], [
        end - start,
        ..._shape.sublist(1)
      ]);
    }

    // now, build a NDList<NDList<X>>, where each element has the same shape
    // Then we will use .cement() to get a NDList<X> with the new shape

    // return shape becomes the same as the current shape, but with specified axis length being end - start
    // final shapeAfterAxis = _shape.sublist(axis + 1);
    // final sliceLength = end - start;
    // final returnShape = [...shapeBeforeAxis, sliceLength, ...shapeAfterAxis];

    // then it's very easy to NDList.from<NDList<X>>(list).cement()

    final compoundIndices = [
      for (var i = 0; i < _shape[0]; i++) [i]
    ];
    for (int sliceIndex = 1; sliceIndex < axis; sliceIndex++) {
      for (int i = 0; i < _shape[sliceIndex]; i++) {
        for (var j = 0; j < compoundIndices.length; j++) {
          compoundIndices[j].add(i);
        }
      }
    }

    final subTensors = compoundIndices
        .map((compoundIndex) => this[compoundIndex].slice(start, end, axis: 0))
        .toList();

    final shapeBeforeAxis = _shape.sublist(0, axis);
    return NDList.from<NDList<X>>(subTensors)
        .reshape(shapeBeforeAxis)
        .cemented();
  }

  List<int> _sliceIndices((int, int) sliceStartEndPlus1, int axis) {
    var start = sliceStartEndPlus1.$1;
    var end = sliceStartEndPlus1.$2;
    if (start < 0) {
      start += _shape[axis];
    }
    if (end < 0) {
      end += _shape[axis];
    }
    if (end < start) {
      return _sliceIndices((end, start), axis);
    }
    return [];
  }

  NDList<X> reshape(List<int> newShape) {
    if (newShape.where((element) => element == 0).isNotEmpty) {
      if (_list.isEmpty) return NDList._([], newShape);
      throw ArgumentError('New shape cannot have a dimension of 0');
    }
    final positiveDims = newShape.where((element) => element < 1).toList();
    if (positiveDims.length > 1) {
      throw ArgumentError('Only one dimension can be -1');
    }
    final nSpecified = _product(positiveDims);
    if (count % nSpecified != 0) {
      throw ArgumentError('New shape must have the same number of elements');
    }

    final otherAxis = count ~/ nSpecified;

    return NDList._(_list, newShape.map((e) => e < 1 ? otherAxis : e).toList());
  }

  NDList<X> flatten() {
    return NDList._(_list, [count]);
  }

  @override
  bool operator ==(Object other) {
    if (other is NDList) {
      // check if the shape and elements match
      if (!_shapeMatches(other)) {
        return false;
      }
      for (var i = 0; i < count; i++) {
        if (this._list[i] != other._list[i]) {
          return false;
        }
      }
      return true;
    }
    return false;
  }

  @override
  int get hashCode => _list.hashCode ^ _shape.hashCode;
}

/// Provides a number of useful extensions in the typical use case of NDList with numbers. This includes methods like `zeros`, `ones`, and element-wise operations.
///
/// In Dart, the `num` abstract class unifies `int` and `double`, so we work with each separately.
extension NumNDList on NDList {
  /// At time of writing, `X extends num` means that `X` is either an `int` or a `double`. Thus, we can just check if `X` is an `int` and return `0` or `0.0` accordingly.
  static X zero<X extends num>() {
    return (X is int) ? 0 as X : 0.0 as X;
  }

  /// Returns appropriate 1 for X's type. See docstring on `.zero<X>()` in this extension.
  static X one<X extends num>() {
    return (X is int) ? 1 as X : 1.0 as X;
  }

  /// Creates a new NDList with the provided shape and filled with zeros of the specified type.
  ///
  /// Thus, if you want a `NDList<int>` of shape `[2, 3]` filled with the integer `0`, you would call `NumNDList.zeros<int>([2, 3])`.
  ///
  /// If we call with `NumNDList.zeros<double>([2, 3])`, we would get a `NDList<double>` filled with `0.0` instead.
  static NDList<X> zeros<X extends num>(List<int> shape) {
    return NDList.filled(shape, NumNDList.zero());
  }

  /// Creates a new NDList with the same shape as the provided NDList and filled with zeros of the specified type.
  static NDList<X> zerosLike<X extends num>(NDList other) {
    return NumNDList.zeros(other.shape);
  }

  /// Creates a new NDList with the provided shape and filled with ones of the specified type.
  static NDList<X> ones<X extends num>(List<int> shape) {
    return NDList.filled(shape, NumNDList.one());
  }

  /// Creates a new NDList with the same shape as the provided NDList and filled with ones of the specified type.
  static NDList<X> onesLike<X extends num>(NDList other) {
    return NumNDList.ones<X>(other.shape);
  }
}

extension ArithmeticNDList<X extends num> on NDList<X> {
  NDList<X> zipWith(NDList<X> other, X Function(X, X) f) {
    if (!_shapeMatches(other)) {
      throw ArgumentError('Shapes do not match');
    }
    final result = NumNDList.zerosLike<X>(this);
    for (var i = 0; i < count; i++) {
      result._list[i] = f(_list[i], other._list[i]);
    }
    return result;
  }

  /// Element-wise addition of two NDLists.
  NDList<X> operator +(NDList<X> other) {
    return this.zipWith(other, ((p0, p1) => (p0 + p1) as X));
  }

  operator -(NDList<X> other) {
    return this.zipWith(other, ((p0, p1) => (p0 - p1) as X));
  }

  operator *(NDList<X> other) {
    return this.zipWith(other, ((p0, p1) => (p0 * p1) as X));
  }

  operator /(NDList<X> other) {
    return this.zipWith(other, ((p0, p1) => (p0 / p1) as X));
  }
}

extension MultiLinear<X> on NDList<NDList<X>> {
  NDList<X> flatten() {
    return NDList.from<X>(_list.expand((element) => element._list).toList());
  }

  /// If
  /// * this `NDList` has `.shape == [a0, .... aM]` and
  /// * every elment is an `NDList<X>` with fixed shape `[b0, ... bN]`,
  ///
  /// then this method returns a new `NDList<X>` with shape `[a0, ... aN, b0, ... bN]`.
  NDList<X> cemented() {
    if (_list.isEmpty) {
      return NDList.empty();
    }
    final cementedShape = [..._shape, ..._list[0].shape];
    final cementedList = _list.expand((element) => element._list).toList();
    return NDList._(cementedList, cementedShape);
  }
}

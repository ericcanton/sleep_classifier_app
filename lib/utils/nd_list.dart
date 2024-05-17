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

  NDList<X> operator [](index) {
    if (_list.isEmpty) {
      throw ArgumentError('Empty NDList, cannot index.');
    }
    if (index is List) {
      // recursively apply indexing
      if (index.length == 1) {
        return this[index[0]];
      }
      return this[index[0]][index.sublist(1)];
    } else if (index is int) {
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
          _list.sublist(index * subLength, (index + 1) * subLength),
          returnShape);
      return theSlice;
    } else if (index is String) {
      if (index == ':') {
        return this;
      } else if (index.contains(':')) {
        final parts = index.split(':');
        final start = parts[0].isEmpty ? 0 : int.parse(parts[0]);
        final end = parts[1].isEmpty ? count : int.parse(parts[1]);
        return slice(start, end);
      } else {
        throw ArgumentError('Invalid index');
      }
    } else {
      throw ArgumentError('Invalid index');
    }
  }

  operator []=(index, X value) {
    // uses similar parsing to [] operator to assign elements of _list
    if (index is List) {
      this[index[0]][index.sublist(1)] = value;
    } else if (index is int) {
      final subLength = _shape[0];
      for (var i = 0; i < subLength; i++) {
        _list[index * subLength + i] = value;
      }
    } else if (index is String) {
      if (index == ':') {
        for (var i = 0; i < count; i++) {
          _list[i] = value;
        }
      } else if (index.contains(':')) {
        final parts = index.split(':');
        final start = parts[0].isEmpty ? 0 : int.parse(parts[0]);
        final end = parts[1].isEmpty ? count : int.parse(parts[1]);
        for (var i = start; i < end; i++) {
          _list[i] = value;
        }
      } else {
        throw ArgumentError('Invalid index');
      }
    } else {
      throw ArgumentError('Invalid index');
    }
  }

  /// Performs a slice along axis 0. Higher axes would be natural to add, but won't be implemented at this time.
  NDList<X> slice(int start, int end) {
    if (start < 0) {
      start += _shape[0];
    }
    if (end < 0) {
      end += _shape[0];
    }
    if (end < start) {
      return this.slice(end, start);
    }
    if (end == start) {
      return NDList._([], [0]);
    }
    final shapeStart = _shape[0] * start;
    final shapeEnd = _shape[0] * end;
    return NDList._(_list.sublist(shapeStart, shapeEnd),
        [end - start, ..._shape.sublist(1)]);
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
    if (count % _product(positiveDims) != 0) {
      throw ArgumentError('New shape must have the same number of elements');
    }
    return NDList._(_list, newShape);
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
      for (var i = 0; i < shape.length; i++) {
        if (shape[i] != other.shape[i]) {
          print("shape mismatch");
          return false;
        }
      }
      for (var i = 0; i < count; i++) {
        if (this._list[i] != other._list[i]) {
          print("element mismatch, $i");
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

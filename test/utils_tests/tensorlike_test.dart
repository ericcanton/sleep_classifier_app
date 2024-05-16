import 'package:test/test.dart';
import 'package:sleep_classifier_app/utils/tensorlike.dart';

void main() {
  group('TensorLike<double>', () {
    test('Create a 1D tensor', () {
      final tensor = TensorLike<double>.fromList([1.0, 2.0, 3.0]);
      expect(tensor.shape, equals([3]));
      expect(tensor[0], equals(1.0));
      expect(tensor[1], equals(2.0));
      expect(tensor[2], equals(3.0));
    });

    test('Create a 2D tensor', () {
      final tensor = TensorLike<double>.fromList2D([
        [1.0, 2.0],
        [3.0, 4.0],
      ]);
      expect(tensor.shape, equals([2, 2]));
      expect(tensor[0][0], equals(1.0));
      expect(tensor[0][1], equals(2.0));
      expect(tensor[1][0], equals(3.0));
      expect(tensor[1][1], equals(4.0));
    });

    test('Access tensor elements using negative indices', () {
      final tensor = TensorLike<double>.fromList([1.0, 2.0, 3.0]);
      expect(tensor[-1], equals(3.0));
      expect(tensor[-2], equals(2.0));
      expect(tensor[-3], equals(1.0));
    });

    test('Slice a tensor', () {
      final tensor = TensorLike<double>.fromList([1.0, 2.0, 3.0, 4.0, 5.0]);
      final slicedTensor = tensor.slice(1, 4);
      expect(slicedTensor.shape, equals([3]));
      expect(slicedTensor[0], equals(2.0));
      expect(slicedTensor[1], equals(3.0));
      expect(slicedTensor[2], equals(4.0));
    });

    test('Reshape a tensor', () {
      final tensor = TensorLike<double>.fromList([1.0, 2.0, 3.0, 4.0, 5.0]);
      final reshapedTensor = tensor.reshape([1, 5]);
      expect(reshapedTensor.shape, equals([1, 5]));
      expect(reshapedTensor[0][0], equals(1.0));
      expect(reshapedTensor[0][1], equals(2.0));
      expect(reshapedTensor[0][2], equals(3.0));
      expect(reshapedTensor[0][3], equals(4.0));
      expect(reshapedTensor[0][4], equals(5.0));
    });
  });
}

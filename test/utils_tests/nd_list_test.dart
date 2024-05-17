import 'package:test/test.dart';
import 'package:sleep_classifier_app/utils/nd_list.dart';

void main() {
  group('NDList<double>', () {
    test('==', () {
      final data = [
        [1.0, 2.0],
        [3.0, 4.0]
      ];
      final ndList = NDList.from<double>(data);
      final ndList0 = NDList.from<double>(data[0]);

      expect(ndList, equals(ndList));
      expect(ndList0, equals(ndList0));
    });

    test('1d Indexing with int', () {
      final data = [91.0, 92.0, 94.0];
      final ndList = NDList.from<double>(data);

      for (var i = 0; i < 3; i++) {
        expect(ndList[i].shape, equals([1]));
        expect(ndList[i].item, equals(data[i]));
      }
    });

    test('2d Indexing with int', () {
      final data = [
        [1.0, 2.0, 4.0],
        [3.0, 4.0, 16.0]
      ];
      final ndList = NDList.from<double>(data);
      final ndList0 = NDList.from<double>(data[0]);

      expect(ndList[0], equals(ndList0));
    });

    test('.item', () {
      final data = [1.0];
      final ndList = NDList.from<double>(data);

      expect(ndList.item, equals(1.0));
    });

    test('Create NDList from List<double>', () {
      final data = [1.0, 2.0, 3.0, 4.0];
      final ndList = NDList.from<double>(data);

      expect(ndList.shape, equals([4]));
      expect(ndList[0].item, equals(1.0));
      expect(ndList[1].item, equals(2.0));
      expect(ndList[2].item, equals(3.0));
      expect(ndList[3].item, equals(4.0));
    });

    test('Create NDList from nested List<double>', () {
      final data = [
        [1.0, 2.0],
        [3.0, 4.0]
      ];
      final ndList = NDList.from<double>(data);
      final ndList0 = NDList.from<double>(data[0]);
      final ndList1 = NDList.from<double>(data[1]);

      expect(ndList.shape, equals([2, 2]));

      expect(ndList0.shape, equals([2]));
      expect(ndList[0].shape, equals([2]));
      expect(ndList[0], equals(ndList0));
      expect(ndList[0][0].item, equals(1.0));
      expect(ndList[0][1].item, equals(2.0));

      expect(ndList1.shape, equals([2]));
      expect(ndList[1].shape, equals([2]));
      expect(ndList[1], equals(ndList1));
      expect(ndList[1][0].item, equals(3.0));
      expect(ndList[1][1].item, equals(4.0));
    });

    test('zeros', () {
      final shape = [3, 2];
      final ndList = NumNDList.zeros<double>(shape);

      expect(ndList.shape, equals(shape));
      for (var i = 0; i < shape[0]; i++) {
        for (var j = 0; j < shape[1]; j++) {
          expect(ndList[i][j].item, equals(0.0));
        }
      }
    });

    test('zeros', () {
      final shape = [3, 2];
      final ndList = NumNDList.zeros<double>(shape);

      expect(ndList.shape, equals(shape));
      for (var i = 0; i < shape[0]; i++) {
        for (var j = 0; j < shape[1]; j++) {
          expect(ndList[i][j].item, equals(0.0));
        }
      }
    });

    test('zerosLike', () {
      final data1x4 = [1.0, 2.0, 3.0, 4.0];
      final ndList1x4 = NDList.from<double>(data1x4);
      final data3x2 = [
        [1.0, 2.0, 4.0],
        [3.0, 4.0, 16.0]
      ];
      final ndList3x2 = NDList.from<double>(data3x2);

      final zerosLike1x4 = NumNDList.zerosLike<double>(ndList1x4);
      final zerosLike3x2 = NumNDList.zerosLike<double>(ndList3x2);

      expect(zerosLike1x4.shape, equals([4]));
      expect(zerosLike1x4[0].item.runtimeType, double);
      for (var i = 0; i < 4; i++) {
        expect(zerosLike1x4[i].item, equals(0.0));
      }

      expect(zerosLike3x2.shape, equals([2, 3]));

      for (var i = 0; i < 2; i++) {
        for (var j = 0; j < 3; j++) {
          expect(zerosLike3x2[i][j].item, equals(0.0));
        }
      }
    });

    test('Access elements using slicing', () {
      final data = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
      ];
      final expectedSliceData = [
        [4.0, 5.0],
        [7.0, 8.0]
      ];
      final ndList = NDList.from<double>(data);
      final expectedSlice = NDList.from<double>(expectedSliceData);

      final sliced = ndList[['1:3', '0:2']];

      expect(sliced.shape, equals([2, 2]));
      expect(sliced, equals(expectedSlice));
      // expect(sliced[0][0], equals(4.0));
      // expect(sliced[0][1], equals(5.0));
      // expect(sliced[1][0], equals(7.0));
      // expect(sliced[1][1], equals(8.0));
    });

    test('Perform element-wise operations', () {
      final data1 = [
        [1.0, 2.0],
        [3.0, 4.0]
      ];
      final data2 = [
        [2.0, 3.0],
        [4.0, 5.0]
      ];
      final ndList1 = NDList.from<double>(data1);
      final ndList2 = NDList.from<double>(data2);

      final sum = ndList1 + ndList2;
      final product = ndList1 * ndList2;

      expect(sum.shape, equals([2, 2]));
      expect(sum[0][0], equals(3.0));
      expect(sum[0][1], equals(5.0));
      expect(sum[1][0], equals(7.0));
      expect(sum[1][1], equals(9.0));

      expect(product.shape, equals([2, 2]));
      expect(product[0][0], equals(2.0));
      expect(product[0][1], equals(6.0));
      expect(product[1][0], equals(12.0));
      expect(product[1][1], equals(20.0));
    });
  });
}

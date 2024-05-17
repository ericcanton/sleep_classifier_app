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
      final ndList0Second = NDList.from<double>([1.0, 2.0]);

      // check they equal themselves
      expect(ndList, equals(ndList));
      expect(ndList0, equals(ndList0));
      expect(ndList0Second, equals(ndList0));
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
      for (int j = 0; j < 2; j++) {
        for (var i = 0; i < 3; i++) {
          expect(ndList[[j, i]].item, equals(data[j][i]));
        }
      }
    });

    test('2d Indexing with List<int>', () {
      final data = [
        [1.0, 2.0, 4.0],
        [3.0, 4.0, 16.0]
      ];
      final ndList = NDList.from<double>(data);

      expect(ndList[[0, 1]].item, equals(2.0));
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
        [1.0, 2.0, 4.0],
        [3.0, 4.0, 16.0]
      ];
      final ndList = NDList.from<double>(data);

      expect(ndList.shape, equals([2, 3]));

      // 1d data
      final ndList0 = NDList.from<double>(data[0]);
      expect(ndList0.shape, equals([data[0].length]));
      expect(ndList[0], equals(ndList0));
      expect(ndList[0][0].item, equals(1.0));
      expect(ndList[0][1].item, equals(2.0));

      // trivially 2-dim [1, N]
      final ndList0Wrapped = NDList.from<double>([data[0]]);
      expect(ndList0Wrapped.shape, equals([1, 3]));
      expect(ndList0Wrapped[0], equals(ndList0));
      expect(ndList0Wrapped[0][0].item, equals(1.0));
      expect(ndList0Wrapped[0][1].item, equals(2.0));
    });

    test('zeros', () {
      final shape = [3, 2];
      final ndList = NumNDList.zeros<double>(shape);

      expect(ndList.shape, equals(shape));
      for (var i = 0; i < shape[0]; i++) {
        expect(ndList[i].shape, equals([2]));
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
      final data2x3 = [
        [1.0, 2.0, 4.0],
        [3.0, 4.0, 16.0]
      ];
      final ndList2x3 = NDList.from<double>(data2x3);

      final zerosLike1x4 = NumNDList.zerosLike<double>(ndList1x4);
      final zerosLike2x3 = NumNDList.zerosLike<double>(ndList2x3);

      expect(zerosLike1x4.shape, equals([4]));
      expect(zerosLike1x4[0].item.runtimeType, double);
      for (var i = 0; i < 4; i++) {
        expect(zerosLike1x4[i].item, equals(0.0));
      }

      expect(zerosLike2x3.shape, equals([2, 3]));

      for (var i = 0; i < 2; i++) {
        for (var j = 0; j < 3; j++) {
          expect(zerosLike2x3[i][j].item, equals(0.0));
        }
      }
    });

    test('Slicing once along axis 0, both end points given', () {
      final data = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0]
      ];
      final ndList = NDList.from<double>(data);
      final sliced02 = ndList['0:2'];
      final sliced13 = ndList['1:3'];
      final sliced24 = ndList['2:4'];

      final parts = ":".split(':');
      expect(parts.length, equals(2));

      final sliceShape = [2, 3];

      expect(sliced02.shape, equals(sliceShape));
      expect(sliced02, equals(NDList.from<double>(data.sublist(0, 2))));
      expect(sliced13.shape, equals(sliceShape));
      expect(sliced13, equals(NDList.from<double>(data.sublist(1, 3))));
      expect(sliced24.shape, equals(sliceShape));
      expect(sliced24, equals(NDList.from<double>(data.sublist(2, 4))));
    });

    test('Slicing once along axis 0, only one point given', () {
      final data = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0]
      ];
      final ndList = NDList.from<double>(data);
      final slicedTil2 = ndList[':2'];
      final slicedFrom2 = ndList['2:'];

      final sliceShape = [2, 3];

      expect(slicedTil2.shape, equals(sliceShape));
      expect(slicedFrom2.shape, equals(sliceShape));

      expect(slicedTil2, equals(NDList.from<double>(data.sublist(0, 2))));
      expect(
          slicedFrom2,
          equals(NDList.from<double>(data.sublist(
            2,
          ))));
    });

    test('Shape of 3D NDList slice', () {
      final testND = NDList.filled([2, 4, 3], 0.0);

      final testSlice = testND[[':', '1:3', ':']];

      expect(testSlice.shape, equals([2, 2, 3]));
    });

    test('Slicing once along axis 1', () {
      final data = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0]
      ];

      final expectedSliceData = [
        [1.0, 2.0],
        [4.0, 5.0],
        [7.0, 8.0],
        [10.0, 11.0]
      ];
      final ndList = NDList.from<double>(data);
      final slicedTil2 = ndList.slice(0, 2, axis: 1);

      final sliceShape = [4, 2];

      expect(slicedTil2.shape, equals(sliceShape));
      expect(slicedTil2, NDList.from<double>(expectedSliceData));
    });

    test('Sum', () {
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

      expect(sum.shape, equals([2, 2]));

      for (var i = 0; i < 2; i++) {
        for (var j = 0; j < 2; j++) {
          expect(sum[[i, j]].item, equals(data1[i][j] + data2[i][j]));
        }
      }
    });
    test('Product', () {
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

      final sum = ndList1 * ndList2;

      expect(sum.shape, equals([2, 2]));

      for (var i = 0; i < 2; i++) {
        for (var j = 0; j < 2; j++) {
          expect(sum[i][j].item, equals(data1[i][j] * data2[i][j]));
        }
      }
    });

    test('Test .cement() for 1x1s', () {
      final data = [
        [1.0, 2.0, 4.0],
        [3.0, 4.0, 16.0]
      ];
      final ndList = NDList.from<double>(data);

      final subNDs = <NDList<double>>[];
      for (int i = 0; i < data.length; i++) {
        for (int j = 0; j < data[i].length; j++) {
          subNDs.add(NDList.from<double>([data[i][j]]));
        }
      }

      final ndOfNDs = NDList.from<NDList<double>>(subNDs);

      expect(ndOfNDs.shape, equals([data.length * data[0].length]));

      final cemented = ndOfNDs.reshape([2, 3]).cemented();

      // note: this works a little strangely for a cement of 1x1s
      expect(cemented.shape, equals([2, 3, 1]));
      expect(cemented.reshape([2, 3]), equals(ndList));
    });

    test('Test .cement() for 1x2s', () {
      final ndLists = [
        for (int i = 0; i < 3 * 2; i++) NDList.from<double>([1.0, 2.0])
      ];

      final ndOfNDs = NDList.from<NDList<double>>(ndLists);

      final cemented = ndOfNDs.reshape([3, 2]).cemented();

      expect(cemented.shape, equals([3, 2, 2]));
    });
  });
}

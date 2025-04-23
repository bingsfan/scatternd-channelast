# 运行结果

root@k230:/yolo-test/smh-channellast/scatterNDTest# ./output/main
scatternd-channellast测试结果：  
3x2x1-1 channellast test average time: 5.21155 ms (over 20 runs)
3x2x1-2 channellast test average time: 17.1691 ms (over 20 runs)
3x3x3 channellast test average time: 5.5058 ms (over 20 runs)
3x4x3 channellast test average time: 41.3303 ms (over 20 runs)
4x5x4-1 channellast test average time: 6.859 ms (over 20 runs)
4x5x4-2 channellast test average time: 6.8793 ms (over 20 runs)

root@k230:/yolo-test/smh-channellast/scatterNDTest# ./output/main
transfer测试结果
3x2x1-1 channellast test average time: 5.33175 ms (over 20 runs)
3x2x1-2 channellast test average time: 18.0245 ms (over 20 runs)
3x3x3 channellast test average time: 7.7676 ms (over 20 runs)
3x4x3 channellast test average time: 45.931 ms (over 20 runs)
4x5x4-1 channellast test average time: 9.18125 ms (over 20 runs)
4x5x4-2 channellast test average time: 9.40175 ms (over 20 runs)

直接channel-last测试结果
3x2x1-1 channellast test average time: 5.1528 ms (over 20 runs)
3x2x1-2 channellast test average time: 17.3249 ms (over 20 runs)
3x3x3 channellast test average time: 6.71945 ms (over 20 runs)
3x4x3 channellast test average time: 40.786 ms (over 20 runs)
4x5x4-1 channellast test average time: 7.8387 ms (over 20 runs)
4x5x4-2 channellast test average time: 7.47195 ms (over 20 runs)

---
root@k230:/yolo-test/smh-channellast/scatterNDTest# ./output/scatternd0423
scatternd_transfer_testcases:
3x2x1-1 channellast test average time: 5.4062 ms (over 20 runs)
  print_and_convert_4d_block_to_hwc avg: 1.76904 ms
  read_and_convert_block_to_hwc avg: 0.954541 ms
3x2x1-2 channellast test average time: 18.3729 ms (over 20 runs)
  print_and_convert_4d_block_to_hwc avg: 5.69772 ms
  read_and_convert_block_to_hwc avg: 3.58067 ms
3x3x3 channellast test average time: 7.09205 ms (over 20 runs)
  print_and_convert_4d_block_to_hwc avg: 4.98232 ms
  read_and_convert_block_to_hwc avg: 1.52486 ms
3x4x3 channellast test average time: 46.4263 ms (over 20 runs)
  print_and_convert_4d_block_to_hwc avg: 18.6598 ms
  read_and_convert_block_to_hwc avg: 12.2158 ms
4x5x4-1 channellast test average time: 8.73215 ms (over 20 runs)
  print_and_convert_4d_block_to_hwc avg: 4.73456 ms
  read_and_convert_block_to_hwc avg: 2.83822 ms
4x5x4-2 channellast test average time: 9.49305 ms (over 20 runs)
  print_and_convert_4d_block_to_hwc avg: 5.43598 ms
  read_and_convert_block_to_hwc avg: 2.82703 ms
scatternd_channellast_test_cases:
3x2x1-1 channellast test average time: 5.15005 ms (over 20 runs)
  print_block_time avg: 1.3459 ms
  read_block_time avg: 0.914476 ms
3x2x1-2 channellast test average time: 17.3019 ms (over 20 runs)
  print_block_time avg: 3.67962 ms
  read_block_time avg: 3.45383 ms
3x3x3 channellast test average time: 5.515 ms (over 20 runs)
  print_block_time avg: 3.83138 ms
  read_block_time avg: 1.37088 ms
3x4x3 channellast test average time: 40.9963 ms (over 20 runs)
  print_block_time avg: 11.7045 ms
  read_block_time avg: 11.4854 ms
4x5x4-1 channellast test average time: 7.25505 ms (over 20 runs)
  print_block_time avg: 4.16016 ms
  read_block_time avg: 2.62016 ms
4x5x4-2 channellast test average time: 6.88125 ms (over 20 runs)
  print_block_time avg: 3.04451 ms
  read_block_time avg: 2.64826 ms

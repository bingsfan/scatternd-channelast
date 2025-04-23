#include "mytestutil.h"
#include "testutil.h"
#include <functional>
#include <iostream>
#include <map>
#include <queue>
#include <sys/time.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
using namespace std;
std::map<std::string, double> convert_time_accumulator;
// 性能测试函数
void time_test_scatter(const string &test_name, function<void()> test_func) {
	struct timeval start, end;
	const int runs = 20;	// 运行20次取平均值

	gettimeofday(&start, NULL);
	for(int i = 0; i < runs; i++) {
		test_func();
	}
	gettimeofday(&end, NULL);

	float time_use = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
	float avg_time = time_use / 1000 / runs;	// 转换为毫秒并计算平均值

	cout << test_name << " average time: " << avg_time << " ms (over " << runs << " runs)"
		 << endl;
	for(const auto &entry : convert_time_accumulator) {
		cout << "  " << entry.first << " avg: " << entry.second / runs << " ms" << endl;
	}
	convert_time_accumulator.clear();
}

void scatternd_channellast_test_cases() {
#if 1
	time_test_scatter("3x2x1-1 channellast test", []() {
		test_scatterND3x2x1channellast(
		 { 4, 4, 48 }, { 768, 3 }, { 768 }, "input/3_2_1_data48/input_data-channellast16.txt",
		 "input/3_2_1_data48/myindices.txt", "input/3_2_1_data48/updates.txt",
		 "input/3_2_1_data48/myoutput-mychan16.txt", 0, 16);
	});
#endif
#if 1
	time_test_scatter("3x2x1-2 channellast test", []() {
		test_scatterND3x2x1channellast({ 8, 8, 48 }, { 3072, 3 }, { 3072 },
									   "input/3_2_1_data48_8/input_data-channellast16.txt",
									   "input/3_2_1_data48_8/indices.txt",
									   "input/3_2_1_data48_8/updates.txt",
									   "input/3_2_1_data48_8/output-mychan16.txt", 0, 16);
	});
#endif
#if 1
	time_test_scatter("3x3x3 channellast test", []() {
		test_scatterND3x3x3channellast(
		 { 32, 4, 16 }, { 1, 5, 2 }, { 1, 5, 4 },
		 "input/3_3_3_data1_32/input_data-channellast16.txt",
		 "input/3_3_3_data1_32/indices.txt", "input/3_3_3_data1_32/updates.txt",
		 "input/3_3_3_data1_32/myoutput-channellast16.txt", 0, 16);
	});
#endif

#if 1
	time_test_scatter("3x4x3 channellast test", []() {
		test_scatterND3x4x3channellast({ 96, 7, 16 }, { 16, 48, 7, 3 }, { 16, 48, 7 },
									   "input/3_4_3_data_16/input_data-channellast4.txt",
									   "input/3_4_3_data_16/indices.txt",
									   "input/3_4_3_data_16/updates.txt",
									   "input/3_4_3_data_16/myoutput-channellast4.txt", 0, 4);
	});
#endif

#if 1
	time_test_scatter("4x5x4-1 channellast test", []() {
		test_scatterND4x5x4channellast({ 1, 48, 1, 48 }, { 1, 6, 6, 1, 4 }, { 1, 6, 6, 1 },
									   "input/4_5_4_data1/input_data-channellast16.txt",
									   "input/4_5_4_data1/indices.txt",
									   "input/4_5_4_data1/updates.txt",
									   "input/4_5_4_data1/myoutput-channellast16.txt", 0, 16);
	});
#endif
#if 1
	time_test_scatter("4x5x4-2 channellast test", []() {
		test_scatterND4x5x4channellast({ 1, 48, 1, 48 }, { 1, 6, 36, 1, 4 }, { 1, 6, 36, 1 },
									   "input/4_5_4_data2/input_data-channellast16.txt",
									   "input/4_5_4_data2/indices.txt",
									   "input/4_5_4_data2/updates.txt",
									   "input/4_5_4_data2/myoutput-channellast16.txt", 0, 16);
	});
#endif
}
void scatternd_transfer_testcases() {
#if 1
	time_test_scatter("3x2x1-1 channellast test", []() {
		test_scatterND_transfer(
		 { 4, 4, 48 }, { 768, 3 }, { 768 }, "input/3_2_1_data48/input_data-channellast16.txt",
		 "input/3_2_1_data48/myindices.txt", "input/3_2_1_data48/updates.txt",
		 "input/3_2_1_data48/transfer-mychan16.txt", 0, 16);
	});
#endif
#if 1
	time_test_scatter("3x2x1-2 channellast test", []() {
		test_scatterND_transfer({ 8, 8, 48 }, { 3072, 3 }, { 3072 },
								"input/3_2_1_data48_8/input_data-channellast16.txt",
								"input/3_2_1_data48_8/indices.txt",
								"input/3_2_1_data48_8/updates.txt",
								"input/3_2_1_data48_8/transfer-mychan16.txt", 0, 16);
	});
#endif
#if 1
	time_test_scatter("3x3x3 channellast test", []() {
		test_scatterND_transfer({ 32, 4, 16 }, { 1, 5, 2 }, { 1, 5, 4 },
								"input/3_3_3_data1_32/input_data-channellast16.txt",
								"input/3_3_3_data1_32/indices.txt",
								"input/3_3_3_data1_32/updates.txt",
								"input/3_3_3_data1_32/transfer-channellast16.txt", 0, 16);
	});
#endif

#if 1
	time_test_scatter("3x4x3 channellast test", []() {
		test_scatterND_transfer({ 96, 7, 16 }, { 16, 48, 7, 3 }, { 16, 48, 7 },
								"input/3_4_3_data_16/input_data-channellast4.txt",
								"input/3_4_3_data_16/indices.txt",
								"input/3_4_3_data_16/updates.txt",
								"input/3_4_3_data_16/transfer-channellast4.txt", 0, 4);
	});
#endif

#if 1
	time_test_scatter("4x5x4-1 channellast test", []() {
		test_scatterND_transfer({ 1, 48, 1, 48 }, { 1, 6, 6, 1, 4 }, { 1, 6, 6, 1 },
								"input/4_5_4_data1/input_data-channellast16.txt",
								"input/4_5_4_data1/indices.txt",
								"input/4_5_4_data1/updates.txt",
								"input/4_5_4_data1/transfer-channellast16.txt", 0, 16);
	});
#endif
#if 1
	time_test_scatter("4x5x4-2 channellast test", []() {
		test_scatterND_transfer({ 1, 48, 1, 48 }, { 1, 6, 36, 1, 4 }, { 1, 6, 36, 1 },
								"input/4_5_4_data2/input_data-channellast16.txt",
								"input/4_5_4_data2/indices.txt",
								"input/4_5_4_data2/updates.txt",
								"input/4_5_4_data2/transfer-channellast16.txt", 0, 16);
	});
#endif
}
int main(int argc, char *argv[]) {
#if 1
	cout<<"scatternd_transfer_testcases: "<<endl;
	// scatternd_transfer_testcases();
	cout << "scatternd_channellast_test_cases: " << endl;
	scatternd_channellast_test_cases();
#endif
	return 0;
}
#include "mytestutil.h"
#include "testutil.h"
#include <iostream>
#include <map>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>
using namespace std;

int main(int argc, char *argv[])
{

#if 0
    test_scatterND3x2x1({48, 4, 4}, {768, 3}, {768}, "input/3_2_1_data48/input_data.txt",
                        "input/3_2_1_data48/myindices.txt", "input/3_2_1_data48/updates.txt",
                        "input/3_2_1_data48/mycreate-output.txt", 0);
#endif

#if 0
    test_scatterND3x2x1channellast({4, 4, 48}, {768, 3}, {768}, "input/3_2_1_data48/input_data-channellast16.txt",
                                   "input/3_2_1_data48/myindices.txt", "input/3_2_1_data48/updates.txt",
                                   "input/3_2_1_data48/myoutput-mychan16.txt", 0, 16);
#endif
#if 0
    test_scatterND3x3x3channellast({32, 4, 16}, {1, 5, 2}, {1, 5, 4},
                                   "input/3_3_3_data1_32/input_data-channellast16.txt",
                                   "input/3_3_3_data1_32/indices.txt", "input/3_3_3_data1_32/updates.txt",
                                   "input/3_3_3_data1_32/myoutput-channellast16.txt", 0);
#endif
#if 1
    test_scatterND3x4x3channellast({96, 7, 16}, {16, 48, 7, 3}, {16, 48, 7},
                                   "input/3_4_3_data_16/input_data-channellast16.txt",
                                   "input/3_4_3_data_16/indices.txt", "input/3_4_3_data_16/updates.txt",
                                   "input/3_4_3_data_16/myoutput-channellast16.txt", 0,16);
#endif
#if 0
    test_scatterND4x5x4channellast({1, 48, 1, 48}, {1, 6, 6, 1, 4}, {1, 6, 6, 1},
                                   "input/4_5_4_data1/input_data-channellast16.txt", "input/4_5_4_data1/indices.txt",
                                   "input/4_5_4_data1/updates.txt", "input/4_5_4_data1/myoutput-channellast16.txt", 0);
#endif
    return 0;
}
